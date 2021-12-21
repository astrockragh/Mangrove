import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, LeakyReLU, Module, ReLU, Sequential, ModuleList
from torch_geometric.nn import SAGEConv, global_mean_pool, norm, global_max_pool, global_add_pool, MetaLayer
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min
from torch import cat, square,zeros

class MLP(Module):
    def __init__(self, n_in, n_out, hidden=64, nlayers=2, layer_norm=True):
        super().__init__()
        '''Simple two_layer MLP class with ReLU activiation + layernorm to use later'''
        layers = [Linear(n_in, hidden), ReLU()]
        for i in range(nlayers):
            layers.append(Linear(hidden, hidden))
            layers.append(ReLU()) 
        if layer_norm:
            layers.append(LayerNorm(hidden))
        layers.append(Linear(hidden, n_out))
        self.mlp = Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Sage(Module):
    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=1, decode_activation='none', layernorm=True):
        super(Sage, self).__init__()
        '''Model built upon the GraphSAGE convolutional layer. This is a node only model (no global, no edge).
        Model takes a data object from a dataloader in the forward call and takes out the rest itself. 
        hidden_channels, n_in, n_out must be specified
        Most other things can be customized at wish, e.g. activation functions for which ReLU and LeakyReLU can be used'''
        self.encode=encode
        if self.encode:
            self.node_enc = MLP(in_channels, hidden_channels, layer_norm=True)
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        # convolutional layers
        self.convs=[]
        if self.encode:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(int(conv_layers-1)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        #linear layer
        self.decoders=[]
        for i in range(decode_layers-1):
            if layernorm:
                self.decoders.append(LayerNorm(normalized_shape=hidden_channels))
            self.decoders.append(Linear(hidden_channels, hidden_channels))
        if layernorm:
            self.decoders.append(LayerNorm(normalized_shape=hidden_channels)) # layer_norm instead of batch_norm

        self.decoders.append(Linear(hidden_channels, out_channels))
        
        self.convs=ModuleList(self.convs)
        self.decode=Sequential(*self.decoders)
        self.conv_act=self.conv_act_f()
        self.decode_act=self.decode_act_f() ## could apply later

    def conv_act_f(self):
        if self.conv_activation =='relu':
            print('RelU conv activation')
            act = ReLU()
            return act
        if self.conv_activation =='leakyrelu':
            print('LeakyRelU conv activation')
            act=LeakyReLU()
            return act
        if not self.conv_activation:
            raise ValueError("Please specify a conv activation function")

    def decode_act_f(self):
        if self.decode_activation =='relu':
            print('RelU decode activation')
            act = ReLU()
            return act
        if self.decode_activation =='leakyrelu':
            print('LeakyRelU decode activation')
            act=LeakyReLU()
            return act
        if not self.decode_activation:
            print("Please specify a decode activation function")
            return None

    def forward(self, graph):

        #get the data
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        if self.encode:
            x = self.node_enc(x)
        
        #convolutions 
        for conv in self.convs:
            x = conv(x, edge_index)
            x=self.conv_act(x)

        x = global_add_pool(x, batch)  
        #decoder
        x = self.decode(x)

        return x
    

######################################
### Make own MetaLayer-based Class ###
######################################

node_aggregation = scatter_sum  # scatter_mean
global_aggregation = scatter_sum  # scatter_mean

class EdgeModel(Module):
    def __init__(self, hidden):
        super(EdgeModel, self).__init__()
        self.mlp = MLP(hidden * 4, hidden, layer_norm=True)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs. ##what is B??
        # batch: [E] with max entry B - 1.
        cur_state = cat([src, dest, edge_attr, u[batch]], 1)
        return edge_attr + self.mlp(cur_state)


class NodeModel(Module):
    def __init__(self, hidden):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden * 2, hidden, layer_norm=True)
        self.node_mlp_2 = MLP(hidden * 3, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = node_aggregation(out, col, dim=0, dim_size=x.size(0))
        out = cat([x, out, u[batch]], dim=1)
        return x + self.node_mlp_2(out)


class GlobalModel(Module):
    def __init__(self, hidden):
        super(GlobalModel, self).__init__()
        self.global_mlp = MLP(hidden * 2, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = cat([u, global_aggregation(x, batch, dim=0)], dim=1)
        return u + self.global_mlp(out) # do these just add on top of each other

class GlobalModelMulti(Module):
    def __init__(self, hidden):
        super(GlobalModelMulti, self).__init__()
        self.global_mlp = MLP(hidden * 5, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        #takes a series of different global things into consideration
        s=scatter_sum(x, batch, dim=0)
        me=scatter_mean(x, batch, dim=0)
        mi=scatter_min(x, batch, dim=0)[0]
        ma=scatter_max(x, batch, dim=0)[0]
        std=scatter_mean(square(x), batch, dim=0)-square(me)
        concat = cat([u, s, mi, ma, std], dim=1)
        return u + self.global_mlp(concat) ## still a bit in doubt over if this should be a sum

class MetaMulti(Module):
    def __init__(self, hidden_states, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=1, decode_activation='none', layernorm=True):
        super(self.__class__, self).__init__()
        hidden=hidden_states
        n_in=in_channels
        n_out=out_channels
        self.node_enc = MLP(n_in, hidden, layer_norm=True)
        self.edge_enc = MLP(3, hidden, layer_norm=True)
        self.decoder = MLP(hidden, n_out)
        self.ops = ModuleList(
            [
                MetaLayer(edge_model=EdgeModel(hidden), node_model=NodeModel(hidden), global_model=GlobalModelMulti(hidden))
                for _ in range(conv_layers)
            ]
        )
        self.hidden = hidden
        self.norm_out=LayerNorm(normalized_shape=self.hidden)
    def forward(self, graph):
        x = self.node_enc(graph.x)  # Take all feats and encode
        e_feat = graph.x[:,[0,3]] # scale factor and virial mass
        adj = graph.edge_index
        e_encode=cat([graph.edge_attr.view(-1,1), e_feat[adj[0]] - e_feat[adj[1]]], -1)
        
        e = self.edge_enc(e_encode) #put in edge_attr
        # Initialize global features as 0:
        u = zeros(
            graph.batch[-1] + 1, self.hidden, device=x.device, dtype=torch.float32
        )
        batch = graph.batch
        for op in self.ops:
            x, e, u = op(x, adj, e, u, batch)
        x = global_add_pool(x, batch)
        
        out = self.decoder(self.norm_out(x))

        return out
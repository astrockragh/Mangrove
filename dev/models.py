import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm, LeakyReLU
from torch_geometric.nn import SAGEConv, global_mean_pool, norm, global_max_pool, global_add_pool


class Sage(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, out_channels, conv_layers=3, conv_activation='relu', decode_layers=1, decode_activation='none', layernorm=True):
        super(Sage2, self).__init__()
        '''infer input_channels and out_channels from data'''
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm


        self.convs=[]
        self.convs.append(SAGEConv(input_channels, hidden_channels))
        for _ in in range(int(conv_layers-1)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Our final linear layer will define our output
        self.norms=[]
        self.decoders=[]
        for i in range(decode_layers-1):
            if layernorm:
                self.norms.append(LayerNorm(normalized_shape=hidden_channels))
            self.decoders.append(Linear(hidden_channels, hidden_channels))
        self.norms.append(LayerNorm(normalized_shape=hidden_channels)) # layer_norm instead of batch_norm
        self.decoders.append(Linear(hidden_channels, out_channels))
    
    def conv_act(self, x):
            if self.conv_activation=='relu':
                return x.relu()
            if self.conv_activation=='leakyrelu':
                act=LeakyReLU()
                return act(x)
            else:
                raise ValueError("Please specify a conv activation function")
    def decode_act(self, x):
            if self.conv_activation=='relu':
                return x.relu()
            if self.conv_activation=='leakyrelu':
                act=LeakyReLU()
                return act(x)
            else:
                raise ValueError("Please specify a decoder activation function")

    def forward(self, x, edge_index, batch):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.conv_act(x) # activation method here

        x = global_add_pool(x, batch)  


        ##could probably add something non-linear in here

        if self.layernorm:
            for norm, lin in zip(self.norms, self.decoders):
                x=lin(norm(x))
        else:
            for lin in self.decoders:
                x=lin(x)
        x = self.lin1(self.norm(x))
        return x
    
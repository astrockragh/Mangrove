import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, LeakyReLU, Module, ReLU, Sequential, ModuleList
from torch_geometric.nn import SAGEConv, global_mean_pool, norm, global_max_pool, global_add_pool


class Sage(Module):
    def __init__(self, hidden_channels, in_channels, out_channels, conv_layers=3, conv_activation='relu', decode_layers=1, decode_activation='none', layernorm=True):
        super(Sage, self).__init__()
        '''infer input_channels and out_channels from data'''
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.convs=[]
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # self.convs.append(self.conv_act())
        for _ in range(int(conv_layers-1)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            # self.convs.append(self.conv_act())
        # Our final linear layer will define our output
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

    def forward(self, x, edge_index, batch):
        #convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x=self.conv_act(x)

        x = global_add_pool(x, batch)  
        #decoder
        x = self.decode(x)

        return x
    
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm
from torch_geometric.nn import SAGEConv, global_mean_pool, norm, global_max_pool, global_add_pool


class Sage2(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, out_channels, nlin=3):
        super(Sage2, self).__init__()
        
        self.conv1 = SAGEConv(input_channels, hidden_channels) 
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Our final linear layer will define our output
        self.norm = LayerNorm(normalized_shape=hidden_channels) # layer_norm instead of batch_norm
        self.lin1 = Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_add_pool(x, batch)  


        x = self.lin1(self.norm(x))
        return x
    
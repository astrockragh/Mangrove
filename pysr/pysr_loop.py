import torch, pickle, time, os, random
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch_geometric as tg
from torch_geometric.loader import DataLoader
# accelerate huggingface to GPU
if torch.cuda.is_available():
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
from pysr import pysr, best
from tqdm import tqdm
torch.manual_seed(42)
random.seed(42)


print('Loading data')

case='vlarge_all_4t_z0.0_quantile_raw'

datat=pickle.load(open(osp.expanduser(f'~/../../../scratch/gpfs/cj1223/GraphStorage/{case}/data.pkl'), 'rb'))

from torch_geometric.data import Data
data=[]
for d in datat:
    data.append(Data(x=d.x[:,[0,3,4,19,20]], edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y[0]))

try:
    n_targ=len(data[0].y)
except:
    n_targ=1
n_feat=len(data[0].x[0])
n_feat, n_targ

print('Loaded data')

from torch.nn import ReLU, Linear, Module, LayerNorm, Sequential
class MLP(Module):
    def __init__(self, n_in, n_out, hidden=64, nlayers=2, layer_norm=True):
        super().__init__()
        layers = [Linear(n_in, hidden), ReLU()]
        for i in range(nlayers):
            layers.append(Linear(hidden, hidden))
            layers.append(ReLU()) 
        if layer_norm:
            layers.append(LayerNorm(hidden)) #yay
        layers.append(Linear(hidden, n_out))
        self.mlp = Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64, n_feat=5, n_targ=1):
        super(GCN, self).__init__()
        self.g1 = MLP(n_feat, hidden_channels)
        self.g2 = MLP(hidden_channels, hidden_channels)
        self.g3= MLP(hidden_channels, hidden_channels) 
    
        self.f = MLP(hidden_channels, n_targ)
        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.g1(x)
        
        adj = edge_index
        neighbours = x[adj[1]]
        # print(neighbours, adj[0], adj[1])
        N_sum = torch.sum(neighbours, dim=0)
        # print(N_sum, N_sum.shape)
        xe = self.g2(N_sum)

        x = self.g3(xe+x)

        x = global_add_pool(x, batch)

        x = self.f(x)

        return x
    
model = GCN(hidden_channels=64)
next(model.parameters()).is_cuda ##check number one

from sklearn.model_selection import train_test_split
criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()
# criterion = torch.nn.SmoothL1Loss(beta=0.3)


n_epochs=25
n_trials=1
batch_size=int(2**1)
split=0.8
# test_data=data[int(len(data)*split):]
# train_data=data[:int(len(data)*split)]
train_data, test_data=train_test_split(data, test_size=0.2)
l1_lambda = 1e-4
l2_lambda = 0

trains, tests, scatter = [], [], []
yss, preds=[],[]
model = GCN(hidden_channels=64)
train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=1)

test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=0)    
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
print('GPU ', next(model.parameters()).is_cuda)
# Initialize our train function
def train():
    model.train()
    
    for data in tqdm(train_loader, total=len(train_loader)): 
#         print('batch')
        out = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out, data.y.view(-1,1)) 
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())


        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad() 
#     print(loss, l1_norm*l1_lambda, l2_norm*l2_lambda)
 # test function

def test(loader): ##### transform back missing
    model.eval()
    outs = []
    ys = []
    with torch.no_grad(): ##this solves it!!!
        for dat in tqdm(loader, total=len(loader)): 
            
            out = model(dat.x, dat.edge_index, dat.batch) 
            ys.append(dat.y.view(-1,n_targ))
            outs.append(out)
    outss=torch.vstack(outs)
    yss=torch.vstack(ys)
    return torch.std(outss - yss, axis=0), outss, yss


#this uses about 1 GB of memory on the GPU
tr_acc, te_acc = [], []
start=time.time()
for epoch in range(n_epochs):
    print(epoch)
    train()
    if (epoch+1)%2==0:
        train_acc, _ , _ = test(train_loader)
        test_acc, _ , _ = test(test_loader)
        tr_acc.append(train_acc.cpu())
        te_acc.append(test_acc.cpu())
        print(f'Epoch: {epoch+1:03d}, Train scatter: {np.round(train_acc.cpu().numpy(), 4)} \n \
        Test scatter: {np.round(test_acc.cpu().numpy(), 4)}')
stop=time.time()
spent=stop-start
print(f"{spent:.2f} seconds spent training, {spent/n_epochs:.3f} seconds per epoch. Processed {len(data)*split*n_epochs/spent:.0f} trees per second")

print('decoder', sum(p.abs().sum() for p in model.f.parameters())/sum(p.numel() for p in model.f.parameters())*100)

print('encoder', sum(p.abs().sum() for p in model.g.parameters())/sum(p.numel() for p in model.g1.parameters())*100)
print('edge', sum(p.abs().sum() for p in model.g.parameters())/sum(p.numel() for p in model.g2.parameters())*100)
print('both', sum(p.abs().sum() for p in model.g.parameters())/sum(p.numel() for p in model.g3.parameters())*100)


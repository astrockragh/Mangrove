import torch, pickle, time, os, random
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from datetime import date
today = date.today()

today = today.strftime("%d%m%y")

torch.manual_seed(42)
random.seed(42)
def load_data(case, split=0.8):
    data=pickle.load(open(f'../../../../scratch/gpfs/cj1223/GraphStorage/{case}/data.pkl', 'rb'))
    test_data=data[int(len(data)*split):]
    train_data=data[:int(len(data)*split)]
    return train_data, test_data

# test function
def test(loader, model):
    model.eval()

    correct = 0
    for dat in loader: 
        out = model(dat.x, dat.edge_index, dat.batch) 
#         print(out)
        correct += (torch.square(out - dat.y.view(-1,1))).sum() 
    return correct / len(loader.dataset) 
# train loop
def train_percentiles(GNN, case, hidden=32):
    
    '''Input model class and data case'''

    train_data, test_data = load_data(case)
    trains, tests, scatter = [], [], []
    yss, preds=[],[]
    meta={'loss':criterion,
     'n_epochs': n_epochs,
     'batch_size':batch_size,
     'len_data': len(data)}
    for _ in range(n_trials):
        model = GNN(hidden_channels=hidden, in_channels=train_data[0].num_node_features, out_channels=len(np.array([train_data[0].y])))
        train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=1)
        test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=1)    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        # Initialize our train function
        def train():
            model.train()

            for data in train_loader:  
                out = model(data.x, data.edge_index, data.batch)  
                loss = criterion(out, data.y.view(-1,1)) 
                loss.backward()
                optimizer.step() 
                optimizer.zero_grad() 
        tr_acc, te_acc=[],[]
        start=time.time()
        for epoch in range(n_epochs):
            train()

            if (epoch+1)%2==0:
                train_acc = test(train_loader, model).cpu().detach().numpy()
                test_acc = test(test_loader, model).cpu().detach().numpy()
                tr_acc.append(np.sqrt(train_acc))
                te_acc.append(np.sqrt(test_acc))
                print(f'Epoch: {epoch+1:03d}, Train scatter: {np.sqrt(train_acc):.4f}, Test scatter: {np.sqrt(test_acc):.4f}')
        stop=time.time()
        spent=stop-start
        tests.append(te_acc)
        trains.append(tr_acc)
        print(f"{spent:.2f} seconds spent training, {spent/n_epochs:.3f} seconds per epoch. Processed {len(data)*split*n_epochs/spent:.0f} trees per second")
        ys, pred=[],[]
        def test(loader):
            model.eval()

            correct = 0
            for dat in loader: 
                out = model(dat.x, dat.edge_index, dat.batch) 
                pred.append(out.view(1,-1).cpu().detach().numpy())
                ys.append(np.array(dat.y.cpu())) 
        test(test_loader)
        ys=np.hstack(ys)
        pred=np.hstack(pred)[0]
        scatter.append(np.std(ys-pred))
        yss.append(ys)
        preds.append(pred)
        if not osp.exists(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}'):
            os.mkdir(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}')
        
        
        with open(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}/meta.pkl', 'wb') as handle:
            pickle.dump(meta, handle)
        with open(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}/scatter.pkl', 'wb') as handle:
            pickle.dump(scatter, handle)
        with open(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}/tests.pkl', 'wb') as handle:
            pickle.dump(tests, handle)
        with open(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}/trains.pkl', 'wb') as handle:
            pickle.dump(trains, handle)
        with open(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}/yss.pkl', 'wb') as handle:
            pickle.dump(yss, handle)
        with open(f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}/preds.pkl', 'wb') as handle:
            pickle.dump(preds, handle)
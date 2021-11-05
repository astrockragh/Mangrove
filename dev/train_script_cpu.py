import torch, pickle, time, os, random, wandb
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from importlib import __import__
from datetime import date
today = date.today()

today = today.strftime("%d%m%y")
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
def train_percentiles(construct_dict):

    """
    Train a model given a construction dictionairy
    """

    # Setup Log 
    wandblog=construct_dict["wandblog"]
    if wandblog:
        import wandb
        run = wandb.init(project = construct_dict["experiment"], entity = "chri862z", group=construct_dict["group"], config = construct_dict, reinit=True, settings=wandb.Settings(start_method="fork"))
        wandb.run.name = construct_dict['model']+'_'+construct_dict['experiment']+'_'+str(wandb.run.id)
    ## should be wandb.config = {
    #   "learning_rate": 0.001,
    #   "epochs": 100,
    #   "batch_size": 128
}
     meta={'loss':criterion,
     'n_epochs': n_epochs,
     'batch_size':batch_size,
     'len_data': len(data)}

    run_params=construct_dict['run_params']
    hyper_params=construct_dict['hyper_params']
    data_params=construct_dict['data_params']

    if run_params.seed==True:
        torch.manual_seed(42)
        random.seed(42)

    ## load data
    train_data, test_data = load_data(**data_params)
    test_loader=DataLoader(test_data, batch_size=run_params.batch_size, shuffle=0)    ##never shuffke

    trains, tests, scatter = [], [], []
    yss, preds=[],[]


    for _ in range(run_params.n_trials):
        hyper_params.in_channels=train_data[0].num_node_features
        hyper_params.out_channels=len(np.array([train_data[0].y]))
        model = GNN(**hyper_params)
        train_loader=DataLoader(train_data, batch_size=run_params.batch_size, shuffle=run_params.shuffle) ## control shuffle

        
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

################################
#      Load dependencies       #
################################

def get_metrics(metric_name):
    '''Returns functions to scrutinize performance on the fly'''
    import dev.metrics as metrics
    metrics=getattr(metrics, metric_name)
    return metrics

################################
# Custom loss not relevant yet #
################################

# def get_loss_func(name):
#     # Return loss func from the loss functions file given a function name
#     import dev.loss_funcs as loss_func_module
#     loss_func = getattr(loss_func_module, name)
#     return loss_func


def get_performance(name):
    '''Returns plotting functions to scrutinize performance on the fly'''
    import dev.eval_model as evals
    performance_plot = getattr(evals, name)
    return performance_plot 

def setup_model(construct_dict, train_data):
    # Retrieve name and params for construction
    model_name    = construct_dict['model']
    hyper_params  = construct_dict['hyper_params']
    experiment    = construct_dict['experiment']
    hyper_params.in_channels=train_data[0].num_node_features
    hyper_params.out_channels=len(np.array([train_data[0].y]))
    # Load model from model folder
    import dev.models as models
    model         = getattr(models, model_name) 
    model         = model(**hyper_params)

    # Make folder for saved states

    model_path    = osp.join(f'../../../../../scratch/gpfs/cj1223/GraphStorage/', experiment, "trained_model") ##!!!!!!! needs to point to the right spot
    if not osp.isdir(model_path):
        os.makedirs(model_path)

    return model, model_path
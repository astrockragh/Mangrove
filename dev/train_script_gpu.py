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

### test function
def test(loader, model):
    '''returns targets and predictions'''
    ys, pred,xs, Mh=[],[],[], []
    model.eval()
    for dat in loader: 
        out = model(dat.x, dat.edge_index, dat.batch) 
        pred.append(out.view(1,-1).cpu().detach().numpy())
        ys.append(np.array(dat.y.cpu())) 
        u, counts = np.unique(dat.batch, return_counts=1)
        xs.append(list(torch.tensor_split(dat.x, torch.cumsum(torch.tensor(counts[:-1]),0))))
        ## compile lists
        ys=np.hstack(ys)
        pred=np.hstack(pred)[0]
        xs=np.hstack(xs)
        xn=[]
        for x in xs:
            x0=x.cpu().detach().numpy()
            xn.append(x0)
            Mh.append(x0[0][3])
        return ys,pred,xn, Mh


# train loop
def train_model(construct_dict):

    """
    Train a model given a construction dictionairy
    """
    case=construct_dict['data_params']['case']
    pointer=f'../../../../../scratch/gpfs/cj1223/GraphStorage/{case}/results_{today}'
    if not osp.exists(pointer):
            os.mkdir(pointer)
    # Setup Log 
    wandblog=construct_dict["wandblog"]
    if wandblog:
        import wandb
        run = wandb.init(project = construct_dict["experiment"], entity = "chri862z", group=construct_dict["group"], config = construct_dict, reinit=True, settings=wandb.Settings(start_method="fork"))
        wandb.run.name = construct_dict['model']+'_'+construct_dict['experiment']+'_'+str(wandb.run.id)

    run_params=construct_dict['run_params']
    data_params=construct_dict['data_params']

    if run_params.seed==True:
        torch.manual_seed(42)
        random.seed(42)

    ## load data
    train_data, test_data = load_data(**data_params)
    test_loader=DataLoader(test_data, batch_size=run_params.batch_size, shuffle=0)    ##never shuffke


    ### train
    n_trials=run_params.n_trials
    n_epochs=run_params.n_epoch
    val_epoch=run_params.val_epoch
    batch_size=run_params.batch_size
    shuffle=run_params.shuffle
    lr=run_params.learning_rate
    save=run_params.save


    # lr_schedule           = get_lr_schedule(construct_dict) 
    loss_func            = get_loss_func(construct_dict['run_params']['loss_func'])
    metric            = get_metrics(construct_dict['run_params']['metrics'])
    performance_plot      = get_performance(construct_dict['run_params']['performance_plot'])

    train_accs, test_accs, scatter, Mhs = [], [], [], []
    yss, preds, xss, lowest = [], [], [], []
    for _ in range(n_trials):
        lowest_metric=np.inf
        model = setup_model(**construct_dict)
        if save:  # Make folder for saved states
            model_path    = osp.join(pointer, wandb.run.name, "trained_model") ##!!!!!!! needs to point to the right spot
            if not osp.isdir(model_path):
                os.makedirs(model_path)
                print('Made folder for saving model')

        train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=shuffle) ## control shuffle

        optimizer = torch.optim.Adam(model.parameters(), lr=lr) ## need a lr schedule

        # Initialize our train function
        def train():
            model.train()
            return_loss=0
            for data in train_loader:  
                out = model(data.x, data.edge_index, data.batch)  
                loss = loss_func(out, data.y.view(-1,1))
                return_loss+=loss
                loss.backward()
                optimizer.step() 
                optimizer.zero_grad() 
            return return_loss/len(train_loader.dataset)
        tr_acc, te_acc=[],[]
        start=time.time()
        ## do a tqdm wrapper
        for epoch in range(n_epochs):
            trainloss=train()

            if (epoch+1)%val_epoch==0:
                train_metric = metric(train_loader, model)
                test_metric = metric(test_loader, model)
                if test_metric<lowest_metric:
                    lowest_metric=test_metric
                tr_acc.append(train_metric)
                te_acc.append(test_metric)

                if wandblog:
                    wandb.log({"Epoch":  epoch ,
                            "Training scatter": trainloss, 
                            "Training scatter": train_metric, 
                            "Test scatter":   test_metric,
                            "Learning rate":   lr})
                print(f'Epoch: {int(epoch+1)}, Train: {train_metric:.4f}, Test scatter: {test_metric:.4f}, Lowest was {lowest_metric:.4f}')
                if (epoch+1)%(int(val_epoch*5))==0 and wandblog:
                    ys, pred, xs, Mh = test(test_loader, model)
                    fig=performance_plot(ys,pred, xs, Mh)
                    title="performanceplot_"+str(epoch)
                    wandb.log({title: [wandb.Image(fig, caption=title)]})
        stop=time.time()
        spent=stop-start
        test_accs.append(te_acc)
        train_accs.append(tr_acc)
        print(f"{spent:.2f} seconds spent training, {spent/n_epochs:.3f} seconds per epoch. Processed {len(train_loader.dataset)*n_epochs/spent:.0f} trees per second")
        
        ys, pred, xs, Mh = test(test_loader, model)
        fig=performance_plot(ys,pred, xs, Mh)
        
        scatter.append(np.std(ys-pred))
        yss.append(ys)
        preds.append(pred)
        xss.append(xs)
        lowest.append(lowest_metric)
        Mhs.append(Mh)
    result_dict={'sigma':scatter,
    'test_acc': test_accs,
    'train_acc': train_accs,
    'ys': yss,
    'pred': pred,
    'xs': xss,
    'Mh': Mhs}
    with open(f'{pointer}/result_dict.pkl', 'wb') as handle:
        pickle.dump(result_dict, handle)




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

def get_loss_func(name):
    # Return loss func from the loss functions file given a function name
    import dev.loss_funcs as loss_func_module
    loss_func = getattr(loss_func_module, name)
    return loss_func


def get_performance(name):
    '''Returns plotting functions to scrutinize performance on the fly'''
    import dev.eval_plot as evals
    performance_plot = getattr(evals, name)
    return performance_plot 

def setup_model(construct_dict, train_data):
    # Retrieve name and params for construction
    model_name    = construct_dict['model']
    hyper_params  = construct_dict['hyper_params']

    hyper_params['in_channels']=train_data[0].num_node_features
    hyper_params['out_channels']=len(np.array([train_data[0].y]))

    # Load model from model folder
    import dev.models as models
    model         = getattr(models, model_name) 
    model         = model(**hyper_params)

    return model
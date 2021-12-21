import torch, pickle, time, os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from importlib import __import__

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if torch.cuda.is_available():
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device

from datetime import date
today = date.today()

today = today.strftime("%d%m%y")
def load_data(case, split=0.8):
    data=pickle.load(open(osp.expanduser(f'~/../../scratch/gpfs/cj1223/GraphStorage/{case}/data.pkl'), 'rb'))
    test_data=data[int(len(data)*split):]
    train_data=data[:int(len(data)*split)]
    return train_data, test_data


def make_id(length=6):
    import random
    import string
    # choose from all lowercase letters
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

### test function
def test(loader, model):
    '''returns targets and predictions'''
    ys, pred,xs, Mh=[],[],[], []
    model.eval()
    with torch.no_grad():
        for dat in loader: 
            out = model(dat) 
            pred.append(out.view(1,-1).cpu().detach().numpy())
            ys.append(np.array(dat.y.cpu().numpy())) 
            u, counts = np.unique(dat.batch.cpu().numpy(), return_counts=1)
            xs.append(np.array(torch.tensor_split(dat.x.cpu(), torch.cumsum(torch.tensor(counts[:-1]),0)), dtype=object))
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
    run_params=construct_dict['run_params']
    data_params=construct_dict['data_params']
    learn_params=construct_dict['learn_params']
    case=data_params['case']
    group=construct_dict['group']
    pointer=osp.expanduser(f'~/../../scratch/gpfs/cj1223/GraphResults/results_{group}_{today}')
    if not osp.exists(pointer):
        try:
            os.makedirs(pointer)
        except:
            print('Folder already exists')
        # Setup Log 
    log=construct_dict["log"]


    print(construct_dict)

    if run_params['seed']==True:
        torch.manual_seed(42)

    ### train
    n_trials=run_params['n_trials']
    n_epochs=run_params['n_epochs']
    val_epoch=run_params['val_epoch']
    batch_size=run_params['batch_size']
    save=run_params['save']
    early_stopping=run_params['early_stopping']
    patience=run_params['patience']
    num_workers=run_params['num_workers']


    lr=learn_params['learning_rate']
    if learn_params['warmup']:
        lr=lr/(learn_params['g_up'])**(learn_params['warmup'])

    ## load data
    train_data, test_data = load_data(**data_params)

    try:
        n_targ=len(train_data[0].y)
    except:
        n_targ=1
    n_feat=len(train_data[0].x[0])

    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=0, num_workers=num_workers)    ##never shuffle test
    construct_dict['hyper_params']['in_channels']=n_feat
    construct_dict['hyper_params']['out_channels']=n_targ

    ### learning related stuff ###
    lr_scheduler          = get_lr_schedule(construct_dict) 
    loss_func            = get_loss_func(construct_dict['run_params']['loss_func'])
    l1_lambda=run_params['l1_lambda']
    l2_lambda=run_params['l2_lambda']


    metric            = get_metrics(construct_dict['run_params']['metrics'])
    performance_plot      = get_performance(construct_dict['run_params']['performance_plot'])

    train_accs, test_accs, scatter, = [], [], []
    preds,lowest, epochexit = [], [], []
    run_name = construct_dict['model']+f'_{case}'+f'_{make_id()}'
    log_dir_glob=osp.join(pointer, run_name)
    for trial in range(n_trials):
        if n_trials>1:
            run_name_n=run_name+f'_{trial+1}_{n_trials}'
        else:
            run_name_n=run_name
        log_dir=osp.join(pointer, run_name_n)
        if log:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(log_dir=log_dir)
        lowest_metric=np.array([np.inf]*n_targ)
        model = setup_model(construct_dict['model'], construct_dict['hyper_params'])
        if save:  # Make folder for saved states
            model_path    = osp.join(log_dir, "trained_model") ##!!!!!!! needs to point to the right spot
            if not osp.isdir(model_path):
                os.makedirs(model_path)
                print('Made folder for saving model')

        train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler=lr_scheduler(optimizer, **learn_params)

        _, _, test_loader = accelerator.prepare(model, optimizer, test_loader)
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        # Initialize our train function
        def train(epoch):
            model.train()
            return_loss=0
            for data in train_loader:  
                out = model(data)  
                loss = loss_func(out, data.y.view(-1,n_targ))
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
                return_loss+=loss
                accelerator.backward(loss)
                optimizer.step() 
                optimizer.zero_grad()
            # if epoch==0:              #Doesn't work right now but could be fun to add back in
            #     writer.add_graph(model,[data]) 
            return return_loss/len(train_loader.dataset)

        tr_acc, te_acc=[],[]
        early_stop=0
        start=time.time()
        ## do a tqdm wrapper
        for epoch in tqdm(range(n_epochs)):
        
            trainloss=train(epoch)
            #learning rate scheduler step
            scheduler.step(epoch)

            if (epoch+1)%val_epoch==0:
                train_metric = metric(train_loader, model, n_targ)
                test_metric = metric(test_loader, model, n_targ)
                # print(test_metric), print(lowest_metric)
                if np.sum(test_metric)<np.sum(lowest_metric):
                    lowest_metric=test_metric
                    early_stop=0
                    if save:
                        torch.save(model.state_dict(), osp.join(model_path,'model.pt'))
                else:
                    early_stop+=val_epoch
                tr_acc.append(train_metric)
                te_acc.append(test_metric)
                lr0=optimizer.state_dict()['param_groups'][0]['lr']
                last10test=np.median(te_acc[-(10//val_epoch):], axis=0)
                if log:
                    writer.add_scalar('train_loss', trainloss,global_step=epoch+1)
                    if n_targ==1:
                        writer.add_scalar('last10test', last10test, global_step=epoch+1)
                        writer.add_scalar('train_scatter', train_metric,global_step=epoch+1)

                        writer.add_scalar('test_scatter', test_metric, global_step=epoch+1)
                        writer.add_scalar('best_scatter', lowest_metric, global_step=epoch+1)
                    writer.add_scalar('learning_rate', lr0, global_step=epoch+1)

                # if n_targ==1:
                #     print(f'Epoch: {int(epoch+1)} done with learning rate {lr0:.5f}, Train loss: {trainloss:.4f}, Train scatter: {train_metric:.4f}')
                #     print(f'Test scatter: {test_metric:.4f}, Lowest was {lowest_metric:.4f}, Last 10 was {last10test:.4f}, Epochs since improvement {val_epoch*early_stop}')
                # else:
                print(f'Epoch: {int(epoch+1)} done with learning rate {lr0:.5f}, Train loss: {np.round(trainloss.cpu().detach().numpy(),4)}, Train scatter: {np.round(train_metric,2)}')
                print(f'Test scatter: {np.round(train_metric,4)}, Lowest was {np.round(lowest_metric,4)}')
                print(f'Last 10 was {np.round(last10test,4)}, Epochs since improvement {val_epoch*early_stop}')
                if n_targ==1:
                    if (epoch+1)%(int(val_epoch*5))==0 and log:
                        ys, pred, xs, Mh = test(test_loader, model)
                        fig=performance_plot(ys,pred, xs, Mh)
                        writer.add_figure(tag=run_name_n, figure=fig, global_step=epoch+1)
                else:
                    continue
                   ### make multi performance plot

            if early_stopping:
                if early_stop>patience:
                    print(f'Exited after {epoch+1} epochs due to early stopping')
                    epochexit.append(epoch)
                    break
                
        stop=time.time()
        spent=stop-start
        test_accs.append(te_acc)
        train_accs.append(tr_acc)
        print(f"{spent:.2f} seconds spent training, {spent/n_epochs:.3f} seconds per epoch. Processed {len(train_loader.dataset)*n_epochs/spent:.0f} trees per second")
        
        ys, pred, xs, Mh = test(test_loader, model)
        fig=performance_plot(ys,pred, xs, Mh)
        if save:
            fig.savefig(f'{log_dir}/performance_ne{n_epochs}_nt{trial}.png')
        sig=np.std(ys-pred)
        scatter.append(sig)
        preds.append(pred)
        lowest.append(lowest_metric)
        last20=np.median(te_acc[-(20//val_epoch):])
        last10=np.median(te_acc[-(10//val_epoch):])

        metricf={'scatter': sig,
        'lowest':lowest_metric,
        'epoch_exit':epoch,
        'last20':last20,
        'last10':last10}
        paramsf=dict(list(data_params.items()) + list(run_params.items()) + list(construct_dict['hyper_params'].items()))
        ##adding number of model parameters
        N_p=sum(p.numel() for p in model.parameters())
        N_t=sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        paramsf['N_params']=N_p
        paramsf['N_trainable']=N_t
        writer.add_hparams(paramsf, metricf, run_name=run_name_n)
    result_dict={'sigma':scatter,
    'test_acc': test_accs,
    'train_acc': train_accs,
    'ys': ys,
    'pred': pred,
    'xs': xs,
    'Mh': Mh,
    'low':lowest,
    'epochexit': epochexit}
    if not osp.exists(log_dir_glob):
            os.makedirs(log_dir_glob)
    with open(f'{log_dir_glob}/result_dict.pkl', 'wb') as handle:
        pickle.dump(result_dict, handle)
    with open(f'{log_dir_glob}/construct_dict.pkl', 'wb') as handle:
        pickle.dump(construct_dict, handle)
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
    return loss_func()


def get_performance(name):
    '''Returns plotting functions to scrutinize performance on the fly'''
    import dev.eval_plot as evals
    performance_plot = getattr(evals, name)
    return performance_plot 

def setup_model(model_name, hyper_params):
    # Retrieve name and params for construction
    # model_name    = construct_dict['model']
    # hyper_params  = construct_dict['hyper_params']

    # Load model from model folder
    import dev.models as models
    model         = getattr(models, model_name) 
    model         = model(**hyper_params)

    return model

def get_lr_schedule(construct_dict):
    schedule  = construct_dict['learn_params']['schedule']

    import dev.lr_schedule as lr_module

    schedule_class = getattr(lr_module, schedule)

    return schedule_class
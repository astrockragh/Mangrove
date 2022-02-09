import torch, pickle, time, os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from importlib import __import__
import random
import string
from tqdm import tqdm

mus, scales = np.array([-1.1917865,  1.7023178,  -0.14979358, -2.5043619]), np.array([0.9338901, 0.17233825, 0.5423821, 0.9948792])

# t_labels = ['Stellar mass', 'v_disk', 'Cold gas mass', 'SFR average over 100 yr']
t_labels = np.array(['m_star', 'v_disk', 'm_cold', 'sfr_100'])


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

from datetime import date
today = date.today()

today = today.strftime("%d%m%y")


# def load_data(case, targets, del_feats, scale, shuffle, split=0.8):
#     datat=pickle.load(open(osp.expanduser(f'~/../../scratch/gpfs/cj1223/GraphStorage/{case}/data.pkl'), 'rb'))
#     a=np.arange(43)
#     feats=np.delete(a, del_feats)
#     if case!="vlarge_all_smass":
#         data=[]
#         for d in datat:
#             if not scale:
#                 data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y[targets]))
#             else:
#                 data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=(d.y[targets]-torch.Tensor(mus[targets]))/torch.Tensor(scales[targets])))
#     else:
#         data=datat
#     test_data=data[int(len(data)*split):]
#     train_data=data[:int(len(data)*split)]
#     return train_data, test_data
    
# for val    
# def load_data(case, targets, del_feats, scale, shuffle, split=0.8):
#     datat=pickle.load(open(osp.expanduser(f'~/../../scratch/gpfs/cj1223/GraphStorage/{case}/data.pkl'), 'rb'))
#     a=np.arange(43)
#     feats=np.delete(a, del_feats)
#     if case!="vlarge_all_smass":
#         data=[]
#         for d in datat:
#             if not scale:
#                 data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y[targets]))
#             else:
#                 data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=(d.y[targets]-torch.Tensor(mus[targets]))/torch.Tensor(scales[targets])))
#     else:
#         data=datat
#     val_data =  data[int(len(data)*split):int(len(data)*(split+0.1))]
#     train_data=data[:int(len(data)*(split))]
#     return train_data, val_data

# for final testing
def load_data(case, targets, del_feats, scale, shuffle, split=0.8):
    datat=pickle.load(open(osp.expanduser(f'~/../../scratch/gpfs/cj1223/GraphStorage/{case}/data.pkl'), 'rb'))
    a=np.arange(43)
    feats=np.delete(a, del_feats)
    if case!="vlarge_all_smass":
        data=[]
        for d in datat:
            if not scale:
                data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y[targets]))
            else:
                data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=(d.y[targets]-torch.Tensor(mus[targets]))/torch.Tensor(scales[targets])))
    else:
        data=datat
    testidx = pickle.load(open(osp.expanduser(f'~/../../scratch/gpfs/cj1223/GraphStorage/tvt_idx/test_idx.pkl'), 'rb'))
    # trainidx = pickle.load(open(osp.expanduser(f'~/../../scratch/gpfs/cj1223/GraphStorage/tvt_idx/train_idx.pkl'), 'rb'))

    test_data=[]
    train_data=[]
    for i, d in enumerate(data):
        if i in testidx:
            test_data.append(d)
        else:
            train_data.append(d)
    print(len(test_data))
    print(len(train_data))

    return train_data, test_data

def make_id(length=6):
    # choose from all lowercase letters
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

### test function
def test(loader, model, targs, l_func, scale):
    '''returns targets and predictions'''
    ys, pred,xs, Mh=[],[],[], []
    model.eval()
    n_targ=len(targs)
    outs = []
    ys = []
    vars= []
    rhos = []
    if scale:
        sca=torch.cuda.FloatTensor(scales[targs])
        ms=torch.cuda.FloatTensor(mus[targs])
    with torch.no_grad(): ##this solves it!!!
        for data in loader: 
            rho = torch.IntTensor(0)
            var = torch.IntTensor(0)
            if l_func in ["L1", "L2", "SmoothL1"]: 
                out = model(data)  
            if l_func in ["Gauss1d", "Gauss2d", "GaussNd"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr", "Gauss4d_corr"]:
                out, var, rho = model(data) 
            if scale:
                ys.append(data.y.view(-1,n_targ)*sca+ms)
                pred.append(out*sca+ms)
            else:
                ys.append(data.y.view(-1,n_targ))
                pred.append(out)
            vars.append(var)
            rhos.append(rho)

            # u, counts = np.unique(data.batch.cpu().numpy(), return_counts=1)
            # xs.append(np.array(torch.tensor_split(data.x.cpu(), torch.cumsum(torch.tensor(counts[:-1]),0)), dtype=object))
            ## compile lists
    ys = torch.vstack(ys)
    pred = torch.vstack(pred)
    vars = torch.vstack(vars)
    rhos = torch.vstack(rhos)
    # xs=np.hstack(xs)
    xn=[]
    # for x in xs:
    #     x0=x.cpu().detach().numpy()
    #     xn.append(x0)
    #     Mh.append(x0[0][3])
    return ys.cpu().numpy(), pred.cpu().numpy(), xn, Mh, vars, rhos

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
    if learn_params['warmup'] and learn_params["schedule"] in ["warmup_exp", "warmup_expcos"]:
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
    for trial in range(n_trials):
        if n_trials>1:
            run_name_n=run_name+f'_{trial+1}_{n_trials}'
        else:
            run_name_n=run_name
        log_dir=osp.join(pointer, run_name_n)
        print(log_dir)
        if log:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(log_dir=log_dir)
        lowest_metric=np.array([np.inf]*n_targ)
        model = setup_model(construct_dict['model'], construct_dict['hyper_params'])
        print(f"N_params {sum(p.numel() for p in model.parameters())}")
        if save:  # Make folder for saved states
            model_path    = osp.join(log_dir, "trained_model") ##!!!!!!! needs to point to the right spot
            if not osp.isdir(model_path):
                os.makedirs(model_path)
                print('Made folder for saving model')

        train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler=lr_scheduler(optimizer, **learn_params, total_steps=n_epochs*len(train_loader))

        _, _, test_loader = accelerator.prepare(model, optimizer, test_loader)
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        # Initialize our train function
        def train(epoch, schedule):
            model.train()
            return_loss=0
            er_loss = torch.cuda.FloatTensor([0])
            si_loss = torch.cuda.FloatTensor([0])
            rh_loss = torch.cuda.FloatTensor([0])
            for data in train_loader: 
                if run_params["loss_func"] in ["L1", "L2", "SmoothL1"]: 
                    out = model(data)  
                    loss = loss_func(out, data.y.view(-1,n_targ))
                    
                if run_params["loss_func"] in ["Gauss1d", "Gauss2d", "GaussNd"]:
                    out, var = model(data)  
                    loss, err_loss, sig_loss = loss_func(out, data.y.view(-1,n_targ), var)
                    er_loss+=err_loss
                    si_loss+=sig_loss

                if run_params["loss_func"] in ["Gauss2d_corr"]:
                    out, var, rho = model(data)  
                    loss, err_loss, sig_loss, rho_loss = loss_func(out, data.y.view(-1,n_targ), var, rho)
                    er_loss+=err_loss
                    si_loss+=sig_loss
                    rh_loss+=rho_loss

                if run_params["loss_func"] in ["Gauss4d_corr"]:
                    out, var, rho = model(data)  
                    loss, err_loss, sig_loss = loss_func(out, data.y.view(-1,n_targ), var, rho)
                    er_loss+=err_loss
                    si_loss+=sig_loss

                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
                return_loss+=loss
                accelerator.backward(loss)
                optimizer.step() 
                optimizer.zero_grad()
                if schedule=="onecycle":
                    scheduler.step(epoch)
            # if epoch==0:              #Doesn't work right now but could be fun to add back in
            #     writer.add_graph(model,[data]) 
            return return_loss, er_loss, si_loss, rh_loss, l1_lambda * l1_norm, l2_lambda * l2_norm

        tr_acc, te_acc=[],[]
        early_stop=0
        start=time.time()
        k=0
        for epoch in tqdm(range(n_epochs)):
        
            trainloss, err_loss, sig_loss, rho_loss, l1_loss, l2_loss = train(epoch, learn_params["schedule"])
            #learning rate scheduler step
            if learn_params["schedule"]!="onecycle":
                scheduler.step(epoch)

            if (epoch+1)%val_epoch==0:
                if run_params['metrics']!='test_multi_varrho':
                    train_metric, _, _ = metric(train_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
                    test_metric, ys, pred = metric(test_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
                else:
                    train_metric, _, _, _, _ = metric(train_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
                    test_metric, ys, pred, vars, rhos = metric(test_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
                if k==0:
                    lowest_metric=test_metric
                    low_ys = ys
                    low_pred = pred
                    k+=1
                if np.any(test_metric<lowest_metric):
                    mask=test_metric<lowest_metric
                    index=np.arange(n_targ)[mask]
                    lowest_metric[mask]=test_metric[mask]
                    # lowest_metric=test_metric

                    low_ys[:,index]=ys[:,index]
                    low_pred[:,index]=pred[:,index]

                    early_stop=0
                    if save:
                        torch.save(model.state_dict(), osp.join(model_path,'model_best.pt'))
                else:
                    early_stop+=val_epoch
                tr_acc.append(train_metric)
                te_acc.append(test_metric)
                lr0=optimizer.state_dict()['param_groups'][0]['lr']
                last10test=np.median(te_acc[-(10//val_epoch):], axis=0)
                if log:
                    writer.add_scalar('train_loss', trainloss,global_step=epoch+1)
                    writer.add_scalar('learning_rate', lr0, global_step=epoch+1)

                    if n_targ==1:
                        writer.add_scalar('last10test', last10test, global_step=epoch+1)
                        writer.add_scalar('train_scatter', train_metric,global_step=epoch+1)
                        writer.add_scalar('test_scatter', test_metric, global_step=epoch+1)
                        writer.add_scalar('best_scatter', lowest_metric, global_step=epoch+1)

                    else:
                        labels = t_labels[data_params["targets"]]
                        for i in range(n_targ):
                            writer.add_scalar(f'last10test_{labels[i]}', last10test[i], global_step=epoch+1)
                            writer.add_scalar(f'train_scatter_{labels[i]}', train_metric[i], global_step=epoch+1)
                            writer.add_scalar(f'test_scatter_{labels[i]}', test_metric[i], global_step=epoch+1)
                            writer.add_scalar(f'best_scatter_{labels[i]}', lowest_metric[i], global_step=epoch+1)

                
                if run_params["loss_func"] in ["Gauss1d", "Gauss2d", "Gauss2d_corr", "Gauss4d_corr", "Gauss_Nd"]:
                    print(f'Epoch: {int(epoch+1)} done with learning rate {lr0:.2E}, Train loss: {trainloss.cpu().detach().numpy():.2E}, [Err/Sig/Rho]: {err_loss.cpu().detach().numpy()[0]:.2E}, {sig_loss.cpu().detach().numpy()[0]:.2E}, {rho_loss.cpu().detach().numpy()[0]:.2E}')
                    print(f'L1 regularization loss: {l1_loss.cpu().detach().numpy():.2E}, L2 regularization loss: {l2_loss.cpu().detach().numpy():.2E}')
                    print(f'Train scatter: {np.round(train_metric,4)}')
                else:
                    print(f'Epoch: {int(epoch+1)} done with learning rate {lr0:.2E}, Train loss: {trainloss.cpu().detach().numpy():.2E}, Train scatter: {np.round(train_metric,4)}')
                    print(f'L1 regularization loss: {l1_loss.cpu().detach().numpy():.2E}, L2 regularization loss: {l2_loss.cpu().detach().numpy():.2E}')
                print(f'Test scatter: {np.round(test_metric,4)}, Lowest was {np.round(lowest_metric,4)}')
 
                print(f'Median for last 10 epochs: {np.round(last10test,4)}, Epochs since improvement {early_stop}')
                if (epoch+1)%(int(val_epoch*5))==0 and log:
                    if n_targ==1:
    
                        ys, pred, xs, Mh, vars, rhos = test(test_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"])
                        fig=performance_plot(ys,pred, xs, Mh, data_params["targets"])
                        writer.add_figure(tag=run_name_n, figure=fig, global_step=epoch+1)
                    else:
                        labels = t_labels[data_params["targets"]]
                        ys, pred, xs, Mh, vars, rhos = test(test_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"])
                        figs = performance_plot(ys,pred, xs, Mh, data_params["targets"])
                        for fig, label in zip(figs, labels):
                            writer.add_figure(tag=f'{run_name_n}_{label}', figure=fig, global_step=epoch+1)

            if early_stopping:
                if early_stop>patience:
                    print(f'Exited after {epoch+1} epochs due to early stopping')
                    epochexit.append(epoch)
                    break
        if early_stopping:
            if early_stop<patience:
                epochexit.append(epoch)
            else:
                epochexit.append(n_epochs)

        stop=time.time()
        spent=stop-start
        test_accs.append(te_acc)
        train_accs.append(tr_acc)
        
        if early_stopping and n_epochs>epochexit[-1]:
            pr_epoch=epochexit[-1]
        else:
            pr_epoch=n_epochs
        print(f"{spent:.2f} seconds spent training, {spent/n_epochs:.3f} seconds per epoch. Processed {len(train_loader.dataset)*pr_epoch/spent:.0f} trees per second")
        
        ys, pred, xs, Mh, vars, rhos = test(test_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"])
        if save:
            if n_targ==1:
                label = t_labels[data_params["targets"]]
                figs=performance_plot(ys,pred, xs, Mh, data_params["targets"])
                for fig in figs:
                    fig.savefig(f'{log_dir}/performance_ne{n_epochs}_{label}.png')
            else:
                labels = t_labels[data_params["targets"]]
                ys, pred, xs, Mh, vars, rhos = test(test_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"])
                figs = performance_plot(ys,pred, xs, Mh, data_params["targets"])
                for fig, label in zip(figs, labels):
                    fig.savefig(f'{log_dir}/performance_ne{n_epochs}_{label}.png')
        sig=np.std(ys-pred, axis=0)
        print(sig)
        scatter.append(sig)
        preds.append(pred)
        lowest.append(lowest_metric)
        last20=np.median(te_acc[-(20//val_epoch):], axis=0)
        last10=np.median(te_acc[-(10//val_epoch):], axis=0)
        
        #################################
        ###    Make saveable params  ###
        #################################
  
        paramsf=dict(list(data_params.items()) + list(run_params.items()) + list(construct_dict['hyper_params'].items()) + list(construct_dict['learn_params'].items()))
        paramsf["targets"]=int("".join([str(i+1) for i in data_params["targets"]]))
        paramsf["del_feats"]=str(data_params["del_feats"])
        ##adding number of model parameters
        N_p=sum(p.numel() for p in model.parameters())
        N_t=sum(p.numel() for p in model.parameters() if p.requires_grad)
        paramsf['N_params']=N_p
        paramsf['N_trainable']=N_t

        #################################
        ###    Make saveable metrics  ###
        #################################
        
        
        labels = t_labels[data_params["targets"]]
        metricf={'epoch_exit':epoch}
        for i in range(n_targ):

            metricf[f'scatter_{labels[i]}']=sig[i]
            metricf[f'lowest_{labels[i]}']=lowest_metric[i]
            metricf[f'last20_{labels[i]}']=last20[i]
            metricf[f'last10_{labels[i]}']=last10[i]
        print(metricf)
        writer.add_hparams(paramsf, metricf, run_name=run_name_n)
        

        print(f'Finished {trial+1}/{n_trials}')
        result_dict={'sigma':scatter,
        'test_acc': te_acc,
        'train_acc': tr_acc,
        'ys': ys,
        'pred': pred,
        'low_ys': low_ys,
        'low_pred': low_pred,
        'vars': vars,
        'rhos': rhos,
        'low':lowest,
        'epochexit': epochexit}
        # if not osp.exists(log_dir):
        #         os.makedirs(log_dir)
        with open(f'{log_dir}/result_dict.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle)
        with open(f'{log_dir}/construct_dict.pkl', 'wb') as handle:
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
#         Custom loss       #
################################

def get_loss_func(name):
    # Return loss func from the loss functions file given a function name
    import dev.loss_funcs as loss_func_module
    loss_func = getattr(loss_func_module, name)
    try:
        l=loss_func()
    except:
        l=loss_func
    return l


def get_performance(name):
    '''Returns plotting functions to scrutinize performance on the fly'''
    import dev.eval_plot as evals
    performance_plot = getattr(evals, name)
    return performance_plot 

def setup_model(model_name, hyper_params):
    # Retrieve name and params for construction

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
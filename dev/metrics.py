from torch import sqrt, sum, square, no_grad, vstack, std, IntTensor
from torch.cuda import FloatTensor
import numpy as np


mus, scales = np.array([-1.1917865,  1.7023178,  -0.14979358, -2.5043619]), np.array([0.9338901, 0.17233825, 0.5423821, 0.9948792])

def scatter(loader, model, n_targ):
    model.eval()

    correct = 0
    with no_grad():
        for dat in loader: 
            out = model(dat) 
            correct += sum(square(out - dat.y.view(-1,n_targ)))
    return sqrt(correct/len(loader.dataset)).cpu().detach().numpy()

def test_multi(loader, model, targs, l_func, scale):
    model.eval()
    n_targ=len(targs)
    outs = []
    ys = []
    if scale:
        sca=FloatTensor(scales[targs])
        ms=FloatTensor(mus[targs])
    with no_grad(): ##this solves it!!!
        for data in loader: 
            if l_func in ["L1", "L2", "SmoothL1"]: 
                out = model(data)  
            if l_func in ["Gauss1d", "Gauss2d", "GaussNd"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr, Gauss4d_coor"]:
                out, var, rho = model(data) 
            if scale:
                ys.append(data.y.view(-1,n_targ)*sca+ms)
                outs.append(out*sca+ms)
            else:
                ys.append(data.y.view(-1,n_targ))
                outs.append(out)
    outss=vstack(outs)
    yss=vstack(ys)
    return std(outss - yss, axis=0).cpu().detach().numpy(), yss.cpu().detach().numpy(), outss.cpu().detach().numpy()

def test_multi_varrho(loader, model, targs, l_func, scale): 
    model.eval()
    n_targ=len(targs)
    outs = []
    ys = []
    vars = []
    rhos = []
    if scale:
        sca=FloatTensor(scales[targs])
        ms=FloatTensor(mus[targs])
    with no_grad(): 
        for data in loader: 
            rho = IntTensor(0)
            var = IntTensor(0)
            if l_func in ["L1", "L2", "SmoothL1"]: 
                out = model(data)  
            if l_func in ["Gauss1d", "Gauss2d", "GaussNd"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr", "Gauss4d_corr"]:
                out, var, rho = model(data) 
            if scale:
                ys.append(data.y.view(-1,n_targ)*sca+ms)
                outs.append(out*sca+ms)
            else:
                ys.append(data.y.view(-1,n_targ))
                outs.append(out)
            vars.append(var)
            rhos.append(rho)

    outss=vstack(outs)
    yss=vstack(ys)
    vars = vstack(vars)
    rhos = vstack(rhos)
    return std(outss - yss, axis=0).cpu().detach().numpy(), yss.cpu().detach().numpy(), outss.cpu().detach().numpy(), vars, rhos
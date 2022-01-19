from torch import sqrt, sum, square, no_grad, vstack, std
from torch.cuda import FloatTensor
import numpy as np

mus, scales = np.array([-1.18660497,  1.70294617,  0.11209364, -1.00491434]), np.array([0.93492807, 0.17271924, 0.46982766, 1.40001287])

def scatter(loader, model, n_targ):
    model.eval()

    correct = 0
    with no_grad():
        for dat in loader: 
            out = model(dat) 
            correct += sum(square(out - dat.y.view(-1,n_targ)))
    return sqrt(correct/len(loader.dataset)).cpu().detach().numpy()

def test_multi(loader, model, targs, l_func, scale): ##### transform back missing
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
            if l_func in ["Gauss1d", "Gauss2d", "GaussN"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr"]:
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
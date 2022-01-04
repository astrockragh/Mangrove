from torch import sqrt, sum, square, no_grad, vstack, std

def scatter(loader, model, n_targ):
    model.eval()

    correct = 0
    with no_grad():
        for dat in loader: 
            out = model(dat) 
            correct += sum(square(out - dat.y.view(-1,n_targ)))
    return sqrt(correct/len(loader.dataset)).cpu().detach().numpy()

def test_multi(loader, model, n_targ, l_func): ##### transform back missing
    model.eval()
    outs = []
    ys = []
    with no_grad(): ##this solves it!!!
        for data in loader: 
            if l_func in ["L1", "L2", "SmoothL1"]: 
                out = model(data)  
            if l_func in ["Gauss1d", "Gauss2d"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr"]:
                out, var, rho = model(data)  
            ys.append(data.y.view(-1,n_targ))
            outs.append(out)
    outss=vstack(outs)
    yss=vstack(ys)
    return std(outss - yss, axis=0).cpu().detach().numpy()
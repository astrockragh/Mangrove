from torch import sqrt, sum, square, no_grad, vstack, std

def scatter(loader, model, n_targ):
    model.eval()

    correct = 0
    with no_grad():
        for dat in loader: 
            out = model(dat.x, dat.edge_index, dat.batch) 
            correct += sum(square(out - dat.y.view(-1,n_targ)))
    return sqrt(correct/len(loader.dataset)).cpu().detach().numpy()

def test_multi(loader, model, n_targ): ##### transform back missing
    model.eval()
    outs = []
    ys = []
    with no_grad(): ##this solves it!!!
        for dat in loader: 
            out = model(dat.x, dat.edge_index, dat.batch) 
            ys.append(dat.y.view(-1,n_targ))
            outs.append(out)
    outss=vstack(outs)
    yss=vstack(ys)
    return std(outss - yss, axis=0).cpu().detach().numpy()
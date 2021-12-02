from torch import sqrt, sum, square, no_grad

def scatter(loader, model):
    model.eval()

    correct = 0
    with no_grad():
        for dat in loader: 
            out = model(dat.x, dat.edge_index, dat.batch) 
            correct += sum(square(out - dat.y.view(-1,1)))
    return sqrt(correct/len(loader.dataset)).cpu().detach().numpy()
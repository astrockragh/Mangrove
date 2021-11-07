from torch.nn import MSELoss, L1Loss

def L2():
    return MSELoss()

def L1():
    return L1Loss()
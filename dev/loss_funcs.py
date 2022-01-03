from torch.nn import MSELoss, L1Loss, SmoothL1Loss
from torch import log, sum
###############################
###    Simple loss funcs   ####
###############################
def L2():
    return MSELoss()

def L1():
    return L1Loss()

def SmoothL1():
    return SmoothL1Loss(beta=0.5)

###############################
###  Homemade loss funcs   ####
###############################

def Gauss2d_corr(pred, ys, sig1, sig2, rho):
    z1=(pred[:,0]-ys[:,0])/sig1
    z2=(pred[:,1]-ys[:,1])/sig2
    sigloss=sum(log(sig1)+log(sig2))
    rholoss=sum(log(1-rho**2)/2)
    factor=1/(2*(1-rho**2))
    err_loss = sum(factor*(z1**2+z2**2-2*rho*z1*z2))
    
    return err_loss+sigloss+rholoss, err_loss, sigloss, rholoss

def Gauss2d(pred, ys, sig1, sig2):
    z1=(pred[:,0]-ys[:,0])/sig1
    z2=(pred[:,1]-ys[:,1])/sig2
    sigloss=sum(log(sig1)+log(sig2))
    err_loss = sum((z1**2+z2**2)/2)
    
    return err_loss+sigloss, err_loss, sigloss

def Gauss1d(pred, ys, sig):
    z=(pred-ys)/sig
    sigloss=sum(log(sig))
    err_loss = sum(z**2)/2
    
    return err_loss+sigloss, err_loss, sigloss
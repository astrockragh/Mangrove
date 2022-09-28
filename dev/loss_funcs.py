from torch.nn import MSELoss, L1Loss, SmoothL1Loss
from torch import log, sum, square, vstack, zeros, bmm, det, inverse
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
##########################################
###  All are negative log likelihoods   ####
##########################################

def Gauss2d_corr(pred, ys, var, rho):
    sig1=var[:,0]
    sig2=var[:,1]
    z1=(pred[:,0]-ys[:,0])/sig1
    z2=(pred[:,1]-ys[:,1])/sig2
    sigloss=sum(log(sig1)+log(sig2))
    rholoss=sum(log(1-rho**2)/2)
    factor=1/(2*(1-rho**2))
    err_loss = sum(factor*(z1**2+z2**2-2*rho*z1*z2))
    
    return err_loss+sigloss+rholoss, err_loss, sigloss, rholoss

def Gauss4d_corr(pred, ys, sig, rho):
    '''This is written assuming normal pytorch, on cuda, inputs should be 4xbatchsize, 4xbatchsize, 4xbatchsize, 6xbatchsize'''
    delta=pred-ys
    bsize = delta.shape[0]
    N = delta.shape[1]
    #this is messy but it works 
    
    #compute the covariance matrix
    vals = vstack([sig[:,0]**2, rho[:,0]*sig[:,0]*sig[:,1], rho[:,1]*sig[:,0]*sig[:,2], rho[:,2]*sig[:,0]*sig[:,3],\
                 sig[:,1]**2,rho[:,3]*sig[:,1]*sig[:,2], rho[:,4]*sig[:,1]*sig[:,3], \
                sig[:,2]**2,rho[:,5]*sig[:,2]*sig[:,3],\
                 sig[:,3]**2])

    A = zeros(N, N,bsize, device='cuda:0') # assuming you're on the gpu
    A[0] = vals[:4]
    A[1] = vstack([vals[1], vals[4:7]])
    A[2] = vstack([vals[2], vals[5], vals[7:9]])
    A[3] = vstack([vals[3], vals[6], vals[8], vals[9]])
    
    A2=A.permute(2,0,1)
    
    dethat = det(A2)
    detloss = sum(log(dethat))/2
    
    sig_inv = inverse(A2)
    
    err=delta*bmm(sig_inv, delta.unsqueeze(2))[:,:,0]
    
    err_loss = sum(err)/2
    
    return err_loss+detloss, err_loss, detloss

def Gauss2d(pred, ys, var):
    '''2d Gaussian, can be subbed for GaussND'''
    sig1=var[:,0]
    sig2=var[:,1]
    z1=(pred[:,0]-ys[:,0])/sig1
    z2=(pred[:,1]-ys[:,1])/sig2
    sigloss=sum(log(sig1)+log(sig2))
    err_loss = sum((z1**2+z2**2)/2)
    
    return err_loss+sigloss, err_loss, sigloss

def GaussNd(pred, ys, var):
    '''General uncorrelated gaussian, if you're having trouble you can try adding a small number to your sigmas'''
    z=(pred-ys)/var
    sigloss=sum(log(var))
    err_loss = sum((square(z)))/2
    
    return err_loss+sigloss, err_loss, sigloss    


def Gauss1d(pred, ys, sig):
    '''1d Gaussian, can be subbed for GaussND'''
    z=(pred-ys)/sig
    sigloss=sum(log(sig))
    err_loss = sum(z**2)/2
    
    return err_loss+sigloss, err_loss, sigloss
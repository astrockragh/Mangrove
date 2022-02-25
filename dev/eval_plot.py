import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

cols_t=np.array(['M_star', 'v_disk', 'm_cold gas', 'sfr_ave100Myr'])

def SAM_base(ys, pred, xs, Mh):
    ''' Basic lotting scheme for an unspecified range of targets, xs/Mh keywords aren't used, left for dependency 
    ys/pred should be the same dimensionality'''
    fig, ax =plt.subplots(figsize=(6,6))
    ax.plot(ys,pred, 'ro', alpha=0.3)
    ax.plot([min(ys),max(ys)],[min(ys),max(ys)], 'k--', label='Perfect correspondance')
    ax.set(xlabel='SAM Truth',ylabel='GNN Prediction', title='True/predicted correlation')
    yhat=r'$\hat{y}$'
    ax.text(0.6,0.15, f'Bias (mean(y-{yhat})) : {np.mean(ys-pred):.3f}', transform=ax.transAxes)
    ax.text(0.6,0.1, r'$\sigma$ :  '+f'{np.std(ys-pred):.3f}', transform=ax.transAxes)
    ax.legend()
    return fig
    
def multi_base(ys, pred, xs, Mh, targets):
    ''' Plotting scheme for an unspecified range of targets, xs/Mh keywords aren't used, left for dependency 
    ys/pred should be the same dimensionality, targets should be numerical indexed, not boolean'''
    n_t = len(targets)
    figs=[]
    for n in range(n_t):
        fig, ax =plt.subplots(1,2, figsize=(12,6))
        ax=ax.flatten()
        ax[0].plot(ys[:,n],pred[:,n], 'ro', alpha=0.3)
        ax[0].plot([min(ys[:,n]),max(ys[:,n])],[min(ys[:,n]),max(ys[:,n])], 'k--', label='Perfect correspondance')
        ax[0].set(xlabel='SAM Truth',ylabel='GNN Prediction', title=cols_t[targets[n]])
        yhat=r'$\hat{y}$'
        ax[0].text(0.6,0.15, f'Bias (mean(y-{yhat})) : {np.mean(ys[:,n]-pred[:,n]):.3f}', transform=ax[0].transAxes)
        ax[0].text(0.6,0.1, r'$\sigma$ :  '+f'{np.std(ys[:,n]-pred[:,n]):.3f}', transform=ax[0].transAxes)
        ax[0].legend()
        vals, x, y, _ =ax[1].hist2d(ys[:,n],pred[:,n],bins=50, norm=mpl.colors.LogNorm(), cmap=mpl.cm.magma)
        X, Y = np.meshgrid((x[1:]+x[:-1])/2, (y[1:]+y[:-1])/2)
        ax[1].contour(X,Y, np.log(vals.T+1), levels=10, colors='black')
        ax[1].plot([min(ys[:,n]),max(ys[:,n])],[min(ys[:,n]),max(ys[:,n])], 'k--', label='Perfect correspondance')
        ax[1].set(xlabel='SAM Truth',ylabel='GNN Prediction', title=cols_t[targets[n]])
        ax[1].legend()
        fig.tight_layout()
        figs.append(fig)
    return figs
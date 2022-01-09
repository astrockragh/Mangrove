import matplotlib.pyplot as plt
import numpy as np


def SAM_base(ys, pred, xs, Mh):
    fig, ax =plt.subplots(figsize=(6,6))
    ax.plot(ys,pred, 'ro', alpha=0.3)
    ax.plot([min(ys),max(ys)],[min(ys),max(ys)], 'k--', label='Perfect correspondance')
    ax.set(xlabel='SAM Truth',ylabel='GNN Prediction', title='True/predicted correlation')
    yhat=r'$\hat{y}$'
    ax.text(0.6,0.15, f'Bias (mean(y-{yhat})) : {np.mean(ys-pred):.3f}', transform=ax.transAxes)
    ax.text(0.6,0.1, r'$\sigma$ :  '+f'{np.std(ys-pred):.3f}', transform=ax.transAxes)
    ax.legend()
    return fig
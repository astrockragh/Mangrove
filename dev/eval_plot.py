import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

cols_t=np.array(['M_star', 'v_disk', 'm_cold gas', 'sfr_ave100Myr']) # old columns

cols_t = np.array(['halo_index (long) (0)',
 'birthhaloid (long long)(1)',
 'roothaloid (long long)(2)',
 'redshift(3)',
 'sat_type 0= central(4)',
 'mhalo total halo mass [1.0E09 Msun](5)',
 'm_strip stripped mass [1.0E09 Msun](6)',
 'rhalo halo virial radius [Mpc)](7)',
 'mstar stellar mass [1.0E09 Msun](8)',
 'mbulge stellar mass of bulge [1.0E09 Msun] (9)',
 ' mstar_merge stars entering via mergers] [1.0E09 Msun](10)',
 ' v_disk rotation velocity of disk [km/s] (11)',
 ' sigma_bulge velocity dispersion of bulge [km/s](12)',
 ' r_disk exponential scale radius of stars+gas disk [kpc] (13)',
 ' r_bulge 3D effective radius of bulge [kpc](14)',
 ' mcold cold gas mass in disk [1.0E09 Msun](15)',
 ' mHI cold gas mass [1.0E09 Msun](16)',
 ' mH2 cold gas mass [1.0E09 Msun](17)',
 ' mHII cold gas mass [1.0E09 Msun](18)',
 ' Metal_star metal mass in stars [Zsun*Msun](19)',
 ' Metal_cold metal mass in cold gas [Zsun*Msun] (20)',
 ' sfr instantaneous SFR [Msun/yr](21)',
 ' sfrave20myr SFR averaged over 20 Myr [Msun/yr](22)',
 ' sfrave100myr SFR averaged over 100 Myr [Msun/yr](23)',
 ' sfrave1gyr SFR averaged over 1 Gyr [Msun/yr](24)',
 ' mass_outflow_rate [Msun/yr](25)',
 ' metal_outflow_rate [Msun/yr](26)',
 ' mBH black hole mass [1.0E09 Msun](27)',
 ' maccdot accretion rate onto BH [Msun/yr](28)',
 ' maccdot_radio accretion rate in radio mode [Msun/yr](29)',
 ' tmerge time since last merger [Gyr] (30)',
 ' tmajmerge time since last major merger [Gyr](31)',
 ' mu_merge mass ratio of last merger [](32)',
 ' t_sat time since galaxy became a satellite in this halo [Gyr](33)',
 ' r_fric distance from halo center [Mpc](34)',
 ' x_position x coordinate [cMpc](35)',
 ' y_position y coordinate [cMpc](36)',
 ' z_position z coordinate [cMpc](37)',
 ' vx x component of velocity [km/s](38)',
 ' vy y component of velocity [km/s](39)',
 ' vz z component of velocity [km/s](40)'])

cols_t = [t.replace(' ','') for t in cols_t]
cols_t = [t.replace('/','') for t in cols_t]
cols_t = np.array(cols_t)

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
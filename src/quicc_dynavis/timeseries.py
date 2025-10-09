import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from .io import get_parameters, get_resolution
from .io import F_conc_timeseries, F_read_dipolarity, F_read_energyQCC, F_read_Nusselt

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'cm'   # use Computer Modern for math
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def plot_timeseries(folderFile, save_dir, show=True, xlim=None, ylim=None):
    """
    Plot full timeseries of kinetic/magnetic energy, dipolarity, dipole angle, and dissipations.

    Args:
        folderFile: path containing run folders (run0, run1, ...)
        show: whether to display the figure immediately
        xlim
        ylim 

    Returns:
        fig, axes
    """
    # Get all run folders
    RunFolders = sorted(glob.iglob(os.path.join(folderFile, 'run*')))

    # Read timeseries
    tkin, kinEtot, kinEtor, kinEpol    = F_conc_timeseries(RunFolders, 'kinE')
    tmag, magEtot, magEtor, magEpol    = F_conc_timeseries(RunFolders, 'magE')
    tdip, fdip, g10, g11, h11          = F_conc_timeseries(RunFolders, 'Dip')
    tkinDis, kinDtot, kinDtor, kinDpol = F_conc_timeseries(RunFolders, 'kinE')
    tmagDis, magDtot, magDtor, magDpol = F_conc_timeseries(RunFolders, 'magE')

    # Calculate dipole angle
    dipangle = np.arccos(g10 / np.sqrt(g10**2 + g11**2 + h11**2)) * 180 / np.pi

    # Read simulation parameters
    Ek, Pm, Pr, q, Ra, Ro = get_parameters(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
    Nres, Mres, Lres = get_resolution(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')

    # Compute time-averaged Rossby number
    t = tkin
    Ro = Ek * np.sqrt(2 * kinEtot)
    startindex = int(0.3 * len(Ro))
    timeavg_Ro = np.round(np.mean(Ro[startindex:]), 4)

    # ------ Create figure------#

    plt.close('all')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True, dpi =180)
    ax1.set_title(f'$Ek: {Ek:.1e}, Ra: {Ra:.3e}, q: {q:.1f}, (N,L,M): ({Nres:.0f},{Mres:.0f},{Lres:.0f})$')

    # Energy plot
    ax1.plot(tkin, kinEtot, label=r'$\mathcal{E}_{kin}$')
    ax1.plot(tmag, magEtot, label=r'$\mathcal{E}_{mag}$')
    ax1.set_yscale('log')
    ax1.set_ylabel('Energy density')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax1.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax1.set_xlim(xlim)        

    if ylim is not None:
       ax1.set_ylim(ylim)    
    ax1.legend()

    # Dipolarity
    if len(tdip) > 0:
        ax2.plot(tdip, fdip, color='red', label='g10', alpha=0.6)
    ax2.set_ylabel('Dipolarity')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax2.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax2.set_xlim(xlim)  
    ax2.set_ylim(0, 1)

    # Dipole latitude
    if len(g10) > 0:
        ax3.plot(tdip, dipangle, 'k', alpha=0.7)
    ax3.set_ylabel('Dipole angle (deg)')
    ax3.set_ylim(-5, 185)
    ax3.axhline(90, color='gray', linestyle='--', alpha=0.4, label=r'$90^{\circ}$')
    #ax3.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax3.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax3.set_xlim(xlim)  
    ax3.legend()

    # Dissipations
    if len(tkinDis) > 0:
        ax4.plot(tkinDis, kinDtot, alpha=0.7, label='kinetic dissipation')
        ax4.plot(tmagDis, magDtot, alpha=0.7, label='magnetic dissipation')
    ax4.set_ylabel('Dissipation')
    ax4.set_xlabel('Time')
    ax4.set_yscale('log')
    #ax4.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    if xlim is None:
      xlim = (np.max(t) - np.min(t)) * 0.05
      ax4.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
      ax4.set_xlim(xlim)  
    ax4.legend()
    
    #save figure
    Ek,Pm,Pr,q,Ra,Ro=get_parameters(folderFile+'/run0/parameters.cfg','no')
    #Ek in format 1e-4
    Ek = f'{Ek:.1e}'
    save_path = os.path.join(save_dir, f'Ek_{Ek}_Ra{Ra}_q{q}_timeseries.png')
    plt.savefig(save_path, dpi=270)

    if show:
        plt.show()
    
    return fig, (ax1, ax2, ax3, ax4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot timeseries from QuICC runs.")
    parser.add_argument("folder", help="Path to the start folder containing run subfolders")
    parser.add_argument("--save", help="Path to save the figure", default=None)
    args = parser.parse_args()

    plot_timeseries(start_folder=args.folder, show=True, save_path=args.save)
    
    # TO RUN it in terminal:
    #python -m quicc_dynavis.timeseries /path/to/my/simulation --save timeseries.png

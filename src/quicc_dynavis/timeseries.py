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
 

    #physcial kientic energy
    if  Pm !=0 and Pr !=0: 
        kinEtot = Ek/Pm * kinEtot 
       
    # ------ Create figure------#

    plt.close('all')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True, dpi =180)
    ax1.set_title(f'$Ek: {Ek:.1e}, Ra: {Ra:.2e}, q: {q:.1f}, (N,L,M): ({Nres:.0f},{Mres:.0f},{Lres:.0f})$')

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
    Ra = f'{Ra:.1e}'
    q = f'{q:.1e}'
    
    save_path = os.path.join(save_dir, f'Ek_{Ek}_Ra{Ra}_q{q}_timeseries.png')
    plt.savefig(save_path, dpi=270)

    if show:
        plt.show()
    
    return fig, (ax1, ax2, ax3, ax4)
#-----------------------------------------

def compare_energy(folderFile1, folderFile2, save_dir=None, show=True, xlim=None, ylim=None):
    """
    Compare kinetic and magnetic energies between two simulation runs.

    Args:
        folderFile1: Path to the first run folder or a directory containing multiple runs (e.g., 'run0', 'run1', ...)
        folderFile2: Path to the second run folder or directory
        save_dir: Directory to save the comparison figure (optional)
        show: Whether to display the figure
        xlim: (xmin, xmax) for time axis
        ylim: (ymin, ymax) for energy axis

    Returns:
        fig, axes
    """
    # Get all run folders inside each directory
    RunFolders1 = sorted(glob.iglob(os.path.join(folderFile1, 'run*')))
    RunFolders2 = sorted(glob.iglob(os.path.join(folderFile2, 'run*')))

    # Read concatenated timeseries
    tkin1, kinEtot1, kinEtor1, kinEpol1 = F_conc_timeseries(RunFolders1, 'kinE')
    tmag1, magEtot1, magEtor1, magEpol1 = F_conc_timeseries(RunFolders1, 'magE')

    tkin2, kinEtot2, kinEtor2, kinEpol2 = F_conc_timeseries(RunFolders2, 'kinE')
    tmag2, magEtot2, magEtor2, magEpol2 = F_conc_timeseries(RunFolders2, 'magE')

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    # Plot kinetic energy comparison
    axes[0].plot(tkin1, kinEtot1, label='Run 1: Kinetic_total', color='tab:blue', lw=1.8)
    axes[0].plot(tkin2, kinEtot2, label='Run 2: Kinetic_total', color='tab:orange', lw=1.8, ls='--')
    axes[0].set_ylabel('Kinetic Energy')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot magnetic energy comparison
    axes[1].plot(tmag1, magEtot1, label='Run 1: Magnetic_total', color='tab:green', lw=1.8)
    axes[1].plot(tmag2, magEtot2, label='Run 2: Magnetic_total', color='tab:red', lw=1.8, ls='--')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Magnetic Energy')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Set limits
    if xlim: axes[1].set_xlim(xlim)
    if ylim: 
        for ax in axes:
            ax.set_ylim(ylim)

    # Save figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'compare_energy.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Energy comparison figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def compare_energy_multi(run_paths, labels=None, save_dir=None, show=True, xlim=None, ylim=None):
    """
    Compare kinetic and magnetic energies from multiple runs.

    Args:
        run_paths: list of paths (each can be a single run or a folder containing run0, run1, ...)
        labels: list of legend labels (optional). If None, folder names will be used.
        save_dir: directory to save the comparison figure
        show: display figure
        xlim: (xmin, xmax)
        ylim: (ymin, ymax) applied to ALL subplots

    Returns:
        fig, axes
    """

    # Validate input
    if not isinstance(run_paths, (list, tuple)):
        raise ValueError("run_paths must be a list of paths")
    if labels and len(labels) != len(run_paths):
        raise ValueError("labels length must match run_paths length")

    # Auto-generate labels if needed
    if labels is None:
        labels = [os.path.basename(p.rstrip('/')) for p in run_paths]

    # Prepare plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    # Colormap for multiple runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_paths)))

    # ---------- Loop over ALL runs ----------
    for idx, path in enumerate(run_paths):

        # Find run folders
        if os.path.isdir(path):
            runs = sorted(glob.iglob(os.path.join(path, "run*")))
            if len(runs) == 0:
                # Maybe user passed a single file instead of folder
                runs = [path]
        else:
            runs = [path]

        # Read timeseries
        tkin, kinEtot, _, _ = F_conc_timeseries(runs, "kinE")
        tmag, magEtot, _, _ = F_conc_timeseries(runs, "magE")
        
        # Read simulation parameters
        Ek, Pm, Pr, q, Ra, Ro = get_parameters(os.path.join(runs[0], 'parameters.cfg'), 'no')
        if  Pm !=0 and Pr !=0: 
            kinEtot = Ek/Pm * kinEtot 

        # ---------- Plot kinetic ----------
        axes[0].plot(tkin, kinEtot,
                     label=f"{labels[idx]} - Kinetic",
                     #color=colors[idx], 
                     lw=1.8)

        # ---------- Plot magnetic ----------
        axes[1].plot(tmag, magEtot,
                     label=f"{labels[idx]} - Magnetic",
                     #color=colors[idx], 
                     lw=1.8)

    # ---------- Styling ----------
    axes[0].set_ylabel("Kinetic Energy")
    axes[1].set_ylabel("Magnetic Energy")
    axes[1].set_xlabel("Time")

    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)

    for ax in axes:
        ax.grid(alpha=0.35)

        if ylim:
            ax.set_ylim(ylim)

    if xlim:
        axes[1].set_xlim(xlim)

    # ---------- Save figure ----------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, "compare_energy_multi.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {fname}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot timeseries from QuICC runs.")
    parser.add_argument("folder", help="Path to the start folder containing run subfolders")
    parser.add_argument("--save", help="Path to save the figure", default=None)
    args = parser.parse_args()

    plot_timeseries(start_folder=args.folder, show=True, save_path=args.save)
    
    # TO RUN it in terminal:
    #python -m quicc_dynavis.timeseries /path/to/my/simulation --save timeseries.png

import os
import re
import csv
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


def input_params_from_path(folderFile):
    Ek,Pm,Pr,q,Ra,Ro=get_parameters(folderFile+'/run0/parameters.cfg','no')
    #Ek in format 1e-4
    Ek = f'{Ek:.1e}'
    Ra = f'{Ra:.1e}'
    q = f'{q:.1f}'

    return Ek,q,Ra

def extract_Ek_root(path):
    """
    Extract the directory up to and including the Ek folder (e.g. E1e-5).

    Example:
    /.../CattaneoHuges/E1e-5/q_1.2/Ra2e3/run0
        --> /.../CattaneoHuges/E1e-5
    """
    parts = os.path.normpath(path).split(os.sep)

    for i, p in enumerate(parts):
        if re.match(r"E\d+e[-+]?\d+", p):
            return os.sep.join(parts[: i + 1])

    raise ValueError(f"Could not find Ek folder in path:\n{path}")

import os
import csv
import numpy as np

def write_dynamo_summary_csv(
    csv_path,
    q, Ra, Ek,
    dynamo,
    dipolarity,
    Elsasser,
    reversal,
    Rm,
    rtol=1e-10
):
    """
    Write or update dynamo diagnostics in a CSV file.

    If (q, Ra, Ek) already exists (within tolerance), overwrite that row.
    Otherwise, append a new row.

    Final table is sorted by:
        1) q
        2) Ra
    """

    # -------- Compact, readable header --------
    header = [
        "q",
        "Ra",
        "Ek",
        "dyn",
        "fdip",
        "Lambda",
        "rev",
        "Rm"
    ]

    new_row = [
        f"{q:.2f}",
        f"{Ra:.2e}",
        f"{Ek:.2e}",
        int(dynamo),
        f"{dipolarity:.2f}",
        f"{Elsasser:.2e}",
        int(reversal),
        f"{Rm:.2f}"
    ]

    rows = []

    # ---------- Read existing file ----------
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if rows and rows[0] != header:
            raise ValueError(f"CSV header mismatch in {csv_path}")
    else:
        rows.append(header)

    # ---------- Search & overwrite ----------
    updated = False
    for i in range(1, len(rows)):
        q_i,Ra_i, Ek_i = map(float, rows[i][:3])

        if (
            abs(Ra_i - Ra) / Ra < rtol and
            abs(q_i - q) / max(1.0, q) < rtol and
            abs(Ek_i - Ek) / Ek < rtol
        ):
            rows[i] = new_row
            updated = True
            break

    # ---------- Append if new ----------
    if not updated:
        rows.append(new_row)

    # ---------- Sort by q, then Ra ----------
    data_rows = rows[1:]

    data_rows.sort(
        key=lambda r: (float(r[0]), float(r[1]))  # q first, then Ra
    )

    rows = [header] + data_rows

    # ---------- Write back ----------
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print("========================================")
    print(f"✅ Dynamo summary CSV updated & sorted: {csv_path}")




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
    timeavg_Ro = np.round(np.mean(Ro[startindex:]), 2)
 

    #physcial kientic energy
    if  Pm !=0 and Pr !=0: 
        kinEtot = Ek/Pm * kinEtot 
       
    # ------ Create figure------#

    plt.close('all')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True, dpi =180)
    ax1.set_title(f'$Ek: {Ek:.1e}, Ra: {Ra:.2e}, q: {q:.1f}, (N,L,M): ({Nres:.0f},{Mres:.0f},{Lres:.0f})$')
    
    # compute magnetic Reynolds number
    Rm = np.sqrt(np.mean(kinEtot))

    print('Input parameters:')
    print('Ek:', Ek)
    print('Ra:', Ra)
    print('q:', q)
    print('-------------------------------')
    print('output diagnostics:')
    print('magnetic Reynolds number Rm=', int(Rm))

    #print('Reynolds number', '%.2E' %(Rm/Pm))
    #print('Rossby number', '%.2E' %((2*Rm*Ek)/Pm))


    # kinetic Energy plot
    ax1.plot(tkin, kinEtot, label=r'$\mathcal{E}_{kin}$')
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


     # magnetic Energy plot
    ax2.plot(tmag, magEtot, label=r'$\mathcal{E}_{mag}$')
    ax2.set_yscale('log')
    ax2.set_ylabel('Energy density')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax2.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax2.set_xlim(xlim)        

    if ylim is not None:
       ax2.set_ylim(ylim)    
    ax2.legend()


    #compute time-averaged Energy kinEtot
    startindex = int(0.3 * len(kinEtot))
    timeavg_kinEtot = np.mean(kinEtot[startindex:])
    print(f"time-averaged kinetic energy  = {timeavg_kinEtot:.2e}")

    #compute time-averaged Energy magEtot
    timeavg_magEtot = np.mean(magEtot[startindex:])
    print(f"time-averaged magnetic energy = {timeavg_magEtot:.2e}")

    #Time averaged Elssaser number
    Lambda = 2*timeavg_magEtot
    print(f"Elsasser number Lambda= {Lambda:.2e}")

    # Determine if dynamo is active
    dynamo_threshold = 1e-4
    if min(magEtot) > dynamo_threshold:
        #if  timeavg_magEtot > dynamo_threshold:
        dynamo =  1
    else:
        dynamo =  0
    print(f"Dynamo active: {dynamo}")


    # Dipolarity
    if len(tdip) > 0:
        ax3.plot(tdip, fdip, color='red', label='g10', alpha=0.6)
    ax3.set_ylabel('Dipolarity')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax3.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax3.set_xlim(xlim)  
    ax3.set_ylim(0, 1)

    # compute time-averaged Dipolarity
    timeavg_fdip = np.mean(fdip[startindex:])
    print(f"Time-averaged dipolarity: {timeavg_fdip:.2e}")


    # Dipole latitude
    if len(g10) > 0:
        ax4.plot(tdip, dipangle, 'k', alpha=0.7)
    ax4.set_ylabel('Dipole angle (deg)')
    ax4.set_ylim(-5, 185)
    ax4.axhline(90, color='gray', linestyle='--', alpha=0.4, label=r'$90^{\circ}$')
    #ax3.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax4.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax4.set_xlim(xlim)  
    ax4.legend()


    reversal = 0    
    excursion = 0

    # only use data from startindex to the end
    dipangle_subset = dipangle[startindex:]
    
    # check for excursion
    for i in range(1, len(dipangle_subset)):
        if (dipangle_subset[i-1] < 90 and dipangle_subset[i] >= 90) or \
           (dipangle_subset[i-1] > 90 and dipangle_subset[i] <= 90):
            excursion = 1
            break
    
    # check for reversal 
    # criteria: dipole angle goes beyond 150 deg and below 30 deg, and dipolarity >0.35 (empirical)    
    if excursion == 1:
        if np.max(dipangle_subset) > 150 and np.min(dipangle_subset) < 30:
            if  timeavg_fdip > 0.35:
                reversal = 1

    # ---------- Write summary CSV ---------- #
    # --- determine Ek root automatically ---
    Ek_root = extract_Ek_root(folderFile)
    os.makedirs(Ek_root, exist_ok=True)

    csv_name = f"data_E_{Ek:.1e}.csv"
    csv_path = os.path.join(Ek_root, csv_name)

    write_dynamo_summary_csv(
        csv_path=csv_path,
        q=q,
        Ra=Ra,
        Ek=Ek,
        dynamo=dynamo,
        dipolarity=timeavg_fdip,
        Elsasser=Lambda,
        reversal=reversal,
        Rm=Rm
    )



    # Dissipations (outputs are missed right now, we need implementation in the code)
   
    # if len(tkinDis) > 0:
    #         ax4.plot(tkinDis, Ek*kinDtot, alpha=0.7, label='viscous dissipation')
    #         ax4.plot(tmagDis, magDtot, alpha=0.7,    label='ohmic dissipation')
    #ax5.set_ylabel('Dissipation')
    #ax5.set_xlabel('Time')
    #ax5.set_yscale('log')

    # if xlim is None:
    #     xlim = (np.max(t) - np.min(t)) * 0.05
    #     ax5.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    # else:
    #     ax5.set_xlim(xlim)  
    #     ax5.legend()

    #save figure
    Ek,Ra,q = input_params_from_path(folderFile)

    save_path = os.path.join(save_dir, f'Ek_{Ek}_q{q}_Ra{Ra}_timeseries.png')
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
        axes[0].semilogy(tkin, kinEtot,
                     label=f"{labels[idx]} - Kinetic",
                     #color=colors[idx], 
                     lw=1.8)

        # ---------- Plot magnetic ----------
        axes[1].semilogy(tmag, magEtot,
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


    if xlim is None:
        xlim = (np.max(tkin) - np.min(tkin)) * 0.05
        axes[1].set_xlim(xlim)
    else:
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


##### timestep ######
def read_cfl_file(fname):
    """
    Read a cfl.dat file.
    
    Returns:
        time: array
        dt: array
    """
    time = []
    dt = []

    if not os.path.exists(fname):
        return np.array([]), np.array([])

    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            time.append(float(parts[0]))
            dt.append(float(parts[1]))

    return np.array(time), np.array(dt)



def F_conc_cfl(run_folders):
    """
    Concatenate cfl.dat time series across multiple restart folders.

    This is the CFL analogue of F_conc_timeseries.
    
    Returns:
        time, dt
    """

    time = np.array([])
    dt = np.array([])

    for i, folder in enumerate(run_folders):

        lastval = time[-1] if time.size > 0 else -np.inf

        tnew, dtnew = read_cfl_file(os.path.join(folder, "cfl.dat"))
        if tnew.size == 0:
            continue

        mask = tnew > lastval
        if not np.any(mask):
            continue

        start_idx = np.argmax(mask)

        time = np.concatenate((time, tnew[start_idx:]))
        dt = np.concatenate((dt, dtnew[start_idx:]))

    return time, dt



def compare_timestep_multi(run_paths, labels=None, save_dir=None, show=True, xlim=None, ylim=None):
    """
    Compare CFL time step evolution from multiple runs.

    Args:
        run_paths: list of paths (each can be a single run or a folder with run0, run1,...)
        labels: legend labels
        save_dir: optional output directory
        show: show plot
        xlim: (xmin, xmax)
        ylim: (ymin, ymax)

    Returns:
        fig, ax
    """

    if not isinstance(run_paths, (list, tuple)):
        raise ValueError("run_paths must be a list")

    if labels and len(labels) != len(run_paths):
        raise ValueError("labels length must match run_paths")

    if labels is None:
        labels = [os.path.basename(p.rstrip('/')) for p in run_paths]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_paths)))

    for idx, path in enumerate(run_paths):

        # discover run folders
        if os.path.isdir(path):
            runs = sorted(glob.iglob(os.path.join(path, "run*")))
            if len(runs) == 0:
                runs = [path]
        else:
            runs = [path]

        # read concatenated CFL
        t, dt = F_conc_cfl(runs)

        if t.size == 0:
            print(f"Warning: no cfl.dat found in {path}")
            continue

        # plot
        ax.semilogy(t, dt, lw=1.8, label=labels[idx])

    # style
    ax.set_xlabel("Time")
    ax.set_ylabel("Timestep (dt)")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=7)

    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    # save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, "compare_timestep_multi.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved: {fname}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot timeseries from QuICC runs.")
    parser.add_argument("folder", help="Path to the start folder containing run subfolders")
    parser.add_argument("--save", help="Path to save the figure", default=None)
    args = parser.parse_args()

    plot_timeseries(start_folder=args.folder, show=True, save_path=args.save)
    
    # TO RUN it in terminal:
    #python -m quicc_dynavis.timeseries /path/to/my/simulation --save timeseries.png

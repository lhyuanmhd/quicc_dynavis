import csv
import glob
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator

from .io import (
    F_conc_timeseries,
    get_boundary_conditions,
    get_parameters,
    get_resolution,
)

matplotlib.rcParams["mathtext.fontset"] = "cm"

from .timeseries_utils import (
    discover_run_folders,
    extract_ek_root,
    input_params_from_path,
    safe_conc_timeseries,
)
_discover_run_folders = discover_run_folders
extract_Ek_root = extract_ek_root

from .summary import write_dynamo_summary_csv
from .timeseries_data import load_timeseries_data, load_hydro_timeseries_data
from .timeseries_diagnostics import (
    compute_dynamo_diagnostics,
    print_dynamo_diagnostics,
    compute_hydro_diagnostics,
    print_hydro_diagnostics
)

from .timeseries_plotting import (
    create_timeseries_figure,
    populate_timeseries_figure,
    resolve_time_limits,
)

from .timeseries_plotting import (
    create_dipolarity_figure,
    populate_dipolarity_figure,
    resolve_time_limits,
)

from .timeseries_plotting import (
    create_hydro_timeseries_figure,
    populate_hydro_timeseries_figure,
    resolve_time_limits,
)



# keep updating according to the need
def plot_timeseries(folderFile, save_dir, show=True, xlim=None, ylim=None):
    """
    Plot dynamo time-series diagnostics for one simulation case.

    Parameters
    ----------
    folderFile
        Directory containing numbered run folders such as run0 and run1.
    save_dir
        Directory in which the figure is saved.
    show
        Display the figure interactively when True.
    xlim
        Optional time-axis limits ``(xmin, xmax)``.
    ylim
        Optional energy-axis limits applied to the energy panels.

    Returns
    -------
    fig
        Matplotlib figure.
    axes
        Tuple of Matplotlib axes.
    """

    # read data
    data = load_timeseries_data(folderFile)

    # diagnostic 
    diagnostics = compute_dynamo_diagnostics(data)
    print_dynamo_diagnostics(data, diagnostics)
    
    # make plots
    time_limits = resolve_time_limits(data.tkin, xlim=xlim)

    fig, axes = create_timeseries_figure(
        has_dissipation=data.has_dissipation,
    )
    populate_timeseries_figure(
        axes=axes,
        data=data,
        diagnostics=diagnostics,
        time_limits=time_limits,
        ylim=ylim,
    )

    # ---------- Write summary CSV ---------- #
    Ek_root = extract_Ek_root(folderFile)
    csv_path = os.path.join(
        Ek_root,
        "diagnostics",
        f"data_E_{data.Ek:.1e}.csv",
    )


    write_dynamo_summary_csv(
        csv_path=csv_path,
        q=data.q,
        Ra=data.Ra,
        Ek=data.Ek,
        E0mag=diagnostics.initial_magnetic_energy,
        dynamo=int(diagnostics.dynamo_active),
        dipolarity=diagnostics.mean_dipolarity,
        Elsasser=diagnostics.elsasser_number,
        visDis=diagnostics.mean_viscous_dissipation,
        ohmDis=diagnostics.mean_ohmic_dissipation,
        fohm=diagnostics.ohmic_fraction,
        L_u=diagnostics.velocity_length_scale,
        L_b=diagnostics.magnetic_length_scale,
        T_perb=diagnostics.mean_thermal_perturbation,
        nusselt=diagnostics.mean_nusselt,
        reversal=int(diagnostics.reversal),
        Rm=diagnostics.magnetic_reynolds_number,
        relative_std_fdip=diagnostics.relative_std_dipolarity,
        bc_mag=data.bc_mag,
        bc_temp=data.bc_temp,
        bc_vel=data.bc_vel,
        N=data.N,
        M=data.M,
        L=data.L,
    )

    #save figure
    save_dir = os.fspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"Ek_{data.Ek}_q{data.q}_Ra{data.Ra}_timeseries.png",
    )

    fig.savefig(
        save_path,
        dpi=270,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    return fig, axes
#-----------------------------------------

# plot compact Emag, dipolarity, dipole angle figure
def plot_timeseries_dipolarity(
    folderFile,
    save_dir,
    show=True,
    xlim=None,
    ylim=None,
):
    """Plot magnetic energy, dipolarity, and dipole tilt angle."""
    data = load_timeseries_data(folderFile)

    diagnostics = compute_dynamo_diagnostics(data)
    print_dynamo_diagnostics(data, diagnostics)

    time_limits = resolve_time_limits(
        data.tkin,
        xlim=xlim,
    )

    fig, axes = create_dipolarity_figure()

    populate_dipolarity_figure(
        axes=axes,
        data=data,
        diagnostics=diagnostics,
        time_limits=time_limits,
        ylim=ylim,
    )

    save_dir = os.fspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        (
            f"Ek_{data.Ek}_q{data.q}_Ra{data.Ra}"
            "_timeseries_Emag_fdip_tiltAngle.png"
        ),
    )

    fig.savefig(
        save_path,
        dpi=270,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    return fig, axes

# plot Ekin, Etemp, and dissipation for hydro case
def plot_timeseries_hydro(
    folderFile,
    save_dir,
    show=True,
    xlim=None,
    ylim=None,
):
    """Plot kinetic, thermal, and dissipation time series for a hydro case."""
    data = load_hydro_timeseries_data(folderFile)

    diagnostics = compute_hydro_diagnostics(data)
    print_hydro_diagnostics(data, diagnostics)

    time_limits = resolve_time_limits(
        data.tkin,
        xlim=xlim,
    )

    fig, axes = create_hydro_timeseries_figure(
        has_dissipation=data.has_dissipation,
    )

    populate_hydro_timeseries_figure(
        axes=axes,
        data=data,
        diagnostics=diagnostics,
        time_limits=time_limits,
        ylim=ylim,
    )

    save_dir = os.fspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"Ek_{data.Ek}_Ra{data.Ra}_timeseries.png",
    )

    fig.savefig(
        save_path,
        dpi=270,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    return fig, axes


#---------------------------------------
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



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot timeseries from QuICC runs.")
    parser.add_argument("folder", help="Path to the start folder containing run subfolders")
    parser.add_argument("--save", help="Path to save the figure", default=None)
    args = parser.parse_args()

    plot_timeseries(start_folder=args.folder, show=True, save_path=args.save)
    
    # TO RUN it in terminal:
    #python -m quicc_dynavis.timeseries /path/to/my/simulation --save timeseries.png

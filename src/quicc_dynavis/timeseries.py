import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["mathtext.fontset"] = "cm"
from .timeseries_utils import (
    extract_ek_root,
)
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
    Ek_root = extract_ek_root(folderFile)
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
        L_u=diagnostics.vel_dis_length_scale,
        L_b=diagnostics.mag_dis_length_scale,
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
        Pm=data.Pm,
        Pr=data.Pr,
    )


    #save figure
    save_dir = os.fspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"Ek_{data.Ek:.1e}_q{data.q:.3g}_Ra_{data.Ra:.2e}_timeseries.png",
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
   #print_dynamo_diagnostics(data, diagnostics)

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
            #f"Ek_{data.Ek}_q{data.q}_Ra{data.Ra}"
            f"Ek_{data.Ek:.1e}_q{data.q:.3g}_Ra_{data.Ra:.2e}"
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
        #f"Ek_{data.Ek}_Ra{data.Ra}_timeseries.png",
        f"Ek_{data.Ek:.1e}_Ra_{data.Ra:.2e}_timeseries.png",
    )

    fig.savefig(
        save_path,
        dpi=270,
        bbox_inches="tight",
    )

    if show:
        plt.show()

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

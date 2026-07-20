from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .timeseries_utils import (
    discover_run_folders,
    safe_conc_timeseries
)

from .io import (
    get_parameters,
)


def resolve_combined_time_limits(
    time_arrays,
    xlim=None,
    margin_fraction=0.05,
):
    """Resolve plotting limits across several time arrays."""
    if xlim is not None:
        return tuple(xlim)

    valid_arrays = [
        np.asarray(time)
        for time in time_arrays
        if len(time) > 0
    ]

    if not valid_arrays:
        raise ValueError(
            "Cannot resolve time limits from empty arrays"
        )

    xmin = min(np.min(time) for time in valid_arrays)
    xmax = max(np.max(time) for time in valid_arrays)

    span = xmax - xmin
    margin = margin_fraction * span if span > 0 else margin_fraction

    return xmin - margin, xmax + margin

def _resolve_run_folders(path) -> list[str]:
    """Resolve a case directory or an individual run directory."""
    path = Path(path)

    if not path.is_dir():
        raise FileNotFoundError(
            f"Simulation path does not exist: {path}"
        )

    if (path / "parameters.cfg").is_file():
        return [str(path)]

    run_folders = discover_run_folders(path)

    if not run_folders:
        raise FileNotFoundError(
            f"No run folders found under {path}"
        )

    return [str(folder) for folder in run_folders]

def compare_energy_multi(
    run_paths,
    labels=None,
    save_dir=None,
    show=True,
    xlim=None,
    ylim=None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Compare kinetic and magnetic energy histories across cases.

    Parameters
    ----------
    run_paths
        Sequence of case directories or individual run directories.
    labels
        Optional legend labels. When omitted, directory names are used.
    save_dir
        Optional output directory.
    show
        Display the figure interactively when True.
    xlim
        Optional time limits ``(xmin, xmax)``.
    ylim
        Optional energy limits applied to both panels.

    Returns
    -------
    fig, axes
        Matplotlib figure and kinetic/magnetic axes.
    """
    if isinstance(run_paths, (str, Path)):
        raise TypeError(
            "run_paths must be a sequence of paths, "
            "not a single path"
        )

    run_paths = list(run_paths)

    if not run_paths:
        raise ValueError("run_paths cannot be empty")

    if labels is None:
        labels = [
            Path(path).resolve().name
            for path in run_paths
        ]
    else:
        labels = list(labels)

        if len(labels) != len(run_paths):
            raise ValueError(
                "labels length must match run_paths length"
            )

    fig, axes_array = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        dpi=180,
    )

    kinetic_ax, magnetic_ax = axes_array
    axes = (kinetic_ax, magnetic_ax)

    all_times = []
    kinetic_count = 0
    magnetic_count = 0

    for path, label in zip(run_paths, labels):
        run_folders = _resolve_run_folders(path)

        tkin, kinetic_energy, _, _ = safe_conc_timeseries(
            run_folders,
            "kinE",
        )

        tmag, magnetic_energy, _, _ = safe_conc_timeseries(
            run_folders,
            "magE",
        )

        parameter_file = (
            Path(run_folders[0]) / "parameters.cfg"
        )

        Ek, Pm, Pr, q, Ra, Ro = get_parameters(
            str(parameter_file),
            "no",
        )

        kinetic_energy = np.asarray(
            kinetic_energy,
            dtype=float,
        ).copy()

        if Pm != 0 and Pr != 0:
            kinetic_energy *= Ek / Pm

        if len(tkin) > 0:
            kinetic_ax.semilogy(
                tkin,
                kinetic_energy,
                linewidth=1.8,
                label=label,
            )
            all_times.append(np.asarray(tkin))
            kinetic_count += 1
        else:
            print(
                f"Warning: no kinetic-energy data found in {path}"
            )

        if len(tmag) > 0:
            magnetic_ax.semilogy(
                tmag,
                magnetic_energy,
                linewidth=1.8,
                label=label,
            )
            all_times.append(np.asarray(tmag))
            magnetic_count += 1
        else:
            print(
                f"Warning: no magnetic-energy data found in {path}"
            )

    kinetic_ax.set_ylabel("Kinetic energy")
    magnetic_ax.set_ylabel("Magnetic energy")
    magnetic_ax.set_xlabel("Time")

    for ax in axes:
        ax.grid(alpha=0.35)

        if ylim is not None:
            ax.set_ylim(ylim)

    if kinetic_count > 0:
        kinetic_ax.legend(fontsize=7)

    if magnetic_count > 0:
        magnetic_ax.legend(fontsize=7)

    if all_times:
        time_limits = resolve_combined_time_limits(
            all_times,
            xlim=xlim,
        )
        magnetic_ax.set_xlim(time_limits)
    elif xlim is not None:
        magnetic_ax.set_xlim(xlim)   

    fig.subplots_adjust(hspace=0.25)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "compare_energy_multi.png"

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved ✅: {save_path}")

    if show:
        plt.show()

    return fig, axes



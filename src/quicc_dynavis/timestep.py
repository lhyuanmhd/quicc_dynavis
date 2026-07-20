"""Read and compare QuICC CFL timestep histories."""

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .timeseries_utils import discover_run_folders


def read_cfl_file(file_path) -> tuple[np.ndarray, np.ndarray]:
    """Read time and timestep values from a QuICC ``cfl.dat`` file.

    Parameters
    ----------
    file_path
        Path to a ``cfl.dat`` file.

    Returns
    -------
    time
        Simulation times.
    timestep
        Corresponding timestep values.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        return np.array([], dtype=float), np.array([], dtype=float)

    time = []
    timestep = []

    with file_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) < 2:
                continue

            try:
                time_value = float(fields[0])
                timestep_value = float(fields[1])
            except ValueError:
                print(
                    f"Warning: invalid CFL entry in "
                    f"{file_path}:{line_number}"
                )
                continue

            time.append(time_value)
            timestep.append(timestep_value)

    return (
        np.asarray(time, dtype=float),
        np.asarray(timestep, dtype=float),
    )


def concatenate_cfl(
    run_folders: Sequence[str | Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate CFL histories across restarted simulation runs.

    Values whose times overlap an earlier run are discarded.
    """
    time_parts = []
    timestep_parts = []

    last_time = -np.inf

    for folder in run_folders:
        folder = Path(folder)

        time_new, timestep_new = read_cfl_file(
            folder / "cfl.dat"
        )

        if time_new.size == 0:
            continue

        mask = time_new > last_time

        if not np.any(mask):
            continue

        first_new_index = int(np.flatnonzero(mask)[0])

        accepted_time = time_new[first_new_index:]
        accepted_timestep = timestep_new[first_new_index:]

        time_parts.append(accepted_time)
        timestep_parts.append(accepted_timestep)

        last_time = accepted_time[-1]

    if not time_parts:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    return (
        np.concatenate(time_parts),
        np.concatenate(timestep_parts),
    )


def _resolve_cfl_run_folders(path) -> list[str]:
    """Resolve either a case directory or a single run directory."""
    path = Path(path)

    if not path.is_dir():
        raise FileNotFoundError(
            f"Timestep path does not exist: {path}"
        )

    if (path / "cfl.dat").is_file():
        return [str(path)]

    run_folders = discover_run_folders(path)

    if not run_folders:
        return [str(path)]

    return run_folders


def compare_timesteps(
    run_paths,
    labels=None,
    save_dir=None,
    show=True,
    xlim=None,
    ylim=None,
) -> tuple[Figure, Axes]:
    """Compare CFL timestep histories from multiple simulation cases.

    Parameters
    ----------
    run_paths
        Sequence of case directories or individual run directories.
    labels
        Optional legend labels. Must match the number of paths.
    save_dir
        Optional directory in which to save the figure.
    show
        Display the figure interactively when True.
    xlim
        Optional time-axis limits.
    ylim
        Optional timestep-axis limits.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
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

    fig, ax = plt.subplots(
        figsize=(8, 4),
        dpi=180,
    )

    plotted_count = 0

    for path, label in zip(run_paths, labels):
        run_folders = _resolve_cfl_run_folders(path)
        time, timestep = concatenate_cfl(run_folders)

        if time.size == 0:
            print(f"Warning: no CFL data found in {path}")
            continue

        ax.semilogy(
            time,
            timestep,
            linewidth=1.8,
            label=label,
        )
        plotted_count += 1

    ax.set_xlabel("Time")
    ax.set_ylabel("Timestep")
    ax.grid(alpha=0.35)

    if plotted_count > 0:
        ax.legend(fontsize=7)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "compare_timesteps.png"

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig, ax


# tet:
# from quicc_dynavis.timestep import compare_timesteps

# fig, ax = compare_timesteps(
#     [
#         "/Users/yuanlonghui/ETH_project/IlessDyn/Fixed_flux/E1e-4/q_1/Ra2e3/stfE1e-4q1Ra1e3",
#         "/Users/yuanlonghui/ETH_project/IlessDyn/Fixed_flux/E1e-4/q_1/Ra2e3/stfE1e-4q1Ra1e3",
#     ],
#     labels=[
#         "Case 1",
#         "Case 2",
#     ],
#     save_dir="/tmp/quicc_dynavis_timestep",
#     show=False,
# )

# print("Number of lines:", len(ax.lines))
# print("Timestep comparison succeeded")
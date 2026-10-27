#!/usr/bin/env python3
"""Compare magnetic diagnostics from two QuICC cases.

Package usage:
    python -m quicc_dynavis.compare_magnetic_cases CASE1 CASE2 --save comparison.pdf
"""

import argparse
from pathlib import Path
import re

import numpy as np


# Optional defaults: edit these paths to run without positional arguments.
DEFAULT_CASE1 = None
DEFAULT_CASE2 = None
DEFAULT_LABELS = ("ED-seeded", "AD-seeded")

# Edit these settings to customize the appearance of all three panels.
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "figure.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

FIELD_LATEX = {
    "Lambda": r"\Lambda",
    "Rm": r"\mathrm{Rm}",
    "fdip": r"f_{\mathrm{dip}}",
    "fohm": r"f_{\mathrm{ohm}}",
    "Emag": r"E_{\mathrm{mag}}",
    "Ro": r"\mathrm{Ro}",
    "local_Ro": r"\mathrm{Ro}_{\ell}",
    "flow_degree": r"\bar{\ell}_u",
    "degree_over_pi": r"\bar{\ell}_u/\pi",
    "Nu": r"\mathrm{Nu}",
    "q": r"q",
    "Ra": r"\mathrm{Ra}",
    "Ek": r"E",
    "Pm": r"\mathrm{Pm}",
    "Pr": r"\mathrm{Pr}",
}


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def _sort_unique(time, *values):
    """Sort by time and retain the last input record at duplicate times."""
    time = np.asarray(time, dtype=float).reshape(-1)
    values = [np.asarray(v, dtype=float).reshape(-1) for v in values]
    if not time.size:
        raise ValueError("The loaded time series is empty.")
    if any(v.size != time.size for v in values):
        raise ValueError("Time and value arrays have different lengths.")
    if not np.isfinite(time).all():
        raise ValueError("Time contains NaN or Inf.")
    order = np.argsort(time, kind="stable")
    time = time[order]
    values = [v[order] for v in values]
    keep = np.r_[time[:-1] != time[1:], True]
    return (time[keep], *(v[keep] for v in values))


def load_magnetic_case(case_dir):
    """Load every restart directory beneath a case or runs directory.

    Files must contain cumulative simulation times. File parsing and restart
    concatenation follow quicc_dynavis.io.F_conc_timeseries.
    """
    from quicc_dynavis.io import F_conc_timeseries

    case_dir = Path(case_dir).expanduser().resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {case_dir}")
    names = ("magnetic_energy.dat", "magnetic_dipolarity.dat")
    folders = set()
    for name in names:
        matches = list(case_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"No {name} files found under {case_dir}")
        folders.update(p.parent for p in matches)
    folders = sorted(folders, key=_natural_key)
    for folder in folders:
        for name in names:
            path = folder / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Required file is missing or empty: {path}")

    print(f"\n{case_dir}: {len(folders)} restart directories")
    for folder in folders:
        print(f"  {folder.relative_to(case_dir)}")
    run_folders = [str(folder) for folder in folders]
    magnetic = F_conc_timeseries(run_folders, "magE")
    dipole = F_conc_timeseries(run_folders, "Dip")
    tmag, mag_total, mag_tor, mag_pol = _sort_unique(*magnetic)
    tdip, fdip, g10, g11, h11 = _sort_unique(*dipole)
    equatorial = np.hypot(g11, h11)
    tilt = np.degrees(np.arctan2(equatorial, g10))
    tilt[np.hypot(g10, equatorial) == 0] = np.nan
    print(f"  Energy: {len(tmag)} samples; t = {tmag[0]:.6g} to {tmag[-1]:.6g}")
    print(f"  Dipole: {len(tdip)} samples; t = {tdip[0]:.6g} to {tdip[-1]:.6g}")
    return dict(
        case_dir=case_dir, run_folders=run_folders,
        tmag=tmag, mag_total=mag_total, mag_tor=mag_tor, mag_pol=mag_pol,
        tdip=tdip, fdip=fdip, g10=g10, g11=g11, h11=h11,
        equatorial_dipole=equatorial, tilt=tilt,
    )


def plot_magnetic_panels(
    case_dir1, case_dir2, *, labels=DEFAULT_LABELS,
    colors=("tab:blue", "tab:orange"), figsize=(10, 8),
    energy_log=True, time_offsets=(0.0, 0.0), xlim=None,
    title=None, save=None, show=True,
):
    """Compare energy, dipole components and tilt; return fig, axes, datasets.

    Tilt is measured from the positive rotation axis using
    atan2(sqrt(g11**2 + h11**2), g10), in degrees. Time offsets are subtracted
    only for plotting; returned data retain their original times and units.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(PLOT_STYLE)

    if any(len(v) != 2 for v in (labels, colors, time_offsets)):
        raise ValueError("labels, colors and time_offsets must have two items.")
    datasets = tuple(load_magnetic_case(p) for p in (case_dir1, case_dir2))
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, constrained_layout=True,
    )
    for data, label, color, offset in zip(datasets, labels, colors, time_offsets):
        tmag = data["tmag"] - offset
        tdip = data["tdip"] - offset
        # Use the same color for a case throughout all panels.
        axes[0].plot(tmag, data["mag_total"], color=color, lw=1.5, label=label)
        axes[1].semilogy(
            tdip, data["g10"], color=color, lw=1.5,
            label=rf"{label}: $g_{{10}}$",
        )
        axes[1].semilogy(
            tdip, data["equatorial_dipole"], color=color, ls="--", lw=1.5,
            label=rf"{label}: $\sqrt{{g_{{11}}^2+h_{{11}}^2}}$",
        )
        axes[2].plot(tdip, data["tilt"], color=color, lw=1.5)
    axes[0].set_ylabel(f"${FIELD_LATEX['Emag']}$")
    axes[0].legend(frameon=False)
    if energy_log:
        axes[0].set_yscale("log")
    axes[1].set_ylabel("Dipole coefficients")
    axes[1].axhline(0, color="0.5", ls=":", lw=0.8)
    axes[1].legend(frameon=False, ncol=2)
    axes[2].set_ylabel(r"Tilt angle ($^\circ$)")
    axes[2].set_ylim(0, 180)
    axes[2].set_yticks([0, 45, 90, 135, 180])
    axes[2].axhline(90, color="0.5", ls=":", lw=0.8)
    axes[2].set_xlabel(r"$\rm time$")
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ax.text(0.015, 0.95, tag, transform=ax.transAxes, va="top", fontsize=15)
        ax.tick_params(which="major", length=5, width=1.0)
        ax.grid(alpha=0.15, linewidth=0.7)
    if xlim is not None:
        axes[2].set_xlim(*xlim)
    if title:
        fig.suptitle(title)
    if save:
        output = Path(save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output.resolve()}")
    if show:
        plt.show()
    return fig, axes, datasets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case1", nargs="?", default=DEFAULT_CASE1)
    parser.add_argument("case2", nargs="?", default=DEFAULT_CASE2)
    parser.add_argument("--labels", nargs=2, default=DEFAULT_LABELS)
    parser.add_argument("--colors", nargs=2, default=("tab:blue", "tab:orange"))
    parser.add_argument("--save", type=Path, help="Output PDF, PNG, or SVG path.")
    parser.add_argument("--title")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("TMIN", "TMAX"))
    parser.add_argument("--time-offsets", nargs=2, type=float, default=(0, 0))
    parser.add_argument("--figsize", nargs=2, type=float, default=(10, 8))
    parser.add_argument("--linear-energy", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="Use a headless backend.")
    args = parser.parse_args(argv)
    if args.case1 is None or args.case2 is None:
        parser.error("Provide two case paths or set DEFAULT_CASE1 and DEFAULT_CASE2.")
    if args.no_show and args.save is None:
        parser.error("--no-show requires --save so the figure is retained.")
    # Select the headless backend before importing pyplot.
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    try:
        plot_magnetic_panels(
            args.case1, args.case2, labels=args.labels, colors=args.colors,
            figsize=args.figsize, energy_log=not args.linear_energy,
            time_offsets=args.time_offsets, xlim=args.xlim, title=args.title,
            save=args.save, show=not args.no_show,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

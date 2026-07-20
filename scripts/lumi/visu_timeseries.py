#!/usr/bin/env python3

import argparse
import contextlib
import io as pyio
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

from quicc_dynavis import io
from quicc_dynavis.timeseries import (
    plot_timeseries,
    plot_timeseries_dipolarity,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate time-series figures and a simulation summary."
    )
    parser.add_argument(
        "case_dir",
        nargs="?",
        default=".",
        help="Simulation case directory (default: current directory)",
    )
    return parser.parse_args()


def find_run0_directory(runs_dir: Path) -> Path:
    """Return the first directory named run0, run00, run000, etc."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    candidates = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"run0+", path.name)
    )

    if not candidates:
        raise FileNotFoundError(
            f"No run0-style directory found under {runs_dir}"
        )

    return candidates[0]


def write_simulation_summary(param_file: Path, summary_file: Path) -> None:
    """Write the simulation summary generated from a parameter file."""
    if not param_file.is_file():
        raise FileNotFoundError(f"Parameter file does not exist: {param_file}")

    buffer = pyio.StringIO()
    with contextlib.redirect_stdout(buffer):
        io.print_simulation_summary(str(param_file))

    summary_file.write_text(buffer.getvalue(), encoding="utf-8")


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()

    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_timeseries(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        show=False,
    )
    plot_timeseries_dipolarity(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        show=False,
    )

    print(f"[OK] Time-series figures saved under {fig_dir}")

    diag_dir = case_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    run0_dir = find_run0_directory(case_dir / "runs")
    param_file = run0_dir / "parameters.cfg"
    summary_file = diag_dir / "run_summary.txt"

    write_simulation_summary(param_file, summary_file)

    print(f"[OK] Simulation summary written to {summary_file}")


if __name__ == "__main__":
    main()

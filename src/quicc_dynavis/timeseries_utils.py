"""Utility functions for locating and reading QuICC time-series data."""

from pathlib import Path
import re

import numpy as np

from .io import F_conc_timeseries, get_parameters



_EMPTY_TIMESERIES = (
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
)


def safe_conc_timeseries(run_folders, tag):
    """Read and concatenate a time series, returning empty arrays if unavailable."""
    try:
        output = F_conc_timeseries(run_folders, tag)

        if output is None:
            return _EMPTY_TIMESERIES

        time, total, toroidal, poloidal = output

        if time is None or len(time) == 0:
            return _EMPTY_TIMESERIES

        return time, total, toroidal, poloidal

    except (FileNotFoundError, OSError, ValueError, IndexError) as exc:
        print(
            f"[WARN] Timeseries '{tag}' was not found or could not be read: "
            f"{exc}"
        )
        return _EMPTY_TIMESERIES


def discover_run_folders(case_dir):
    """Return numeric run directories, supporting new and legacy layouts."""
    case_path = Path(case_dir)

    candidate_roots = (
        case_path / "runs",
        case_path,
    )

    for root in candidate_roots:
        if not root.is_dir():
            continue

        run_folders = sorted(
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name.startswith("run")
            and path.name[3:].isdigit()
        )

        if run_folders:
            return [str(path) for path in run_folders]

    return []


def input_params_from_path(case_dir):
    """Read Ekman number, q, and Rayleigh number from the initial run."""
    case_path = Path(case_dir)
    runs_dir = case_path / "runs"

    candidates = (
        runs_dir / "run0",
        runs_dir / "run000",
        case_path / "run0",
        case_path / "run000",
    )

    run_dir = next((path for path in candidates if path.is_dir()), None)

    if run_dir is None:
        raise FileNotFoundError(
            f"No run0 or run000 directory found under {case_path}"
        )

    ekman, _, _, q_value, rayleigh, _ = get_parameters(
        str(run_dir / "parameters.cfg"),
        "no",
    )

    return (
        f"{ekman:.1e}",
        f"{q_value:.1e}",
        f"{rayleigh:.1e}",
    )


def extract_ek_root(path):
    """Return the path ending at the first Ekman-number directory."""
    input_path = Path(path)

    pattern = re.compile(r"E\d+(?:\.\d+)?e[-+]?\d+")

    parts = input_path.parts
    for index, part in enumerate(parts):
        if pattern.fullmatch(part):
            return str(Path(*parts[: index + 1]))

    raise ValueError(f"Could not find an Ekman-number directory in: {path}")


from pathlib import Path

import numpy as np
import pandas as pd

from quicc_dynavis.io import F_conc_timeseries


ROOT = Path(
    "/scratch/project_465001528/lhyuan/"
    "IlessDyn/CattaneoHuges/bifurcation"
)

HERE = Path(__file__).resolve().parent
SEGMENTS = HERE / "segments.csv"
OUTPUT = HERE / "bifurcation_summary.csv"



import re


def discover_cases():
    """Index standard bifurcation cases."""

    cases = {}

    for branch in ("q_up_from_ED", "q_down_from_AD"):
        for path in (ROOT / branch).glob("E*/Ra*/q_*"):

            # Only accept q_1.35, q_1.75, etc.
            match = re.fullmatch(r"q_(\d+(?:\.\d+)?)", path.name)

            if match is None:
                print(f"Skipping special case: {path}")
                continue

            E = float(path.parts[-3][1:])
            Ra = float(path.parts[-2][2:])
            q = float(match.group(1))

            key = (branch, E, Ra, q)

            if key in cases:
                raise ValueError(f"Duplicate case: {key}")

            cases[key] = path

    return cases


def find_run_dirs(case_dir):
    """Find the runXXX directories belonging to one simulation."""

    groups = []

    for runs_dir in case_dir.rglob("runs"):
        run_dirs = sorted(
            p for p in runs_dir.glob("run[0-9]*")
            if p.is_dir() and p.name[3:].isdigit()
        )

        if run_dirs:
            groups.append(run_dirs)

    if len(groups) != 1:
        raise ValueError(
            f"Expected one runs group in {case_dir}, "
            f"found {len(groups)}"
        )

    return [str(p) for p in groups[0]]


def window_stats(t, y, start, end):
    """Statistics within a manually selected time window."""

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if not start < end:
        raise ValueError("t_start must be smaller than t_end")

    mask = (
        (t >= start)
        & (t <= end)
        & np.isfinite(t)
        & np.isfinite(y)
    )

    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"Insufficient data in [{start}, {end}]"
        )

    return {
        "mean": float(np.mean(y[mask])),
        "std": float(np.std(y[mask])),
        "n_samples": int(np.count_nonzero(mask)),
    }


def main():

    segments = pd.read_csv(SEGMENTS)
    cases = discover_cases()

    output_rows = []
    cache = {}

    for row in segments.itertuples(index=False):

        key = (
            row.branch,
            float(row.E),
            float(row.Ra),
            float(row.q),
        )

        if key not in cases:
            raise FileNotFoundError(f"Case not found: {key}")

        case_dir = cases[key]

        # Read each simulation only once
        if key not in cache:
            run_dirs = find_run_dirs(case_dir)

            t, emag, _, _ = F_conc_timeseries(
                run_dirs, "magE"
            )

            cache[key] = (t, emag)

        t, emag = cache[key]

        stats = window_stats(
            t, emag, row.t_start, row.t_end
        )

        output_rows.append({
            "branch": row.branch,
            "E": row.E,
            "Ra": row.Ra,
            "q": row.q,
            "state": row.state,
            "phase": row.phase,
            "t_start": row.t_start,
            "t_end": row.t_end,
            "time_span": row.t_end - row.t_start,
            "Emag_mean": stats["mean"],
            "Emag_std": stats["std"],
            "n_samples": stats["n_samples"],
        })

        print(
            f"{row.branch} q={row.q}, "
            f"{row.state}: "
            f"Emag={stats['mean']:.5e}"
        )

    pd.DataFrame(output_rows).to_csv(
        OUTPUT, index=False
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

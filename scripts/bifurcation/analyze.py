
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

        # Load each simulation only once
        if key not in cache:
            run_dirs = find_run_dirs(cases[key])

            tmag, emag, _, _ = F_conc_timeseries(
                run_dirs, "magE"
            )
            tkin, ekin, _, _ = F_conc_timeseries(
                run_dirs, "kinE"
            )
            ttem, etem = F_conc_timeseries(
                run_dirs, "temE"
            )
            tdip, fdip, g10, g11, h11 = F_conc_timeseries(
                run_dirs, "Dip"
            )

            # Consistent with timeseries_diagnostics.py
            denominator = np.sqrt(g10**2 + g11**2 + h11**2)
            ratio = np.divide(
                g10,
                denominator,
                out=np.full_like(g10, np.nan, dtype=float),
                where=denominator > 0,
            )
            dipole_tilt = np.degrees(
                np.arccos(np.clip(ratio, -1.0, 1.0))
            )

            cache[key] = {
                "mag": (tmag, emag),
                "kin": (tkin, ekin),
                "tem": (ttem, etem),
                "fdip": (tdip, fdip),
                "g10_squared": (tdip, g10**2),
                "deq_squared": (tdip, g11**2 + h11**2),
                "tilt": (tdip, dipole_tilt),
            }

        data = cache[key]
        start = float(row.t_start)
        end = float(row.t_end)

        mag = window_stats(*data["mag"], start, end)
        kin = window_stats(*data["kin"], start, end)
        tem = window_stats(*data["tem"], start, end)
        fdip = window_stats(*data["fdip"], start, end)
        tilt = window_stats(*data["tilt"], start, end)

        g10_sq = window_stats(
            *data["g10_squared"], start, end
        )
        deq_sq = window_stats(
            *data["deq_squared"], start, end
        )

        q = float(row.q)
        Ra = float(row.Ra)

        # Same definitions as compute_dynamo_diagnostics()
        Lambda = 2.0 * mag["mean"]
        Rm = np.sqrt(kin["mean"])
        Pe = Rm / q

        thermal_factor = 3.0 / 5.0 * (q * Ra)**2
        T_perb = thermal_factor * tem["mean"]

        output_rows.append({
            "branch": row.branch,
            "E": row.E,
            "Ra": Ra,
            "q": q,
            "state": row.state,
            "phase": row.phase,
            "t_start": start,
            "t_end": end,
            "time_span": end - start,

            # Magnetic diagnostics
            "Emag_mean": mag["mean"],
            "Emag_std": mag["std"],
            "Lambda": Lambda,
            "Lambda_std": 2.0 * mag["std"],

            # Flow diagnostics
            "Ekin_mean": kin["mean"],
            "Ekin_std": kin["std"],
            "Rm": Rm,
            "Pe": Pe,

            # Temperature diagnostics
            "Tem_mean": tem["mean"],
            "Tem_std": tem["std"],
            "T_perb": T_perb,
            "T_perb_std": thermal_factor * tem["std"],

            # Dipole diagnostics
            "fdip_mean": fdip["mean"],
            "fdip_std": fdip["std"],
            "g10_rms": np.sqrt(g10_sq["mean"]),
            "deq_rms": np.sqrt(deq_sq["mean"]),
            "dipole_tilt_mean": tilt["mean"],

            "n_samples": mag["n_samples"],
        })

        print(
            f"{row.branch:16s} "
            f"q={q:.3f} "
            f"state={row.state:3s} "
            f"Lambda={Lambda:.4e} "
            f"Rm={Rm:.2f} "
            f"Pe={Pe:.2f} "
            f"T_perb={T_perb:.5e}"
        )

    result = pd.DataFrame(output_rows)
    result.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

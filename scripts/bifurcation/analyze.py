
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


def fit_g10_rate(t, g10, t_start, t_end):
    """
    Fit ln|g10| = intercept + sigma * t.

    The selected interval must have no polarity crossing
    and should exhibit an approximately exponential trend.
    """
    t = np.asarray(t, dtype=float)
    g10 = np.asarray(g10, dtype=float)

    mask = (
        (t >= t_start)
        & (t <= t_end)
        & np.isfinite(t)
        & np.isfinite(g10)
    )

    t_fit = t[mask]
    g_fit = g10[mask]

    order = np.argsort(t_fit)
    t_fit = t_fit[order]
    g_fit = g_fit[order]

    if len(t_fit) < 8:
        raise ValueError("Insufficient points for g10 fitting")

    if np.any(g_fit == 0):
        raise ValueError("g10 reaches zero in fitting window")

    if np.any(np.sign(g_fit[1:]) != np.sign(g_fit[:-1])):
        raise ValueError(
            "g10 changes sign: choose a different fitting window"
        )

    log_g10 = np.log(np.abs(g_fit))

    sigma, intercept = np.polyfit(t_fit, log_g10, 1)

    prediction = sigma * t_fit + intercept
    residual = log_g10 - prediction

    ss_res = np.sum(residual**2)
    ss_tot = np.sum((log_g10 - log_g10.mean())**2)

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0 else np.nan
    )

    sxx = np.sum((t_fit - t_fit.mean())**2)

    sigma_fit_se = np.sqrt(
        ss_res / (len(t_fit) - 2) / sxx
    )

    return {
        "sigma_g10": float(sigma),
        "sigma_fit_se": float(sigma_fit_se),
        "sigma_r2": float(r_squared),
        "sigma_npoints": len(t_fit),
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
                # Magnetic energy
                "mag": (tmag, emag),
                "mag_tor": (tmag, emag_tor),
                "mag_pol": (tmag, emag_pol),

                # Kinetic energy
                "kin": (tkin, ekin),
                "kin_tor": (tkin, ekin_tor),
                "kin_pol": (tkin, ekin_pol),

                # Temperature
                "tem": (ttem, etem),

                # Dipole diagnostics
                "fdip": (tdip, fdip),
                "g10": (tdip, g10),
                "g10_squared": (tdip, g10**2),
                "deq_squared": (tdip, g11**2 + h11**2),
                "tilt": (tdip, dipole_tilt),
            }


        data = cache[key]
        start = float(row.t_start)
        end = float(row.t_end)

        mag = window_stats(*data["mag"], start, end)
        kin = window_stats(*data["kin"], start, end)

        mag_tor = window_stats(*data["mag_tor"], start, end)
        mag_pol = window_stats(*data["mag_pol"], start, end)

        kin_tor = window_stats(*data["kin_tor"], start, end)
        kin_pol = window_stats(*data["kin_pol"], start, end)

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

        
        fit_start = getattr(row, "fit_start", np.nan)
        fit_end = getattr(row, "fit_end", np.nan)

        sigma_stats = {
            "sigma_g10": np.nan,
            "sigma_fit_se": np.nan,
            "sigma_r2": np.nan,
            "sigma_npoints": np.nan,
        }

        if pd.notna(fit_start) or pd.notna(fit_end):
            if pd.isna(fit_start) or pd.isna(fit_end):
                raise ValueError(
                    f"Both fit_start and fit_end are required: {key}"
                )

            if not float(fit_start) < float(fit_end):
                raise ValueError(f"Invalid g10 fitting interval: {key}")

            sigma_stats = fit_g10_rate(
                *data["g10"],
                float(fit_start),
                float(fit_end),
            )

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

            # Magnetic energy decomposition
            "Emag_tor_mean": mag_tor["mean"],
            "Emag_pol_mean": mag_pol["mean"],
            "Emag_tor_std": mag_tor["std"],
            "Emag_pol_std": mag_pol["std"],

            # Kinetic energy decomposition
            "Ekin_tor_mean": kin_tor["mean"],
            "Ekin_pol_mean": kin_pol["mean"],
            "Ekin_tor_std": kin_tor["std"],
            "Ekin_pol_std": kin_pol["std"],

            # Optional energy fractions
            "Emag_tor_fraction": mag_tor["mean"] / mag["mean"],
            "Ekin_tor_fraction": kin_tor["mean"] / kin["mean"],

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

            "sigma_g10": sigma_stats["sigma_g10"],
            "sigma_fit_se": sigma_stats["sigma_fit_se"],
            "sigma_r2": sigma_stats["sigma_r2"],
            "sigma_npoints": sigma_stats["sigma_npoints"],
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

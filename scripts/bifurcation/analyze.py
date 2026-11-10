#!/usr/bin/env python3
"""Summarize manually selected ED/AD intervals, including dissipation diagnostics.

Input: diagnostics/E1e-5/segments.csv
Output: diagnostics/E1e-5/bifurcation_summary.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from quicc_dynavis.io import F_conc_timeseries


ROOT = Path(
    "/scratch/project_465001528/lhyuan/"
    "IlessDyn/CattaneoHuges/bifurcation"
)
RESULTS_DIR = ROOT / "diagnostics/E1e-5"
SEGMENTS = RESULTS_DIR / "segments.csv"
OUTPUT = RESULTS_DIR / "bifurcation_summary.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def discover_cases():
    """Index standard bifurcation cases (excluding non-standard q folders)."""
    cases = {}
    for branch in ("q_up_from_ED", "q_down_from_AD"):
        for path in (ROOT / branch).glob("E*/Ra*/q_*"):
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
            (p for p in runs_dir.glob("run[0-9]*")
             if p.is_dir() and p.name[3:].isdigit()),
            key=lambda p: int(p.name[3:]),
        )
        if run_dirs:
            groups.append(run_dirs)

    if len(groups) != 1:
        raise ValueError(
            f"Expected one runs group in {case_dir}, found {len(groups)}"
        )
    return [str(p) for p in groups[0]]


def window_stats(t, y, start, end):
    """Unweighted mean/std of finite samples within a chosen time window."""
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
    n = int(np.count_nonzero(mask))
    if n < 2:
        raise ValueError(f"Insufficient data in [{start}, {end}]")
    return {
        "mean": float(np.mean(y[mask])),
        "std": float(np.std(y[mask])),
        "n_samples": n,
    }


def fit_g10_rate(t, g10, t_start, t_end):
    """Fit ln|g10| = intercept + sigma*t without polarity crossings."""
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
        raise ValueError("g10 changes sign: choose a different fitting window")

    log_g10 = np.log(np.abs(g_fit))
    sigma, intercept = np.polyfit(t_fit, log_g10, 1)
    residual = log_g10 - (sigma * t_fit + intercept)
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((log_g10 - log_g10.mean())**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sxx = np.sum((t_fit - t_fit.mean())**2)
    sigma_fit_se = np.sqrt(ss_res / (len(t_fit) - 2) / sxx)

    return {
        "sigma_g10": float(sigma),
        "sigma_fit_se": float(sigma_fit_se),
        "sigma_r2": float(r_squared),
        "sigma_npoints": len(t_fit),
    }


def load_scalar_timeseries(run_dirs, name):
    """Return time and total signal for a scalar QuICC diagnostic.

    If a diagnostic also contains component columns, the first signal is
    assumed to be the total, consistent with magE / kinE handling above.
    """
    columns = F_conc_timeseries(run_dirs, name)
    if len(columns) < 2:
        raise ValueError(f"{name} did not return time + data")
    t, y = columns[:2]
    if len(t) != len(y):
        raise ValueError(f"Time/data length mismatch in {name}")
    return t, y


def main():
    segments = pd.read_csv(SEGMENTS)
    cases = discover_cases()
    output_rows = []
    cache = {}

    for row in segments.itertuples(index=False):
        key = (row.branch, float(row.E), float(row.Ra), float(row.q))
        if key not in cases:
            raise FileNotFoundError(f"Case not found: {key}")

        # Load each physical simulation only once, even for multiple segments.
        if key not in cache:
            run_dirs = find_run_dirs(cases[key])
            tmag, emag, emag_tor, emag_pol = F_conc_timeseries(
                run_dirs, "magE"
            )
            tkin, ekin, ekin_tor, ekin_pol = F_conc_timeseries(
                run_dirs, "kinE"
            )
            ttem, etem = F_conc_timeseries(run_dirs, "temE")
            tdip, fdip, g10, g11, h11 = F_conc_timeseries(run_dirs, "Dip")

            # NEW: QuICC output kinDis = viscous dissipation,
            #                   magDis = ohmic dissipation.
            tvis, visdis = load_scalar_timeseries(run_dirs, "kinDis")
            tohm, ohmdis = load_scalar_timeseries(run_dirs, "magDis")

            denominator = np.sqrt(g10**2 + g11**2 + h11**2)
            ratio = np.divide(
                g10,
                denominator,
                out=np.full_like(g10, np.nan, dtype=float),
                where=denominator > 0,
            )
            dipole_tilt = np.degrees(np.arccos(np.clip(ratio, -1.0, 1.0)))

            cache[key] = {
                "mag": (tmag, emag),
                "mag_tor": (tmag, emag_tor),
                "mag_pol": (tmag, emag_pol),
                "kin": (tkin, ekin),
                "kin_tor": (tkin, ekin_tor),
                "kin_pol": (tkin, ekin_pol),
                "tem": (ttem, etem),
                "fdip": (tdip, fdip),
                "g10": (tdip, g10),
                "g10_squared": (tdip, g10**2),
                "deq_squared": (tdip, g11**2 + h11**2),
                "tilt": (tdip, dipole_tilt),
                # NEW: keep their own time arrays; no forced alignment.
                "visDis": (tvis, visdis),
                "ohmDis": (tohm, ohmdis),
            }

        data = cache[key]
        start, end = float(row.t_start), float(row.t_end)

        mag = window_stats(*data["mag"], start, end)
        kin = window_stats(*data["kin"], start, end)
        mag_tor = window_stats(*data["mag_tor"], start, end)
        mag_pol = window_stats(*data["mag_pol"], start, end)
        kin_tor = window_stats(*data["kin_tor"], start, end)
        kin_pol = window_stats(*data["kin_pol"], start, end)
        tem = window_stats(*data["tem"], start, end)
        fdip = window_stats(*data["fdip"], start, end)
        tilt = window_stats(*data["tilt"], start, end)
        g10_sq = window_stats(*data["g10_squared"], start, end)
        deq_sq = window_stats(*data["deq_squared"], start, end)

        # NEW: time-window averages of dissipation.
        vis = window_stats(*data["visDis"], start, end)
        ohm = window_stats(*data["ohmDis"], start, end)
        buoyancy_power = vis["mean"] + ohm["mean"]
        # Ratio of mean powers (NOT mean of instantaneous ratios).
        fohm = ohm["mean"] / buoyancy_power if buoyancy_power > 0 else np.nan

        q, Ra = float(row.q), float(row.Ra)
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
                raise ValueError(f"Both fit_start and fit_end are required: {key}")
            if not float(fit_start) < float(fit_end):
                raise ValueError(f"Invalid g10 fitting interval: {key}")
            sigma_stats = fit_g10_rate(
                *data["g10"], float(fit_start), float(fit_end)
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
            # Energy fractions
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
            # NEW: dissipation and inferred buoyancy power
            "visDis": vis["mean"],
            "visDis_std": vis["std"],
            "ohmDis": ohm["mean"],
            "ohmDis_std": ohm["std"],
            "fohm": fohm,
            "buoyancy_power": buoyancy_power,
            "n_samples_visDis": vis["n_samples"],
            "n_samples_ohmDis": ohm["n_samples"],
            # Other metadata and sigma fitting
            "n_samples": mag["n_samples"],
            **sigma_stats,
        })

        print(
            f"{row.branch:16s} "
            f"q={q:.3f} "
            f"state={row.state:3s} "
            f"Lambda={Lambda:.4e} "
            f"Rm={Rm:.2f} "
            f"Pe={Pe:.2f} "
            f"T_perb={T_perb:.5e} "
            f"visDis={vis['mean']:.4e} "
            f"ohmDis={ohm['mean']:.4e} "
            f"fohm={fohm:.4f}"
        )

    result = pd.DataFrame(output_rows)
    result.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
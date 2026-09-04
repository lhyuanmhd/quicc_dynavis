#!/usr/bin/env python3
"""
visu_udotgradu.py

Standalone plotter for |u · ∇u| magnitude:
- Equatorial slice + meridional slice (phi = atphi*pi)
- Uses NPZ produced by extractor (expects key: u_dot_grad_u_magnitude)
- Keeps CLI compatible with visu_snapshots.py (run/tag selection, safe-latest, --visu-dir, --force, etc.)
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quicc_dynavis import fields_snapshot

# -----------------------------
# Utilities: runs discovery
# -----------------------------
_RUN_RE = re.compile(r"^run(\d+)$")
_VISU_RE = re.compile(r"^visu_(\d+)$")

def _discover_run_roots(case_dir: Path):
    roots = []
    if (case_dir / "runs").is_dir():
        roots.append(case_dir / "runs")
    roots.append(case_dir)
    return roots

def _discover_runs(case_dir: Path):
    runs = []
    for root in _discover_run_roots(case_dir):
        runs += [p for p in root.glob("run*") if p.is_dir() and _RUN_RE.match(p.name)]

    def rkey(p: Path):
        m = _RUN_RE.match(p.name)
        return int(m.group(1)) if m else -1

    return sorted(set(runs), key=rkey)

def _pick_run(case_dir: Path, run: str):
    if run != "auto":
        cand = case_dir / "runs" / run
        if cand.is_dir():
            return cand
        cand = case_dir / run
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"Run directory not found: {run} under {case_dir}")

    runs = _discover_runs(case_dir)
    if not runs:
        raise FileNotFoundError(f"No run* directories found under {case_dir}")
    return max(runs, key=lambda p: int(_RUN_RE.match(p.name).group(1)))


# -----------------------------
# Utilities: snapshot tags
# -----------------------------
def _list_available_visu_tags(run_dir: Path) -> List[str]:
    tags = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = _VISU_RE.match(p.name)
        if m:
            tags.append("{:04d}".format(int(m.group(1))))
    return sorted(set(tags), key=lambda s: int(s))

def _parse_tags_csv(s: str) -> List[str]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return ["{:04d}".format(int(x)) for x in items]

def _choose_tags(run_dir: Path,
                 tag: Optional[str],
                 tags: Optional[str],
                 latest: bool,
                 safe_latest: bool,
                 nlatest: int) -> List[str]:
    avail = _list_available_visu_tags(run_dir)
    if not avail:
        raise FileNotFoundError(f"No visu_#### directories found in {run_dir}")

    if tag is not None:
        chosen = ["{:04d}".format(int(tag))]
    elif tags is not None:
        chosen = _parse_tags_csv(tags)
    elif nlatest and nlatest > 0:
        if latest:
            chosen = avail[-nlatest:]
        else:
            if len(avail) >= 2:
                chosen = avail[:-1][-nlatest:]
            else:
                chosen = avail[-nlatest:]
    else:
        if latest:
            chosen = [avail[-1]]
        else:
            if len(avail) >= 2:
                chosen = [avail[-2]]
            else:
                chosen = [avail[-1]]

    aset = set(avail)
    missing = [t for t in chosen if t not in aset]
    if missing:
        raise FileNotFoundError(
            f"Requested tags not found in {run_dir}: {missing}\n"
            f"Available (head/tail): {avail[:5]} ... {avail[-5:]}"
        )
    return chosen


# -----------------------------
# Find vis_fields NPZ for a tag
# -----------------------------
def _find_vis_fields_npz(run_dir: Path, tag: str) -> Path:
    visu = run_dir / f"visu_{tag}"
    if not visu.is_dir():
        raise FileNotFoundError(f"visu directory not found: {visu}")

    preferred = [
        visu / "vis_fields_0000.npz",
        visu / f"vis_fields{tag}.npz",
        visu / "vis_fields.npz",
    ]
    for fp in preferred:
        if fp.exists():
            return fp

    cands = sorted(visu.glob("vis_fields*.npz"))
    if cands:
        return cands[-1]

    raise FileNotFoundError(f"No vis_fields*.npz found in {visu}")


# -----------------------------
# Path parsing (Ek/q/Ra)
# -----------------------------
def _input_params_from_path(case_dir: Path):
    s = str(case_dir)

    def find_after(prefix):
        m = re.search(rf"{prefix}([^/]+)", s)
        return m.group(1) if m else None

    Ek = find_after("E")
    q  = find_after("q_")
    Ra = find_after("Ra")
    return Ek, q, Ra


# -----------------------------
# Plot: |u dot gradu| magnitude
# -----------------------------
def plot_udotgradu_panel(case_dir: Path, data, save_path: Path,
                         atphi=2/3):
    """
    1x2 panel:
      (|u·∇u| equatorial, |u·∇u| meridional)
    Uses fields_snapshot plotting functions.
    """

    key = "u_dot_grad_u_magnitude"
    if key not in data:
        raise KeyError(
            f"NPZ missing '{key}'. "
            "Re-run extractor with include_udotgradu_mag=True."
        )

    try:
        from quicc_dynavis import timeseries as ts
        Ek, q, Ra = ts.input_params_from_path(str(case_dir))
    except Exception:
        Ek, q, Ra = _input_params_from_path(case_dir)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Equatorial 
    fields_snapshot.plot_equatorial(
        str(case_dir),
        data,
        key,
        ax=ax0
    )

    # Meridional 
    fields_snapshot.plot_meridional(
        str(case_dir),
        data,
        key,
        atphi=atphi,
        ax=ax1,
        
    )

    time = float(data["time"]) if "time" in data else float("nan")

    fig.suptitle(
        "Ek={}, q={}, Ra={}, time={:.2e}".format(Ek, q, Ra, time),
        y=0.98,
        fontsize=14
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("[OK] Saved figure: {}".format(save_path))



# -----------------------------
# Explicit visu-dir mode
# -----------------------------
def _infer_case_dir_from_visu_dir(visu_dir: Path) -> Path:
    run_dir = visu_dir.parent
    if not _RUN_RE.match(run_dir.name):
        raise ValueError(f"--visu-dir must be inside run*/visu_####. Got: {visu_dir}")
    parent = run_dir.parent
    if parent.name == "runs":
        return parent.parent
    return parent

def _plot_one_visu_dir(visu_dir: Path,
                       out: Optional[Path],
                       force: bool,
                       atphi: float,
                       dry_run: bool):
    visu_dir = visu_dir.resolve()
    run_dir = visu_dir.parent
    m_run = _RUN_RE.match(run_dir.name)
    m_visu = _VISU_RE.match(visu_dir.name)
    if not m_run or not m_visu:
        raise ValueError(f"--visu-dir must match run*/visu_####. Got: {visu_dir}")

    tag = "{:04d}".format(int(m_visu.group(1)))
    case_dir = _infer_case_dir_from_visu_dir(visu_dir).resolve()
    npz_path = _find_vis_fields_npz(run_dir, tag)

    if out is None:
        fig_dir = case_dir / "figures"
        try:
            from quicc_dynavis import timeseries as ts
            Ek, q, Ra = ts.input_params_from_path(str(case_dir))
        except Exception:
            Ek, q, Ra = _input_params_from_path(case_dir)
        out = fig_dir / f"Ek_{Ek}_q{q}_Ra{Ra}_{run_dir.name}_{tag}_udotgradu.pdf"
    else:
        out = out.resolve()

    print("========================================")
    print("[INFO] case_dir : {}".format(case_dir))
    print("[INFO] run_dir  : {}".format(run_dir))
    print("[INFO] visu_dir : {}".format(visu_dir))
    print("[INFO] tag      : {}".format(tag))
    print("[INFO] npz      : {}".format(npz_path))
    print("[INFO] out      : {}".format(out))
    print("========================================")

    if out.exists() and not force:
        print("[SKIP] Output exists (use --force to overwrite): {}".format(out))
        return
    if dry_run:
        return

    data = np.load(npz_path)
    plot_udotgradu_panel(case_dir=case_dir, data=data, save_path=out, atphi=atphi)


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="LUMI diagnostics: |u·∇u| magnitude (equatorial + meridional).")

    p.add_argument("case_dir", nargs="?", default=".",
                   help="Case directory (default: current directory).")

    p.add_argument("--visu-dir", default=None,
                   help="Direct path to run*/visu_####. If provided, run/tag selection is ignored.")
    p.add_argument("--out", default=None, help="Explicit output PNG path (optional).")
    p.add_argument("--force", action="store_true", help="Overwrite existing output figure.")

    p.add_argument("--run", default="auto", help="Run folder name (run3) or 'auto' (default).")

    p.add_argument("--tag", default=None, help="Single snapshot tag like 0040 (legacy).")
    p.add_argument("--tags", default=None, help="Comma-separated tags like 0040,0027,0011.")
    p.add_argument("--latest", action="store_true", help="Use latest available tag (may be writing).")
    p.add_argument("--safe-latest", action="store_true",
                   help="Use safe latest tag (default): usually second-latest to avoid partial writes.")
    p.add_argument("--nlatest", type=int, default=0, help="Plot latest N tags (default safe behavior).")

    p.add_argument("--atphi", type=float, default=2/3,
                   help="Meridional cut position in units of pi (0..2).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print what would be plotted, then exit.")

    args = p.parse_args()

    if args.visu_dir is not None:
        visu_dir = Path(args.visu_dir)
        out = Path(args.out) if args.out is not None else None
        _plot_one_visu_dir(
            visu_dir=visu_dir,
            out=out,
            force=bool(args.force),
            atphi=float(args.atphi),
            dry_run=bool(args.dry_run),
        )
        return

    case_dir = Path(args.case_dir).resolve()
    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _pick_run(case_dir, args.run)

    latest = bool(args.latest)
    safe_latest = bool(args.safe_latest) or (
        (not latest) and args.tag is None and args.tags is None and args.nlatest == 0
    )
    if latest:
        safe_latest = False

    chosen_tags = _choose_tags(
        run_dir=run_dir,
        tag=args.tag,
        tags=args.tags,
        latest=latest,
        safe_latest=safe_latest,
        nlatest=args.nlatest,
    )

    try:
        from quicc_dynavis import timeseries as ts
        Ek, q, Ra = ts.input_params_from_path(str(case_dir))
    except Exception:
        Ek, q, Ra = _input_params_from_path(case_dir)

    print("========================================")
    print("[INFO] case_dir : {}".format(case_dir))
    print("[INFO] run_dir  : {}".format(run_dir))
    print("[INFO] tags     : {}".format(chosen_tags))
    print("========================================")

    if args.dry_run:
        return

    for t in chosen_tags:
        npz_path = _find_vis_fields_npz(run_dir, t)
        out = fig_dir / "Ek_{}_q{}_Ra{}_{}_{}_udotgradu.pdf".format(Ek, q, Ra, run_dir.name, t)

        if out.exists() and not args.force:
            print("[SKIP] Output exists (use --force to overwrite): {}".format(out))
            continue

        data = np.load(npz_path)
        plot_udotgradu_panel(case_dir=case_dir, data=data, save_path=out, atphi=args.atphi)


if __name__ == "__main__":
    main()


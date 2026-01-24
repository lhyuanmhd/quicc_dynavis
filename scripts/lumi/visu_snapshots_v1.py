#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

# Headless for LUMI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/scratch/project_465001528/lhyuan/codes/quicc_dynavis/src')
from quicc_dynavis import io, spectra, timeseries, fields_snapshot 

def _default_case_dir() -> Path:
    return Path(".").resolve()


def _find_vis_fields_npz(case_dir: Path, run: str = None, tag: str = None) -> Path:
    """
    Find a vis_fields.npz.
    Expected layout examples:
      case_dir/runs/run3/visu/vis_fields_0000.npz
      case_dir/runs/run3/visu/vis_fields.npz
      (or old layout) case_dir/run3/visu/...
    """
    # candidate run roots: new + old layout
    run_roots = []
    if (case_dir / "runs").is_dir():
        run_roots.append(case_dir / "runs")
    run_roots.append(case_dir)  # fallback old layout

    candidates = []

    for root in run_roots:
        if run is not None:
            run_dirs = [root / run]
        else:
            run_dirs = sorted([p for p in root.glob("run*") if p.is_dir()])

        for rd in run_dirs:
            
            if tag is not None:
                visu = rd / f"visu_{tag}"
                if not visu.is_dir():
                    continue

            #if tag is not None:
                # accept either vis_fields_0000.npz or vis_fields0000.npz depending on your convention
                patt = [f"vis_fields_0000.npz", f"vis_fields{tag}.npz", "vis_fields.npz"]
                for fn in patt:
                    fp = visu / fn
                    if fp.exists():
                        candidates.append(fp)
            else:
                # prefer numbered files, else plain vis_fields.npz
                candidates += sorted(visu.glob("vis_fields*.npz"))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No vis_fields*.npz found under {case_dir} (run={run}, tag={tag})")

    # If tag not provided, choose "latest" by natural sort on digits (or mtime fallback)
    def _key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        if m:
            return (0, int(m.group(1)))
        return (1, p.stat().st_mtime)

    candidates = sorted(set(candidates), key=_key)
    return candidates[-1]


def _input_params_from_path(case_dir: Path):
    """
    Keep your favorite method: parse Ek/q/Ra from path like .../E1e-5/q_1/Ra1e3/...
    If you already have timeseries.input_params_from_path, use that instead.
    """
    s = str(case_dir)

    def find_after(prefix):
        m = re.search(rf"{prefix}([^/]+)", s)
        return m.group(1) if m else None

    Ek = find_after("E")
    q = find_after("q_")
    Ra = find_after("Ra")

    # best-effort normalization
    return Ek, q, Ra


def plot_snapshot_panel(case_dir: Path, data: np.lib.npyio.NpzFile, save_path: Path,
                        atphi=2/3, show_grid=True):
    """
    Reproduce your preferred 2x3 panel layout.
    """
    # Import here to keep script robust if user runs without these modules installed
    #from quicc_dynavis import fields_snapshot
    # If your input_params_from_path already exists, use that:
    try:
        from quicc_dynavis import timeseries as ts
        Ek, q, Ra = ts.input_params_from_path(str(case_dir))
    except Exception:
        Ek, q, Ra = _input_params_from_path(case_dir)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.45], wspace=0.1, hspace=0.25)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2], projection="mollweide")

    # make mollweide less cramped
    for ax in (ax02, ax12):
        ax.set_aspect("auto")

    # --- Row 1 ---
    fields_snapshot.plot_equatorial(str(case_dir), data, "u_r", ax=ax00)
    fields_snapshot.plot_meridional(str(case_dir), data, "u_phi", atphi=atphi, ax=ax01)
    fields_snapshot.plot_equatorial(str(case_dir), data, "curl_u_axial", ax=ax02)

    # --- Row 2 ---
    fields_snapshot.plot_equatorial(str(case_dir), data, "T", ax=ax10, include_background=True)
    fields_snapshot.plot_meridional(str(case_dir), data, "B_r", atphi=atphi, ax=ax11)
    fields_snapshot.plot_cmb(str(case_dir), data, "B_r", ax=ax12)

    # Title (optional): keep compact
    time = data["time"]
    time = f'{time:.2e}'
    fig.suptitle(f"Ek={Ek}, q={q}, Ra={Ra}, time={time}",  y=0.98, fontsize=16)

    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def main():
    p = argparse.ArgumentParser(description="LUMI snapshots: 2x3 panel (equatorial/meridional/CMB).")
    p.add_argument("case_dir", nargs="?", default=".", help="Case directory (default: current directory).")
    p.add_argument("--run", default=None, help="Run folder name, e.g. run3. Default: auto.")
    p.add_argument("--tag", default=None, help="Snapshot tag, e.g. 0000. Default: auto-latest.")
    p.add_argument("--atphi", type=float, default=2/3, help="Meridional cut position in units of pi (0..2).")
    p.add_argument("--no-grid", action="store_true", help="Disable lon/lat grid on CMB plot.")
    args = p.parse_args()

    case_dir = Path(args.case_dir).resolve()
    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    npz_path = _find_vis_fields_npz(case_dir, run=args.run, tag=args.tag)
    data = np.load(npz_path)

    # label from npz and run
    run_tag = args.run if args.run else "auto"
    snap_tag = args.tag if args.tag else "latest"

    # If you have canonical Ek/q/Ra parsing, use it
    try:
        from quicc_dynavis import timeseries as ts
        Ek, q, Ra = ts.input_params_from_path(str(case_dir))
    except Exception:
        Ek, q, Ra = _input_params_from_path(case_dir)

    out = fig_dir / f"Ek_{Ek}_q{q}_Ra{Ra}_{run_tag}_{snap_tag}_snapshots.png"

    plot_snapshot_panel(
        case_dir=case_dir,
        data=data,
        save_path=out,
        atphi=args.atphi,
        show_grid=(not args.no_grid),
    )


if __name__ == "__main__":
    main()

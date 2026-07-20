#!/usr/bin/env python3
"""
visu_all_forces.py

Standalone plotter for all forces in the momentum equation:
- Inertia: |(u·∇)u|
- Coriolis: |ẑ × u|
- Viscous: |E ∇²u|
- Lorentz: |(∇×B)×B|
- Buoyancy: |q Ra T r|

Plots equatorial + meridional slices for each force.
Can plot individually or in a combined 2x3 panel.
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

sys.path.append('/scratch/project_465001528/lhyuan/codes/quicc_dynavis/src')
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

    # Try different possible filenames - 优先使用新的forces文件
    candidates = [
        visu / "vis_fields_forces.npz",  # New forces file
        visu / "vis_fields_fixed.npz",   # Previous version
        visu / "vis_fields_0000.npz",    # Original
        visu / f"vis_fields{tag}.npz",
        visu / "vis_fields.npz",
    ]
    
    for fp in candidates:
        if fp.exists():
            return fp

    # Fallback to any vis_fields*.npz
    cands = sorted(visu.glob("vis_fields*.npz"))
    if cands:
        return cands[-1]

    raise FileNotFoundError(f"No vis_fields*.npz found in {visu}")


# -----------------------------
# Path parsing (Ek/q/Ra) - 备用方案
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
# 从数据中获取物理参数
# -----------------------------
def _get_phys_params(data):
    """Extract physical parameters from data dictionary."""
    params = {}
    
    # 尝试从数据中读取
    params['ekman'] = data.get("ekman", "?")
    params['rayleigh'] = data.get("rayleigh", "?")
    params['roberts'] = data.get("roberts", "?")
    params['time'] = float(data["time"]) if "time" in data else float("nan")
    
    # 字符串参数
    for key in ["velocity_str", "temperature_str", "magnetic_str"]:
        if key in data:
            params[key] = data[key]
    
    return params


# -----------------------------
# Simple color limit helper
# -----------------------------
def get_vmax_from_field(data, field_key, quantile=0.98):
    """Get vmax from a specific field for common scaling."""
    if field_key not in data:
        return None
    field = data[field_key]
    valid = field[np.isfinite(field)]
    if len(valid) == 0:
        return None
    return np.quantile(np.abs(valid), quantile)


# -----------------------------
# Plotting functions
# -----------------------------
FORCE_CONFIG = {
    'coriolis_magnitude': {
        'title': r'Coriolis $|\hat{\mathbf{z}} \times \mathbf{u}|$',
        'short': 'coriolis',
        'cmap':  "cividis"
    },
    'lorentz_magnitude': {
        'title': r'Lorentz $|(\nabla\times\mathbf{B})\times\mathbf{B}|$',
        'short': 'lorentz',
        'cmap': "cividis",
    },
    'buoyancy_magnitude': {
        'title': r'Buoyancy $|q Ra T \mathbf{r}|$',
        'short': 'buoyancy',
        'cmap': "cividis",
    },
    'viscous_magnitude': {
        'title': r'Viscous $|E\nabla^2\mathbf{u}|$',
        'short': 'viscous',
        'cmap': "cividis",
    },
    'inertia_magnitude': {
        'title': r'Inertia $E_\eta|\mathbf{u}\cdot\nabla\mathbf{u}|$',
        'short': 'inertia',
        'cmap':  "cividis"
    },
}

def plot_force_panel(case_dir: Path, data, save_path: Path,
                     force_key: str, atphi: float = 2/3,
                     vmax: Optional[float] = None):
    """
    Plot 1x2 panel for a single force:
      (equatorial slice, meridional slice)
    """
    if force_key not in data:
        available = [k for k in FORCE_CONFIG.keys() if k in data]
        raise KeyError(
            f"NPZ missing '{force_key}'. Available forces: {available}\n"
            "Re-run extractor with appropriate flags."
        )

    # 优先从数据中获取参数，失败则从路径解析
    params = _get_phys_params(data)
    if params['ekman'] == '?' or params['rayleigh'] == '?' or params['roberts'] == '?':
        try:
            from quicc_dynavis import timeseries as ts
            Ek, q, Ra = ts.input_params_from_path(str(case_dir))
            params['ekman'] = Ek
            params['roberts'] = q
            params['rayleigh'] = Ra
        except Exception:
            Ek, q, Ra = _input_params_from_path(case_dir)
            params['ekman'] = Ek if Ek else '?'
            params['roberts'] = q if q else '?'
            params['rayleigh'] = Ra if Ra else '?'

    config = FORCE_CONFIG[force_key]
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Equatorial slice
    fields_snapshot.plot_equatorial(
        str(case_dir),
        data,
        force_key,
        ax=ax0,
        vmin=0,
        vmax=vmax
    )
    title = f"Equatorial slice - {config['title']}"
    if vmax is not None:
        title += f"\nvmax={vmax:.2e}"
    ax0.set_title(title)

    # Meridional slice
    fields_snapshot.plot_meridional(
        str(case_dir),
        data,
        force_key,
        atphi=atphi,
        ax=ax1,
        vmin=0,
        vmax=vmax
    )
    title = f"Meridional slice - {config['title']}"
    if vmax is not None:
        title += f"\nvmax={vmax:.2e}"
    ax1.set_title(title)

    # Add physical parameters to title
    scale_info = f" [vmax={vmax:.2e}]" if vmax is not None else ""
    fig.suptitle(
        f"Ek={params['ekman']}, q={params['roberts']}, Ra={params['rayleigh']}, time={params['time']:.2e}{scale_info}",
        y=0.98,
        fontsize=14
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] Saved {config['title']} figure: {save_path}")


def plot_all_forces_panel(case_dir: Path, data, save_path: Path,
                          atphi: float = 2/3,
                          common_vmax: Optional[float] = None):
    """
    Plot 2x3 panel showing all five forces:
    Top row: Inertia, Coriolis, Viscous
    Bottom row: Lorentz, Buoyancy, (empty or legend)
    """
    # Check which forces are available
    available_forces = [k for k in FORCE_CONFIG.keys() if k in data]
    
    if len(available_forces) == 0:
        raise KeyError("No force data found in NPZ file")

    # obtain parameters from data
    params = _get_phys_params(data)
    if params['ekman'] == '?' or params['rayleigh'] == '?' or params['roberts'] == '?':
        try:
            from quicc_dynavis import timeseries as ts
            Ek, q, Ra = ts.input_params_from_path(str(case_dir))
            params['ekman'] = Ek
            params['roberts'] = q
            params['rayleigh'] = Ra
        except Exception:
            Ek, q, Ra = _input_params_from_path(case_dir)
            params['ekman'] = Ek if Ek else '?'
            params['roberts'] = q if q else '?'
            params['rayleigh'] = Ra if Ra else '?'

    # Create 2x3 grid
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, wspace=0.2, hspace=0.25)

    # Map forces to grid positions
    force_positions = [
        (0, 0, 'coriolis_magnitude'), 
        (0, 1, 'buoyancy_magnitude'),
        (0, 2, 'lorentz_magnitude'),
        (1, 0, 'viscous_magnitude'),
        (1, 1, 'inertia_magnitude'),
    ]

    for i, j, force_key in force_positions:
        if force_key not in data:
            continue
            
        ax = fig.add_subplot(gs[i, j])
        config = FORCE_CONFIG[force_key]
        
        # Plot equatorial slice
        fields_snapshot.plot_equatorial(
            str(case_dir),
            data,
            force_key,
            ax=ax,
            #vmin=0,
            vmax=common_vmax
        )
        
        title = config['title']
        if common_vmax is not None:
            title += f"\nvmax={common_vmax:.2e}"
        ax.set_title(title, fontsize=12)

    # Hide empty subplot and add info panel
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    # Build info text from parameters
    info_text = (
        f"Parameters:\n"
        f"Ek = {params['ekman']}\n"
        f"q = {params['roberts']}\n"
        f"Ra = {params['rayleigh']}\n"
        f"time = {params['time']:.2e}\n\n"
    )
    
    # Add boundary conditions if available
    for key in ["velocity_str", "temperature_str", "magnetic_str"]:
        if key in params:
            name = key.replace("_str", "")
            info_text += f"{name}: {params[key]}\n"
    
    info_text += f"\nForces:\n"
    info_text += "\n".join([f"• {FORCE_CONFIG[k]['short']}" for k in available_forces])
    
    if common_vmax is not None:
        info_text += f"\n\n[Common Scale]\nvmax = {common_vmax:.2e}"
    
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    scale_info = f" [vmax={common_vmax:.2e}]" if common_vmax is not None else ""
    fig.suptitle(
        f"Forces - Ek={params['ekman']}, q={params['roberts']}, Ra={params['rayleigh']}, time={params['time']:.2e}{scale_info}",
        y=0.98,
        fontsize=16
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] Saved all forces panel: {save_path}")


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
                       mode: str,
                       force_key: Optional[str],
                       common_vmax: Optional[float],
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
        # 尝试从数据获取参数来确定文件名
        try:
            data_tmp = np.load(npz_path)
            params = _get_phys_params(data_tmp)
            ekman = params['ekman']
            roberts = params['roberts'] 
            rayleigh = params['rayleigh']
        except:
            # 失败则从路径解析
            try:
                from quicc_dynavis import timeseries as ts
                ekman, roberts, rayleigh = ts.input_params_from_path(str(case_dir))
            except Exception:
                ekman, roberts, rayleigh = _input_params_from_path(case_dir)
        
        vmax_suffix = f"_vmax{common_vmax:.2e}" if common_vmax is not None else ""
        
        if mode == "all":
            out = fig_dir / f"Ek_{ekman}_q{roberts}_Ra{rayleigh}_{run_dir.name}_{tag}_all_forces{vmax_suffix}.png"
        elif mode == "single" and force_key:
            short = FORCE_CONFIG[force_key]['short']
            out = fig_dir / f"Ek_{ekman}_q{roberts}_Ra{rayleigh}_{run_dir.name}_{tag}_{short}{vmax_suffix}.png"
    else:
        out = out.resolve()

    print("========================================")
    print(f"[INFO] case_dir : {case_dir}")
    print(f"[INFO] run_dir  : {run_dir}")
    print(f"[INFO] visu_dir : {visu_dir}")
    print(f"[INFO] tag      : {tag}")
    print(f"[INFO] mode     : {mode}")
    if force_key:
        print(f"[INFO] force    : {force_key}")
    print(f"[INFO] common_vmax: {common_vmax}")
    print(f"[INFO] npz      : {npz_path}")
    print(f"[INFO] out      : {out}")
    print("========================================")

    if out.exists() and not force:
        print(f"[SKIP] Output exists (use --force to overwrite): {out}")
        return
    if dry_run:
        return

    data = np.load(npz_path)
    
    if mode == "all":
        plot_all_forces_panel(case_dir=case_dir, data=data, save_path=out, 
                              atphi=atphi, common_vmax=common_vmax)
    elif mode == "single" and force_key:
        plot_force_panel(case_dir=case_dir, data=data, save_path=out,
                        force_key=force_key, atphi=atphi, vmax=common_vmax)


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(
        description="LUMI diagnostics: Plot all forces in the momentum equation"
    )

    p.add_argument("case_dir", nargs="?", default=".",
                   help="Case directory (default: current directory).")

    p.add_argument("--visu-dir", default=None,
                   help="Direct path to run*/visu_####. If provided, run/tag selection is ignored.")
    p.add_argument("--out", default=None, help="Explicit output PNG path (optional).")
    p.add_argument("--force", action="store_true", help="Overwrite existing output figure.")
    
    # Simple common vmax option
    p.add_argument("--common-vmax", type=float, default=None,
                   help="Use this vmax value for all plots (for comparison).")
    p.add_argument("--vmax-from", default=None,
                   choices=['inertia_magnitude', 'coriolis_magnitude', 'viscous_magnitude',
                           'lorentz_magnitude', 'buoyancy_magnitude'],
                   help="Set vmax based on this force's 98th percentile.")
    p.add_argument("--quantile", type=float, default=0.98,
                   help="Quantile for vmax-from calculation (default: 0.98)")
    
    # Mode selection
    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--all", action="store_true",
                          help="Plot all forces in a 2x3 panel")
    mode_group.add_argument("--inertia", action="store_true",
                          help="Plot inertia force |(u·∇)u| only")
    mode_group.add_argument("--coriolis", action="store_true",
                          help="Plot Coriolis force |ẑ×u| only")
    mode_group.add_argument("--viscous", action="store_true",
                          help="Plot viscous force |E∇²u| only")
    mode_group.add_argument("--lorentz", action="store_true",
                          help="Plot Lorentz force |(∇×B)×B| only")
    mode_group.add_argument("--buoyancy", action="store_true",
                          help="Plot buoyancy force |qRaTr| only")

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

    # Determine mode and force key
    mode = "all" if args.all else "single"
    force_key = None
    if args.inertia:
        force_key = "inertia_magnitude"
    elif args.coriolis:
        force_key = "coriolis_magnitude"
    elif args.viscous:
        force_key = "viscous_magnitude"
    elif args.lorentz:
        force_key = "lorentz_magnitude"
    elif args.buoyancy:
        force_key = "buoyancy_magnitude"

    if args.visu_dir is not None:
        visu_dir = Path(args.visu_dir)
        out = Path(args.out) if args.out is not None else None
        
        # If vmax-from is specified, we need to load data first to get vmax
        common_vmax = args.common_vmax
        if args.vmax_from is not None and common_vmax is None:
            # Need to peek at data to get vmax
            run_dir = visu_dir.parent
            tag = "{:04d}".format(int(_VISU_RE.match(visu_dir.name).group(1)))
            npz_path = _find_vis_fields_npz(run_dir, tag)
            data_tmp = np.load(npz_path)
            common_vmax = get_vmax_from_field(data_tmp, args.vmax_from, args.quantile)
            print(f"[INFO] Setting vmax = {common_vmax:.2e} from {args.vmax_from}")
        
        _plot_one_visu_dir(
            visu_dir=visu_dir,
            out=out,
            force=bool(args.force),
            atphi=float(args.atphi),
            mode=mode,
            force_key=force_key,
            common_vmax=common_vmax,
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

    print("========================================")
    print(f"[INFO] case_dir : {case_dir}")
    print(f"[INFO] run_dir  : {run_dir}")
    print(f"[INFO] tags     : {chosen_tags}")
    print(f"[INFO] mode     : {'all' if args.all else force_key}")
    print("========================================")

    if args.dry_run:
        return

    for t in chosen_tags:
        npz_path = _find_vis_fields_npz(run_dir, t)
        
        data_tmp = np.load(npz_path)
        params = _get_phys_params(data_tmp)
        
        if params['ekman'] == '?' or params['rayleigh'] == '?' or params['roberts'] == '?':
            try:
                from quicc_dynavis import timeseries as ts
                ekman, roberts, rayleigh = ts.input_params_from_path(str(case_dir))
            except Exception:
                ekman, roberts, rayleigh = _input_params_from_path(case_dir)
        else:
            ekman = params['ekman']
            roberts = params['roberts']
            rayleigh = params['rayleigh']
        
        # Determine vmax to use
        common_vmax = args.common_vmax
        if args.vmax_from is not None and common_vmax is None:
            common_vmax = get_vmax_from_field(data_tmp, args.vmax_from, args.quantile)
            print(f"[INFO] Tag {t}: Setting vmax = {common_vmax:.2e} from {args.vmax_from}")
        
        vmax_suffix = f"_vmax{common_vmax:.2e}" if common_vmax is not None else ""
        
        if args.all:
            out = fig_dir / f"Ek_{ekman}_q{roberts}_Ra{rayleigh}_{run_dir.name}_{t}_all_forces{vmax_suffix}.png"
        else:
            short = FORCE_CONFIG[force_key]['short']
            out = fig_dir / f"Ek_{ekman}_q{roberts}_Ra{rayleigh}_{run_dir.name}_{t}_{short}{vmax_suffix}.png"

        if out.exists() and not args.force:
            print(f"[SKIP] Output exists (use --force to overwrite): {out}")
            continue

        data = np.load(npz_path)
        
        if args.all:
            plot_all_forces_panel(case_dir=case_dir, data=data, save_path=out, 
                                  atphi=args.atphi, common_vmax=common_vmax)
        else:
            plot_force_panel(case_dir=case_dir, data=data, save_path=out,
                           force_key=force_key, atphi=args.atphi, vmax=common_vmax)


if __name__ == "__main__":
    main()
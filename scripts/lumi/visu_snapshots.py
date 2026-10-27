#!/usr/bin/env python3
"""
visu_snapshots.py (Python 3.8/3.9 compatible)

Key improvements:
- Adds --force to overwrite existing figures (default: skip if figure exists).
- Adds --visu-dir to target a specific directory runX/visu_#### directly.
- Adds --out to explicitly set the output PNG path (optional).
- Keeps existing CLI behavior intact (run discovery, tag selection, safe-latest).
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
# ----------------------------
_RUN_RE = re.compile(r"^run(\d+)$")
_VISU_RE = re.compile(r"^visu_(\d+)$")

def _discover_run_roots(case_dir: Path):
    """Return candidate run-roots: new layout (case/runs) then old layout (case)."""
    roots = []
    if (case_dir / "runs").is_dir():
        roots.append(case_dir / "runs")
    roots.append(case_dir)  # fallback old layout
    return roots

def _discover_runs(case_dir: Path):
    """Return list of run directories (Path). Supports new and old layouts."""
    runs = []
    for root in _discover_run_roots(case_dir):
        runs += [p for p in root.glob("run*") if p.is_dir() and _RUN_RE.match(p.name)]

    def rkey(p: Path):
        m = _RUN_RE.match(p.name)
        return int(m.group(1)) if m else -1

    runs = sorted(set(runs), key=rkey)
    return runs

def _pick_run(case_dir: Path, run: str):
    """
    run:
      - "auto": pick largest run index
      - "run3": pick that exact run (supports new/old layout)
    """
    if run != "auto":
        cand = case_dir / "runs" / run
        if cand.is_dir():
            return cand
        cand = case_dir / run
        if cand.is_dir():
            return cand
        raise FileNotFoundError("Run directory not found: {} under {}".format(run, case_dir))

    runs = _discover_runs(case_dir)
    if not runs:
        raise FileNotFoundError("No run* directories found under {}".format(case_dir))

    def rkey(p: Path):
        m = _RUN_RE.match(p.name)
        return int(m.group(1)) if m else -1

    return max(runs, key=rkey)


# -----------------------------
# Utilities: snapshot tags
# -----------------------------
def _list_available_visu_tags(run_dir: Path) -> List[str]:
    """
    Available tags from directories like visu_0040.
    Returns list of normalized tags ["0000","0011",...], sorted increasing.
    """
    tags = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = _VISU_RE.match(p.name)
        if m:
            tags.append("{:04d}".format(int(m.group(1))))
    tags = sorted(set(tags), key=lambda s: int(s))
    return tags

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
        raise FileNotFoundError("No visu_#### directories found in {}".format(run_dir))

    if tag is not None:
        chosen = ["{:04d}".format(int(tag))]
    elif tags is not None:
        chosen = _parse_tags_csv(tags)
    elif nlatest and nlatest > 0:
        if latest:
            chosen = avail[-nlatest:]
        else:
            # Safe selection: drop the very last tag if possible
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
            "Requested tags not found in {}: {}\nAvailable (head/tail): {} ... {}".format(
                run_dir, missing, avail[:5], avail[-5:]
            )
        )
    return chosen


# -----------------------------
# Find vis_fields NPZ for a tag
# -----------------------------
def _find_vis_fields_npz(run_dir: Path, tag: str) -> Path:
    """
    Find vis_fields NPZ under:
      run_dir/visu_TAG/vis_fields_0000.npz
      run_dir/visu_TAG/vis_fieldsTAG.npz
      run_dir/visu_TAG/vis_fields.npz
      run_dir/visu_TAG/vis_fields*.npz
    """
    visu = run_dir / "visu_{}".format(tag)
    if not visu.is_dir():
        raise FileNotFoundError("visu directory not found: {}".format(visu))

    preferred = [
        visu / "vis_fields_0000.npz",
        visu / "vis_fields{}.npz".format(tag),
        visu / "vis_fields.npz",
    ]
    for fp in preferred:
        if fp.exists():
            return fp

    cands = sorted(visu.glob("vis_fields*.npz"))
    if cands:
        return cands[-1]

    raise FileNotFoundError("No vis_fields*.npz found in {}".format(visu))


# -----------------------------
# Path parsing (Ek/q/Ra)
# -----------------------------
def _input_params_from_path(case_dir: Path):
    s = str(case_dir)

    def find_after(prefix):
        m = re.search(r"{}([^/]+)".format(prefix), s)
        return m.group(1) if m else None

    Ek = find_after("E")
    q = find_after("q_")
    Ra = find_after("Ra")
    return Ek, q, Ra


# -----------------------------
# Plot panel
# -----------------------------
# def plot_snapshot_panel(case_dir: Path, data, save_path: Path,
#                         atphi=2/3, show_grid=False):
#     """
#     2x4 panel layout:
#     Row 1: u_r (eq), u_phi (eq), curl_u_axial (eq), zonal_flow (meridional)
#     Row 2: T (eq), B_r (mer), B_r (CMB), [reserved for additional field]
#     """
#     try:
#         from quicc_dynavis import timeseries as ts
#         Ek, q, Ra = ts.input_params_from_path(str(case_dir))
#     except Exception:
#         Ek, q, Ra = _input_params_from_path(case_dir)

#     # Create 2x4 grid with custom width ratios
#     fig = plt.figure(figsize=(18, 10))
#     gs = fig.add_gridspec(2, 3, 
#                           width_ratios=[1, 1, 1.45], 
#                           wspace=0.1, 
#                           hspace=0.25)

#     # Row 1 (equatorial plane views)
#     ax00 = fig.add_subplot(gs[0, 0])  
#     ax01 = fig.add_subplot(gs[0, 1])  
#     ax02 = fig.add_subplot(gs[0, 2])  

#     # Row 2
#     ax10 = fig.add_subplot(gs[1, 0])  # T equatorial
#     ax11 = fig.add_subplot(gs[1, 1])
#     ax12 = fig.add_subplot(gs[1, 2],  projection="mollweide")  # B_r CMB

#     # Adjust aspect ratios
#     for ax in (ax02, ax12,  #ax03, #ax13
#                ):
#         ax.set_aspect("auto")

#     # Row 1 plots
#     fields_snapshot.plot_equatorial(str(case_dir), data, "u_r", ax=ax00)
#     # Plot zonal flow (u_phi zonal average) in meridional plane
#     fields_snapshot.plot_meridional(str(case_dir), data, "u_phi", phi_average=True,
#                                     ax=ax01, cmap='RdBu_r')
#     ax02.set_title(r'$\langle u_\phi \rangle_\phi$', fontsize=12)
#     fields_snapshot.plot_equatorial(str(case_dir), data, "w_z", ax=ax02)
    
  
#     # Row 2 plots
#     fields_snapshot.plot_equatorial(str(case_dir), data, "T", ax=ax10, include_background=True)
#     ax10.set_title(r'$T_0 + T$', pad=10, fontsize=16)
#     fields_snapshot.plot_meridional(str(case_dir), data, "B_r", atphi=atphi, ax=ax11)
#     fields_snapshot.plot_cmb(str(case_dir), data, "B_r", ax=ax12, show_grid=False)
    

#     # Add panel labels
#     panel_labels = ['(a)', '(b)', '(c)', '(d)', 
#                     '(e)', '(f)', #'(g)',
#                     #'(h)'
#                     ]
#     axes = [ax00, ax01, ax02,  
#             ax10, ax11, ax12, 
#             #ax13
#             ]
#     #for ax, label in zip(axes, panel_labels):
#     #    ax.text(0.02, 0.98, label, transform=ax.transAxes,
#     #            fontsize=12, fontweight='bold',
#     #            verticalalignment='top',
#     #            bbox=dict(boxstyle="round, pad=0.3", facecolor='white', alpha=0.8)
#     #            )

#     # Title
#     #time = data["time"]
#     #fig.suptitle(f"Ek={Ek:.2e}, q={q:.2e}, Ra={Ra:.2e}, time={float(time):.2e}",
#     #             y=0.98, fontsize=14)
    
#     time = data["time"]
#     #fig.suptitle("Ek={}, q={}, Ra={}, time={:.2e}".format(Ek, q, Ra, float(time)),
#     #              y=0.98, fontsize=16)
 
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
#     plt.close(fig)
#     print(f"[OK] Saved figure: {save_path}")

def plot_snapshot_panel(
    case_dir: Path,
    data,
    save_path: Path,
    atphi=2/3,
    show_grid=False,
):
    """
    Save two figures:

    1. Main 2x3 panel:
       Row 1: u_r (eq), zonal flow (mer), w_z (eq)
       Row 2: T_0 + T (eq), B_r (mer), B_r (CMB)

    2. Additional 1x2 panel:
       u_phi - <u_phi>_phi in equatorial and meridional planes.

    Assumes that the last axis of data["u_phi"] is phi.
    """
    case_dir = Path(case_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Main figure: 2x3 panels
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 1.45],
        wspace=0.1,
        hspace=0.25,
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])

    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2], projection="mollweide")

    fields_snapshot.plot_equatorial(
        str(case_dir), data, "u_r",
        ax=ax00,
    )

    fields_snapshot.plot_meridional(
        str(case_dir), data, "u_phi",
        phi_average=True,
        ax=ax01,
        cmap="RdBu_r",
    )
    ax01.set_title(
        r"$\langle u_\phi\rangle_\phi$",
        fontsize=14,
        pad=10,
    )

    fields_snapshot.plot_equatorial(
        str(case_dir), data, "w_z",
        ax=ax02,
    )

    fields_snapshot.plot_equatorial(
        str(case_dir), data, "T",
        ax=ax10,
        include_background=True,
    )
    ax10.set_title(
        r"$T_0 + T$",
        fontsize=16,
        pad=10,
    )

    fields_snapshot.plot_meridional(
        str(case_dir), data, "B_r",
        atphi=atphi,
        ax=ax11,
    )

    fields_snapshot.plot_cmb(
        str(case_dir), data, "B_r",
        ax=ax12,
        show_grid=False,
    )

    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"[OK] Saved figure: {save_path}")

def plot_zonal_flow_figures(
    case_dir: Path,
    data,
    output_dir: Path,
    prefix: str,
    atphi=2/3,
):
    """
    Generate zonal-flow diagnostic figures.

    Fig. 1:
        Zonal flow <u_phi>_phi:
        equatorial plane + meridional plane.

    Fig. 2:
        Nonaxisymmetric azimuthal velocity u_phi - <u_phi>_phi:
        equatorial plane + meridional plane.

    Assumes data["u_phi"] has shape (Nr, Ntheta, Nphi)
    on a uniformly spaced phi grid without a duplicated endpoint.
    """
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(fig, suffix):
        save_path = output_dir / f"{prefix}_{suffix}.pdf"
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)
        print(f"[OK] Saved figure: {save_path}")

    # ------------------------------------------------------------
    # Prepare axisymmetric and nonaxisymmetric velocity fields
    # ------------------------------------------------------------
    u_phi = np.asarray(data["u_phi"])
    zonal_flow = np.mean(u_phi, axis=-1, keepdims=True)

    # Expand the mean to the original 3D shape for the plotting routines.
    data_zonal = dict(data)
    data_zonal["u_phi"] = np.broadcast_to(zonal_flow, u_phi.shape)

    data_nonaxisym = dict(data)
    data_nonaxisym["u_phi"] = u_phi - zonal_flow

    # ============================================================
    # Fig. 1: zonal flow
    # ============================================================
    fig, (ax_eq, ax_mer) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={"wspace": 0.3},
    )

    fields_snapshot.plot_equatorial(
        str(case_dir),
        data_zonal,
        "u_phi",
        ax=ax_eq,
        cmap="RdBu_r",
    )
    ax_eq.set_title(
        r"$\langle u_\phi\rangle_\phi$ (equatorial)",
        fontsize=14,
        pad=10,
    )

    fields_snapshot.plot_meridional(
        str(case_dir),
        data_zonal,
        "u_phi",
        atphi=atphi,
        phi_average=False,
        ax=ax_mer,
        cmap="RdBu_r",
    )
    ax_mer.set_title(
        r"$\langle u_\phi\rangle_\phi$ (meridional)",
        fontsize=14,
        pad=10,
    )

    save_figure(fig, "zonal_flow")

    # ============================================================
    # Fig. 2: nonaxisymmetric azimuthal velocity
    # ============================================================
    fig, (ax_eq, ax_mer) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={"wspace": 0.3},
    )

    fields_snapshot.plot_equatorial(
        str(case_dir),
        data_nonaxisym,
        "u_phi",
        ax=ax_eq,
        cmap="RdBu_r",
    )
    ax_eq.set_title(
        r"$u_\phi-\langle u_\phi\rangle_\phi$ (equatorial)",
        fontsize=14,
        pad=10,
    )

    fields_snapshot.plot_meridional(
        str(case_dir),
        data_nonaxisym,
        "u_phi",
        atphi=atphi,
        phi_average=False,
        ax=ax_mer,
        cmap="RdBu_r",
    )
    ax_mer.set_title(
        r"$u_\phi-\langle u_\phi\rangle_\phi$ (meridional)",
        fontsize=14,
        pad=10,
    )

    save_figure(fig, "uphi_nonaxisym")
    
def plot_helicity_figures(
    case_dir: Path,
    data,
    output_dir: Path,
    prefix: str,
    atphi=2/3,
    helicity_lim=None,
    helicity_magnitude_max=None,
):    
    """
    Fig. 1: meridional slices of u_z, omega_z, and h_z.
    Fig. 2: phi-averaged u_z, omega_z, and h_z.
    Fig. 3: kinetic helicity and phi-averaged kinetic helicity.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(fig, suffix):
        save_path = output_dir / f"{prefix}_{suffix}.pdf"
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)
        print(f"[OK] Saved figure: {save_path}")

    axial_fields = ("u_z", "w_z", "axial_helicity")

    # ============================================================
    # Fig. 1: meridional slices
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, field_name, title in zip(
        axes,
        axial_fields,
        (r"$u_z$", r"$\omega_z$", r"$h_z=u_z\omega_z$"),
    ):
        fields_snapshot.plot_meridional(
            str(case_dir),
            data,
            field_name,
            atphi=atphi,
            phi_average=False,
            ax=ax,
        )
        ax.set_title(title)

    fig.subplots_adjust(wspace=0.25)
    save_figure(fig, "axial_helicity_slice")

    # ============================================================
    # Fig. 2: phi-averaged axial quantities
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, field_name, title in zip(
        axes,
        axial_fields,
        (
            r"$\langle u_z\rangle_\phi$",
            r"$\langle\omega_z\rangle_\phi$",
            r"$\langle u_z\omega_z\rangle_\phi$",
        ),
    ):
        fields_snapshot.plot_meridional(
            str(case_dir),
            data,
            field_name,
            phi_average=True,
            ax=ax,
        )
        ax.set_title(title)

    fig.subplots_adjust(wspace=0.25)
    save_figure(fig, "axial_helicity_phi_average")

    # # ============================================================
    # # Fig. 3: kinetic helicity, slice and phi average
    # # ============================================================
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # for ax, phi_average, title in zip(
    #     axes,
    #     (False, True),
    #     (
    #         r"$h=\boldsymbol{u}\cdot\boldsymbol{\omega}$",
    #         r"$\langle h\rangle_\phi$",
    #     ),
    # ):
    #     fields_snapshot.plot_meridional(
    #         str(case_dir),
    #         data,
    #         "helicity",
    #         atphi=atphi,
    #         phi_average=phi_average,
    #         ax=ax,
    #     )
    #     ax.set_title(title)
    
    # ============================================================
    # Fig. 3: helicity slice and phi-averaged helicity magnitude
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, field_name, phi_average, title in zip(
        axes,
        ("helicity", "helicity_magnitude"),
        (False, True),
        (
            r"$h=\mathbf{u}\cdot\boldsymbol{\omega}$",
            r"$\langle |h|\rangle_\phi$",
        ),
    ):
        is_magnitude = field_name == "helicity_magnitude"

        if is_magnitude:
            vmin = 0
            vmax = helicity_magnitude_max
        else:
            vmin = -helicity_lim if helicity_lim is not None else None
            vmax = helicity_lim

        fields_snapshot.plot_meridional(
            str(case_dir),
            data,
            field_name,
            atphi=atphi,
            phi_average=phi_average,
            ax=ax,
            cmap="inferno" if is_magnitude else "seismic",
            sym_cbar=not is_magnitude,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)

    fig.subplots_adjust(wspace=0.25)
    save_figure(fig, "helicity_slice_and_phi_average_magnitude")


# -----------------------------
# Explicit visu-dir mode
# -----------------------------
def _infer_case_dir_from_visu_dir(visu_dir: Path) -> Path:
    """
    Infer case directory from:
      .../<case>/runs/run3/visu_0020
    or
      .../<case>/run3/visu_0020
    """
    run_dir = visu_dir.parent
    if not _RUN_RE.match(run_dir.name):
        raise ValueError("--visu-dir must be inside run*/visu_####. Got: {}".format(visu_dir))

    parent = run_dir.parent
    if parent.name == "runs":
        return parent.parent
    return parent

def _plot_one_visu_dir(visu_dir: Path,
                       out: Optional[Path],
                       force: bool,
                       atphi: float,
                       show_grid: bool,
                       dry_run: bool):

    visu_dir = visu_dir.resolve()
    run_dir = visu_dir.parent
    m_run = _RUN_RE.match(run_dir.name)
    m_visu = _VISU_RE.match(visu_dir.name)
    if not m_run or not m_visu:
        raise ValueError("--visu-dir must match run*/visu_####. Got: {}".format(visu_dir))

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
        #out = fig_dir / "Ek_{}_q{}_Ra{}_{}_{}_snapshots.pdf".format(Ek, q, Ra, run_dir.name, tag)
        if udotgradu:
            out = fig_dir / "Ek_{}_q{}_Ra{}_{}_{}_udotgradu.pdf".format(Ek, q, Ra, run_dir.name, tag)
        else:
            out = fig_dir / "Ek_{}_q{}_Ra{}_{}_{}_snapshots.pdf".format(Ek, q, Ra, run_dir.name, tag)
    else:
        out = out.resolve()

    print("========================================")
    print("[INFO] case_dir : {}".format(case_dir))
    print("[INFO] run_dir  : {}".format(run_dir))
    print("[INFO] visu_dir : {}".format(visu_dir))
    print("[INFO] tag      : {}".format(tag))
    print("[INFO] npz      : {}".format(npz_path))
    print("[INFO] out      : {}".format(out))
    print("[INFO] grid     : {}".format(show_grid))
    print("========================================")

    if out.exists() and not force:
        print("[SKIP] Output exists (use --force to overwrite): {}".format(out))
        return

    if dry_run:
        return

    data = np.load(npz_path)
    plot_snapshot_panel(
        case_dir=case_dir,
        data=data,
        save_path=out,
        atphi=atphi,
        show_grid=show_grid,
    )

# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="snapshots: 2x3 panel (equatorial/meridional/CMB).")

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
    p.add_argument("--no-grid", action="store_true", help="Disable lon/lat grid on CMB plot.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print what would be plotted, then exit.")
    p.add_argument("--udotgradu", action="store_true",
                   help="Plot |u dot grad u| in equatorial + meridional planes (separate figure).")

    args = p.parse_args()
     

    print("[DEBUG] __file__ =", __file__)
    print("[DEBUG] argv =", sys.argv)
    print("[DEBUG] args.udotgradu =", args.udotgradu)
 
    if args.visu_dir is not None:
        visu_dir = Path(args.visu_dir)
        out = Path(args.out) if args.out is not None else None
  
        _plot_one_visu_dir(
            visu_dir=visu_dir,
            out=out,
            force=bool(args.force),
            atphi=float(args.atphi),
            show_grid=(not args.no_grid),
            dry_run=bool(args.dry_run),
            udotgradu=bool(args.udotgradu),
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
    print("[INFO] grid     : {}".format(not args.no_grid))
    print("========================================")

    if args.dry_run:
        return

    for t in chosen_tags:
        npz_path = _find_vis_fields_npz(run_dir, t)

        prefix = "Ek_{}_q{}_Ra{}_{}_{}".format(
            Ek, q, Ra, run_dir.name, t,
        )
        out = fig_dir / f"{prefix}_snapshots.pdf"

        with np.load(npz_path) as data:

            # ----------------------------------------------------
            # Main snapshot panel
            # ----------------------------------------------------
            if out.exists() and not args.force:
                print(
                    "[SKIP] Main figure exists "
                    f"(use --force to overwrite): {out}"
                )
            else:
                plot_snapshot_panel(
                    case_dir=case_dir,
                    data=data,
                    save_path=out,
                    atphi=args.atphi,
                    show_grid=not args.no_grid,
                )

            # ----------------------------------------------------
            # Helicity diagnostics
            # ----------------------------------------------------
            helicity_outputs = [
                fig_dir / f"{prefix}_{suffix}.pdf"
                for suffix in (
                    "axial_helicity_slice",
                    "axial_helicity_phi_average",
                    "helicity_slice_and_phi_average",
                )
            ]

            if args.force or not all(
                path.exists() for path in helicity_outputs
            ):
                plot_helicity_figures(
                    case_dir=case_dir,
                    data=data,
                    output_dir=fig_dir,
                    prefix=prefix,
                    atphi=args.atphi,
                    helicity_lim=6e6,              # left
                    helicity_magnitude_max=1e6,  # right
                )
            else:
                print(f"[SKIP] Helicity figures exist: {prefix}")

            # ----------------------------------------------------
            # Zonal-flow diagnostics
            # ----------------------------------------------------
            zonal_outputs = [
                fig_dir / f"{prefix}_{suffix}.pdf"
                for suffix in (
                    "zonal_flow",
                    "uphi_nonaxisym",
                )
            ]

            if args.force or not all(
                path.exists() for path in zonal_outputs
            ):
                plot_zonal_flow_figures(
                    case_dir=case_dir,
                    data=data,
                    output_dir=fig_dir,
                    prefix=prefix,
                    atphi=args.atphi,
                )
            else:
                print(f"[SKIP] Zonal-flow figures exist: {prefix}")
        
if __name__ == "__main__":
    main()


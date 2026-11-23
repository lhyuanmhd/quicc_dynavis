"""Cylindrical profiles of azimuthal velocity in a sphere or spherical shell.

Input ``data`` is a mapping with one-dimensional ``r``, ``theta`` (colatitude,
0..pi), ``phi`` (radians), and ``u_phi`` of shape (Nr, Ntheta, Nphi).

Definitions:
    zonal(s,z) = <u_phi(s,z,phi)>_phi
    Ug(s)      = (1/L(s)) int_column zonal(s,z) dz

For a full sphere: L(s) = 2 sqrt(ro**2-s**2).
For a shell with inner radius ri, integrate the two fluid segments separately.

The averaging over phi is done on the original spherical grid *before*
interpolation, which is much faster than evaluating a 3-D interpolator at each
(s,z,phi) and is equivalent for a linear interpolator.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid  # works with NumPy 1.x and 2.x
from scipy.interpolate import RegularGridInterpolator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "savefig.bbox": "tight",
})

_TWOPI = 2.0 * np.pi


def _sorted_unique_axis(x: np.ndarray, field: np.ndarray, axis: int,
                        name: str, tol: float = 1e-12):
    """Sort grid and reorder its field axis, dropping duplicate coordinates."""
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} grid contains non-finite values")
    order = np.argsort(x)
    xs = x[order]
    field = np.take(field, order, axis=axis)
    scale = max(1.0, float(np.max(np.abs(xs))))
    keep = np.r_[True, np.diff(xs) > tol * scale]
    if np.count_nonzero(keep) < 1:
        raise ValueError(f"{name} grid is empty")
    return xs[keep], np.take(field, np.flatnonzero(keep), axis=axis)


def _prepare_zonal_mean(data: Mapping, field_key: str = "u_phi",
                        ri: float = 0.0):
    """Return r, theta and a 2-D regular-grid interpolator for zonal u_phi."""
    r = np.asarray(data["r"], dtype=float).ravel()
    theta = np.asarray(data["theta"], dtype=float).ravel()
    phi = np.asarray(data["phi"], dtype=float).ravel()
    u = np.asarray(data[field_key], dtype=float)
    expected = (r.size, theta.size, phi.size)
    if u.shape != expected:
        raise ValueError(f"{field_key}.shape={u.shape}, expected {expected}")
    if phi.size < 1:
        raise ValueError("Empty azimuthal grid")

    r, u = _sorted_unique_axis(r, u, axis=0, name="r")
    theta, u = _sorted_unique_axis(theta, u, axis=1, name="theta")
    phi, u = _sorted_unique_axis(np.mod(phi, _TWOPI), u, axis=2, name="phi")
    if r.size < 2 or theta.size < 2:
        raise ValueError("Interpolation requires at least two r and theta nodes")
    if theta[0] < -1e-10 or theta[-1] > np.pi + 1e-10:
        raise ValueError("theta must be colatitude in [0, pi] (radians)")
    if not np.all(np.isfinite(u)):
        raise ValueError(f"{field_key} contains NaNs or infinities")

    if phi.size == 1:
        zonal = u[:, :, 0]  # axisymmetric input
    else:
        # The periodic trapezoidal rule handles both uniform and nonuniform phi.
        phi_ext = np.r_[phi, phi[0] + _TWOPI]
        u_ext = np.concatenate([u, u[:, :, :1]], axis=2)
        zonal = trapezoid(u_ext, x=phi_ext, axis=2) / _TWOPI

    # For a regular vector field, the zonal azimuthal component vanishes on the
    # rotation axis. Spectral angular grids often omit theta=0 and theta=pi.
    tol = 1e-12
    if theta[0] > tol:
        theta = np.r_[0.0, theta]
        zonal = np.concatenate([np.zeros_like(zonal[:, :1]), zonal], axis=1)
    if theta[-1] < np.pi - tol:
        theta = np.r_[theta, np.pi]
        zonal = np.concatenate([zonal, np.zeros_like(zonal[:, :1])], axis=1)

    # For a *full sphere* whose radial collocation excludes exactly r=0,
    # enforce regularity at the centre. Never do this for a spherical shell.
    if ri <= 0.0 and r[0] > tol:
        if r[0] / r[-1] > 0.10:
            raise ValueError(
                "r does not approach zero. If this is a shell, pass ri=<inner radius>."
            )
        r = np.r_[0.0, r]
        zonal = np.concatenate([np.zeros_like(zonal[:1]), zonal], axis=0)

    interp = RegularGridInterpolator(
        (r, theta), zonal, method="linear", bounds_error=False,
        fill_value=np.nan,
    )
    return r, theta, interp


def _sample_zonal(interp, s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate <u_phi>_phi on broadcastable cylindrical (s,z) arrays."""
    s, z = np.broadcast_arrays(np.asarray(s, float), np.asarray(z, float))
    radius = np.hypot(s, z)
    theta = np.arccos(np.clip(np.divide(
        z, radius, out=np.zeros_like(radius), where=radius > 0.0
    ), -1.0, 1.0))
    result = interp(np.column_stack([radius.ravel(), theta.ravel()])).reshape(s.shape)
    # A smooth vector field has vanishing zonal azimuthal velocity at s=0.
    return np.where(np.abs(s) < 1e-14, 0.0, result)


def _s_grid(ro: float, n_s: int, s_max: float | None):
    if int(n_s) < 2:
        raise ValueError("n_s must be >= 2")
    if s_max is None:
        s_max = ro
    if not 0 < s_max <= ro + 1e-12:
        raise ValueError(f"s_max must lie in (0, ro={ro}]")
    return np.linspace(0.0, float(s_max), int(n_s))


def compute_azimuthally_averaged_uphi(
    data, *, z: float = 0.0, n_s: int = 200,
    s_max: float | None = None, ro: float | None = None,
    ri: float = 0.0, field_key: str = "u_phi",
):
    """Compute <u_phi>_phi(s,z) at a fixed cylindrical height z.

    Returns (s, u_phi_zonal). Points outside the fluid domain are NaN.
    """
    r, _, interp = _prepare_zonal_mean(data, field_key, ri=ri)
    ro = float(r[-1] if ro is None else ro)
    if not 0 <= ri < ro <= r[-1] + 1e-12:
        raise ValueError("Require 0 <= ri < ro <= max(r)")
    s = _s_grid(ro, n_s, s_max)
    z_arr = np.full_like(s, float(z))
    u = _sample_zonal(interp, s, z_arr)
    rr = np.hypot(s, z_arr)
    u[(rr > ro + 1e-12) | (rr < ri - 1e-12)] = np.nan
    return s, u


def _compute_geostrophic_from_interp(
    interp, s: np.ndarray, *, ro: float, ri: float, n_z: int
):
    if int(n_z) < 2:
        raise ValueError("n_z must be >= 2")
    ug = np.full_like(s, np.nan, dtype=float)
    for i, sv in enumerate(s):
        if sv >= ro:
            continue  # the column has zero length at the outer tangent point
        ho = float(np.sqrt(max(0.0, ro**2 - sv**2)))
        hi = float(np.sqrt(max(0.0, ri**2 - sv**2))) if sv < ri else 0.0
        if hi > 0:
            segments = [(-ho, -hi), (hi, ho)]
            length = 2.0 * (ho - hi)
        else:
            segments = [(-ho, ho)]
            length = 2.0 * ho
        if length <= 0:
            continue

        total = 0.0
        valid = True
        for za, zb in segments:
            zz = np.linspace(za, zb, int(n_z))
            zz_s = np.full_like(zz, sv)
            values = _sample_zonal(interp, zz_s, zz)
            # Do not let NaNs silently contaminate the quadrature or
            # accidentally integrate across missing parts of a fluid segment.
            if not np.all(np.isfinite(values)):
                valid = False
                break
            total += float(trapezoid(values, x=zz))
        if valid:
            ug[i] = total / length
    return ug


def compute_column_averaged_geostrophic_flow(
    data, n_s: int = 200, n_z: int = 256, n_phi=None,
    s_max: float | None = None, ro: float | None = None,
    ri: float = 0.0, field_key: str = "u_phi",
):
    """Compute Ug(s)=<u_phi>_(phi,z) along a full fluid column.

    ``n_phi`` is retained for compatibility; azimuthal averaging now uses
    the existing phi grid rather than repeated 3-D interpolation.
    Default ri=0 identifies a full sphere (NOT min(r) of a spectral grid).
    """
    if n_phi is not None:
        warnings.warn("n_phi is no longer needed: native phi grid is used.",
                      FutureWarning, stacklevel=2)
    r, _, interp = _prepare_zonal_mean(data, field_key, ri=ri)
    ro = float(r[-1] if ro is None else ro)
    if not 0 <= ri < ro <= r[-1] + 1e-12:
        raise ValueError("Require 0 <= ri < ro <= max(r)")
    s = _s_grid(ro, n_s, s_max)
    return s, _compute_geostrophic_from_interp(interp, s, ro=ro, ri=ri, n_z=n_z)


def plot_column_averaged_geostrophic_flow(
    data_sources, n_s: int = 200, n_z: int = 256, n_phi=None,
    s_max: float | None = None, ax=None, savefig=None, show: bool = True,
    field_key: str = "u_phi", ro: float | None = None, ri: float = 0.0,
):
    """Backward-compatible plot; returns {label: (s, Ug)}."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    else:
        fig = ax.figure
    out = {}
    for label, data in data_sources.items():
        s, ug = compute_column_averaged_geostrophic_flow(
            data, n_s=n_s, n_z=n_z, n_phi=n_phi, s_max=s_max,
            ro=ro, ri=ri, field_key=field_key,
        )
        out[label] = (s, ug)
        ax.plot(s, ug, lw=1.7, label=label)
    ax.set(xlabel=r"$s$", ylabel=r"$U_g(s)$")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    if savefig is not None:
        fig.savefig(savefig, dpi=300)
    if show and created:
        plt.show()
    return out


def plot_uphi_and_geostrophic_flow(
    data_sources, *, z: float = 0.0, n_s: int = 200,
    n_z: int = 256, s_max: float | None = None,
    ro: float | None = None, ri: float = 0.0,
    field_key: str = "u_phi", savefig=None, show: bool = True,
    figsize=(10, 4.3), axes=None,
):
    """Two panels: zonal u_phi(s,z) and depth-averaged geostrophic Ug(s).

    Compute each dataset's zonal 2-D interpolator only once.
    Returns {label: {"s": s, "uphi_zonal": u, "Ug": Ug}}.
    """
    created = axes is None
    if created:
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=figsize,
                                 constrained_layout=True)
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain exactly two Matplotlib Axes")
        fig = axes[0].figure
    out = {}
    for label, data in data_sources.items():
        r, _, interp = _prepare_zonal_mean(data, field_key=field_key, ri=ri)
        outer = float(r[-1] if ro is None else ro)
        if not 0 <= ri < outer <= r[-1] + 1e-12:
            raise ValueError("Require 0 <= ri < ro <= max(r)")
        s = _s_grid(outer, n_s, s_max)
        u = _sample_zonal(interp, s, np.full_like(s, z))
        rr = np.hypot(s, z)
        u[(rr > outer + 1e-12) | (rr < ri - 1e-12)] = np.nan
        ug = _compute_geostrophic_from_interp(
            interp, s, ro=outer, ri=ri, n_z=n_z,
        )
        out[label] = {"s": s, "uphi_zonal": u, "Ug": ug}
        axes[0].plot(s, u, lw=1.6, label=label)
        axes[1].plot(s, ug, lw=1.6, label=label)

    axes[0].set(ylabel=r"$\langle u_\phi\rangle_\phi(s,z)$",
                title=rf"Azimuthal mean at $z={z:g}$")
    axes[1].set(ylabel=r"$U_g(s)=\langle u_\phi\rangle_{\phi,z}$",
                title="Column-averaged geostrophic flow")
    for ax in axes:
        ax.set_xlabel(r"$s$")
        ax.axhline(0, color="0.45", ls="--", lw=0.7)
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    if savefig is not None:
        fig.savefig(savefig, dpi=300)
    if show and created:
        plt.show()
    return out


def plot_uphi_vs_s_fixed_z_phi(
    data_sources, z_list=(0.0, 0.5), phi0: float = 0.0,
    n_s: int = 200, ax=None, savefig=None, show: bool = True,
    ylim=None, field_key: str = "u_phi",
):
    """Legacy instantaneous u_phi(s,z,phi0), NOT an azimuthal average.

    Correctly handles sorted theta/r/phi and periodic phi even if phi[0]>0.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    else:
        fig = ax.figure
    for label, data in data_sources.items():
        r = np.asarray(data["r"], float)
        theta = np.asarray(data["theta"], float)
        phi = np.mod(np.asarray(data["phi"], float), _TWOPI)
        u = np.asarray(data[field_key], float)
        if u.shape != (len(r), len(theta), len(phi)):
            raise ValueError("Expected u_phi.shape=(Nr,Ntheta,Nphi)")
        r, u = _sorted_unique_axis(r, u, 0, "r")
        theta, u = _sorted_unique_axis(theta, u, 1, "theta")
        phi, u = _sorted_unique_axis(phi, u, 2, "phi")
        if phi.size < 2:
            raise ValueError("Fixed-phi plot requires at least two phi nodes")
        phi_ext = np.r_[phi[-1] - _TWOPI, phi, phi[0] + _TWOPI]
        u_ext = np.concatenate([u[:, :, -1:], u, u[:, :, :1]], axis=2)
        interpolator = RegularGridInterpolator(
            (r, theta, phi_ext), u_ext, bounds_error=False,
            fill_value=np.nan,
        )
        phi_query = phi[0] + ((phi0 - phi[0]) % _TWOPI)
        for z in z_list:
            if abs(z) >= r[-1]:
                continue
            ss = np.linspace(0, np.sqrt(r[-1]**2 - z**2), int(n_s))
            rr = np.hypot(ss, z)
            th = np.arccos(np.clip(
                np.divide(z, rr, out=np.zeros_like(rr), where=rr > 0),
                -1.0, 1.0,
            ))
            points = np.column_stack([rr, th, np.full_like(ss, phi_query)])
            vals = interpolator(points)
            vals[rr == 0.0] = np.nan  # fixed-phi component is undefined at origin
            ax.plot(ss, vals, lw=1.5, label=f"{label}, z={z:g}")
    ax.set(xlabel=r"$s$", ylabel=r"$u_\phi(s,z,\phi_0)$")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    if savefig is not None:
        fig.savefig(savefig, dpi=300)
    if show and created:
        plt.show()


# Example usage (after loading the original simulation data):
# data_sources = {"$L_B=40$": data40, "$L_B=60$": data60, "$L_B=80$": data80}
# result = plot_uphi_and_geostrophic_flow(
#     data_sources, z=0.0, n_s=200, n_z=256,
#     savefig="uphi_zonal_and_geostrophic.pdf", show=False,
# )
# s = result["$L_B=80$"]["s"]
# uphi_zonal = result["$L_B=80$"]["uphi_zonal"]
# Ug = result["$L_B=80$"]["Ug"]

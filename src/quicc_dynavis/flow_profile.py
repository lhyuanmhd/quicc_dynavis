"""Azimuthal-velocity profiles in full spheres and spherical shells.

Expected NPZ/mapping keys:
    r      : radial coordinates, shape (Nr,)
    theta  : colatitude (radians), shape (Ntheta,)
    phi    : azimuth (radians), shape (Nphi,)
    u_phi  : azimuthal velocity, shape (Nr, Ntheta, Nphi)

Three distinct diagnostics are provided:

1. Fixed-ray instantaneous velocity: u_phi(s,z,phi0)
2. Azimuthal mean:             <u_phi(s,z,phi)>_phi
3. Column-averaged zonal flow: Ug(s) = (1/L(s)) integral_column <u_phi>_phi dz

Use ``plot_uphi_vs_s_fixed_z_phi`` for instantaneous velocity comparisons:
it does NOT perform an azimuthal average. A fixed ray generally includes m != 0.

Designed for NumPy 1.x/2.x (uses SciPy's trapezoid, not np.trapz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

_TWOPI = 2.0 * np.pi


def _sort_axis(x, values, axis: int, name: str, tol: float = 1e-12):
    """Sort coordinates and reorder field, dropping duplicate grid points."""
    x = np.asarray(x, dtype=float).ravel()
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError(f"Invalid {name} grid: empty or non-finite")
    order = np.argsort(x)
    xs = x[order]
    values = np.take(values, order, axis=axis)
    scale = max(1.0, np.max(np.abs(xs)))
    keep = np.r_[True, np.diff(xs) > tol * scale]
    return xs[keep], np.take(values, np.flatnonzero(keep), axis=axis)


def _read_grids_and_field(data: Mapping, field_key: str):
    r = np.asarray(data["r"], dtype=float).ravel()
    theta = np.asarray(data["theta"], dtype=float).ravel()
    phi = np.asarray(data["phi"], dtype=float).ravel()
    u = np.asarray(data[field_key], dtype=float)
    expected = (len(r), len(theta), len(phi))
    if u.shape != expected:
        raise ValueError(f"Expected {field_key}.shape={expected}, got {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError(f"{field_key} contains NaN or infinite values")

    # QuICC grids may be descending; an azimuthal grid may include 2*pi.
    r, u = _sort_axis(r, u, 0, "r")
    theta, u = _sort_axis(theta, u, 1, "theta")
    phi, u = _sort_axis(np.mod(phi, _TWOPI), u, 2, "phi")
    if len(r) < 2 or len(theta) < 2:
        raise ValueError("At least 2 radial and 2 colatitude points are required")
    if theta[0] < -1e-10 or theta[-1] > np.pi + 1e-10:
        raise ValueError("theta must be colatitude in [0,pi], in radians")
    return r, theta, phi, u


def _fixed_phi_grid(phi: np.ndarray, u: np.ndarray, phi0: float):
    """Interpolate just once in periodic phi; return a (Nr,Ntheta) slice.

    Much cheaper than a full 3-D RegularGridInterpolator for a fixed-phi ray.
    """
    if len(phi) == 1:
        return u[..., 0]

    # Shift into a single period beginning at the first stored grid point.
    target = phi[0] + ((float(phi0) - phi[0]) % _TWOPI)
    k = int(np.searchsorted(phi, target, side="right"))
    if k == len(phi):
        pa, pb = phi[-1], phi[0] + _TWOPI
        ua, ub = u[..., -1], u[..., 0]
    else:
        pa, pb = phi[k - 1], phi[k]
        ua, ub = u[..., k - 1], u[..., k]
    weight = (target - pa) / (pb - pa)
    return (1.0 - weight) * ua + weight * ub


def _zonal_grid(phi: np.ndarray, u: np.ndarray):
    """Periodic trapezoidal phi mean on the native azimuth grid."""
    if len(phi) == 1:
        return u[..., 0]
    ph_ext = np.r_[phi, phi[0] + _TWOPI]
    u_ext = np.concatenate([u, u[..., :1]], axis=-1)
    return trapezoid(u_ext, x=ph_ext, axis=-1) / _TWOPI


def _interp_2d(r: np.ndarray, theta: np.ndarray, slice_2d: np.ndarray,
               *, zonal: bool, ri: float):
    """Construct (r,theta) interpolator, with safe zonal-only axis padding."""
    if zonal:
        # The phi-averaged u_phi vanishes at the rotation axis for a regular
        # Cartesian vector field. This is NOT imposed for a fixed-phi field.
        if theta[0] > 1e-12:
            theta = np.r_[0.0, theta]
            slice_2d = np.concatenate(
                [np.zeros_like(slice_2d[:, :1]), slice_2d], axis=1
            )
        if theta[-1] < np.pi - 1e-12:
            theta = np.r_[theta, np.pi]
            slice_2d = np.concatenate(
                [slice_2d, np.zeros_like(slice_2d[:, :1])], axis=1
            )
        if ri == 0.0 and r[0] > 1e-12:
            if r[0] / r[-1] > 0.1:
                raise ValueError("Grid does not approach r=0; set ri for a shell")
            r = np.r_[0.0, r]
            slice_2d = np.concatenate(
                [np.zeros_like(slice_2d[:1, :]), slice_2d], axis=0
            )

    return RegularGridInterpolator(
        (r, theta), slice_2d, bounds_error=False, fill_value=np.nan,
        method="linear"
    )


def _evaluate(interp, s, z):
    """Vectorised spherical-coordinate sampling of a (s,z) cylindrical ray."""
    s, z = np.broadcast_arrays(np.asarray(s, float), np.asarray(z, float))
    rr = np.hypot(s, z)
    theta = np.arccos(np.clip(
        np.divide(z, rr, out=np.zeros_like(rr), where=rr > 0),
        -1.0, 1.0
    ))
    pts = np.column_stack((rr.ravel(), theta.ravel()))
    return interp(pts).reshape(s.shape)


def _bounds(r, ro, ri):
    ro = float(r[-1] if ro is None else ro)
    ri = float(ri)
    if not 0 <= ri < ro <= r[-1] + 1e-12:
        raise ValueError("Require 0 <= ri < ro <= max(r)")
    return ro, ri


def _s_grid(ro, n_s, s_min, s_max):
    if int(n_s) < 2:
        raise ValueError("n_s must be at least 2")
    if s_max is None:
        s_max = ro
    if not 0 <= s_min < s_max <= ro + 1e-12:
        raise ValueError("Require 0 <= s_min < s_max <= ro")
    return np.linspace(float(s_min), float(s_max), int(n_s))


def compute_uphi_vs_s_fixed_z_phi(
    data, z: float = 0.0, phi0: float = 0.0, n_s: int = 250,
    s_max: float | None = None, ro: float | None = None,
    ri: float = 0.0, field_key: str = "u_phi", s_min: float = 0.0,
):
    """Instantaneous u_phi(s,z,phi0), *without* phi averaging.

    Returns ``(s, u_phi)``. Points outside the fluid and the origin (r=0),
    where the spherical vector component is ambiguous, are NaN.

    To exclude suspicious outer boundary points from a plot, set e.g.
    ``s_max=0.98`` explicitly rather than silently filtering data.
    """
    r, theta, phi, u = _read_grids_and_field(data, field_key)
    ro, ri = _bounds(r, ro, ri)
    s = _s_grid(ro, n_s, s_min, s_max)
    uphi_phi0 = _fixed_phi_grid(phi, u, phi0)
    interp = _interp_2d(r, theta, uphi_phi0, zonal=False, ri=ri)
    zz = np.full_like(s, float(z))
    values = _evaluate(interp, s, zz)
    rr = np.hypot(s, zz)
    values[(rr > ro + 1e-12) | (rr < ri - 1e-12) | (rr <= 1e-14)] = np.nan
    return s, values


def _make_zonal_interpolator(data, field_key, ri):
    r, theta, phi, u = _read_grids_and_field(data, field_key)
    mean = _zonal_grid(phi, u)
    interp = _interp_2d(r, theta, mean, zonal=True, ri=ri)
    return r, interp


def compute_azimuthally_averaged_uphi(
    data, *, z: float = 0.0, n_s: int = 200,
    s_max: float | None = None, ro: float | None = None,
    ri: float = 0.0, field_key: str = "u_phi", s_min: float = 0.0,
):
    """Azimuthally averaged <u_phi>_phi(s,z), not the fixed-phi value."""
    r, interp = _make_zonal_interpolator(data, field_key, ri)
    ro, ri = _bounds(r, ro, ri)
    s = _s_grid(ro, n_s, s_min, s_max)
    zz = np.full_like(s, z)
    values = _evaluate(interp, s, zz)
    rr = np.hypot(s, zz)
    values[(rr > ro + 1e-12) | (rr < ri - 1e-12)] = np.nan
    # Azimuthal mean of u_phi is exactly zero on the rotation axis.
    values[np.abs(s) < 1e-14] = 0.0
    return s, values


def _integrate_columns(interp, s, ro, ri, n_z):
    if int(n_z) < 2:
        raise ValueError("n_z must be at least 2")
    ug = np.full_like(s, np.nan)
    for i, sval in enumerate(s):
        if sval >= ro:
            continue  # zero-length column at tangent boundary
        h_outer = float(np.sqrt(max(ro**2 - sval**2, 0.0)))
        h_inner = float(np.sqrt(max(ri**2 - sval**2, 0.0))) if sval < ri else 0.0
        if h_inner > 0:
            segments = [(-h_outer, -h_inner), (h_inner, h_outer)]
            length = 2.0 * (h_outer - h_inner)
        else:
            segments = [(-h_outer, h_outer)]
            length = 2.0 * h_outer
        total = 0.0
        for a, b in segments:
            z = np.linspace(a, b, int(n_z))
            value = _evaluate(interp, np.full_like(z, sval), z)
            if not np.all(np.isfinite(value)):
                break  # do not silently integrate across missing values
            total += float(trapezoid(value, x=z))
        else:
            ug[i] = total / length
    return ug


def compute_column_averaged_geostrophic_flow(
    data, n_s: int = 200, n_z: int = 256, n_phi=None,
    s_max: float | None = None, ro: float | None = None,
    ri: float = 0.0, field_key: str = "u_phi", s_min: float = 0.0,
):
    """Return ``(s, Ug)`` with Ug=(1/L) integral_column <u_phi>_phi dz.

    ``n_phi`` is accepted for backward compatibility; the native phi grid
    is now used for accurate/efficient azimuthal averaging.
    """
    if n_phi is not None:
        warnings.warn("n_phi is ignored; using the native azimuthal grid",
                      FutureWarning, stacklevel=2)
    r, interp = _make_zonal_interpolator(data, field_key, ri)
    ro, ri = _bounds(r, ro, ri)
    s = _s_grid(ro, n_s, s_min, s_max)
    return s, _integrate_columns(interp, s, ro, ri, n_z)


def _decorate(ax, *, xlabel=r"$s$", ylabel=None, title=None, ylim=None,
              xlim=None, legend=True):
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.axhline(0.0, color="0.45", lw=0.7, ls="--")
    ax.grid(alpha=0.2, lw=0.7)
    if legend:
        ax.legend(frameon=False, loc="best")


def plot_uphi_vs_s_fixed_z_phi(
    data_sources, z_list=(0.0,), phi0: float = 0.0,
    n_s: int = 250, ax=None, savefig=None, show: bool = True,
    ylim=None, field_key: str = "u_phi", s_min: float = 0.0,
    s_max: float | None = None, ro: float | None = None,
    ri: float = 0.0, figsize=(7, 4.5), title: bool = False,
):
    """Plot instantaneous radial rays for multiple resolutions.

    Legacy name/signature retained; default z_list=(0,) for this use case.
    Returns {label: {z: (s, u_phi)}}.
    """
    if np.isscalar(z_list):
        z_list = [float(z_list)]
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    result = {}
    for label, data in data_sources.items():
        result[label] = {}
        for z in z_list:
            s, uphi = compute_uphi_vs_s_fixed_z_phi(
                data, z=float(z), phi0=phi0, n_s=n_s,
                s_min=s_min, s_max=s_max, ro=ro, ri=ri,
                field_key=field_key,
            )
            result[label][float(z)] = (s, uphi)
            legend_label = label if len(z_list) == 1 else f"{label}, $z={z:g}$"
            ax.plot(s, uphi, lw=1.7, label=legend_label)
    _decorate(
        ax, ylabel=r"$u_\phi(s,z,\phi_0)$",
        title=rf"$u_\phi(s)$ at $z={z_list[0]:g}$, $\phi={phi0:g}$" if title and len(z_list) == 1 else None,
        ylim=ylim, xlim=(s_min, s_max) if s_max is not None else None,
    )
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
    if show and created:
        plt.show()
    return result


def plot_column_averaged_geostrophic_flow(
    data_sources, n_s: int = 200, n_z: int = 256, n_phi=None,
    s_max: float | None = None, ax=None, savefig=None,
    show: bool = True, field_key: str = "u_phi",
    ro: float | None = None, ri: float = 0.0,
    s_min: float = 0.0, ylim=None, figsize=(7, 4.5),
):
    """Plot Ug(s); legacy return format {label: (s, Ug)} retained."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    result = {}
    for label, data in data_sources.items():
        s, ug = compute_column_averaged_geostrophic_flow(
            data, n_s=n_s, n_z=n_z, n_phi=n_phi, s_min=s_min,
            s_max=s_max, ro=ro, ri=ri, field_key=field_key,
        )
        result[label] = (s, ug)
        ax.plot(s, ug, lw=1.7, label=label)
    _decorate(ax, ylabel=r"$U_g(s)$", ylim=ylim,
              xlim=(s_min, s_max) if s_max is not None else None)
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
    if show and created:
        plt.show()
    return result


def plot_uphi_and_geostrophic_flow(
    data_sources, *, z: float = 0.0, n_s: int = 200,
    n_z: int = 256, s_max: float | None = None,
    ro: float | None = None, ri: float = 0.0,
    field_key: str = "u_phi", savefig=None, show: bool = True,
    figsize=(10, 4.3), axes=None, s_min: float = 0.0,
):
    """Legacy two-panel diagnostic: phi-averaged u_phi and Ug (not fixed phi)."""
    created = axes is None
    if created:
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain two Axes")
        fig = axes[0].figure
    result = {}
    for label, data in data_sources.items():
        r, interp = _make_zonal_interpolator(data, field_key, ri)
        outer, inner = _bounds(r, ro, ri)
        s = _s_grid(outer, n_s, s_min, s_max)
        z_values = np.full_like(s, z)
        rr = np.hypot(s, z_values)
        zonal = _evaluate(interp, s, z_values)
        zonal[(rr > outer + 1e-12) | (rr < inner - 1e-12)] = np.nan
        zonal[np.abs(s) < 1e-14] = 0.0
        ug = _integrate_columns(interp, s, outer, inner, n_z)
        result[label] = {"s": s, "uphi_zonal": zonal, "Ug": ug}
        axes[0].plot(s, zonal, lw=1.6, label=label)
        axes[1].plot(s, ug, lw=1.6, label=label)
    _decorate(axes[0], ylabel=r"$\langle u_\phi\rangle_\phi(s,z)$",
              title=rf"Azimuthal mean at $z={z:g}$")
    _decorate(axes[1], ylabel=r"$U_g(s)$",
              title="Column-averaged geostrophic flow", legend=False)
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
    if show and created:
        plt.show()
    return result


__all__ = [
    "compute_uphi_vs_s_fixed_z_phi",
    "compute_azimuthally_averaged_uphi",
    "compute_column_averaged_geostrophic_flow",
    "plot_uphi_vs_s_fixed_z_phi",
    "plot_column_averaged_geostrophic_flow",
    "plot_uphi_and_geostrophic_flow",
]

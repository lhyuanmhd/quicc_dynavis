import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def _ensure_strictly_increasing_1d(x, axis_name="x", eps=0.0):
    """
    Make a 1D grid strictly increasing by sorting and removing duplicates.
    Returns:
      x_new, idx_keep, order
    where:
      x_sorted = x[order]
      x_new = x_sorted[idx_keep]
    """
    x = np.asarray(x).astype(float)

    # sort
    order = np.argsort(x)
    xs = x[order]

    # remove duplicates / non-strict steps
    # strictly increasing => diff > 0
    dx = np.diff(xs)
    keep = np.ones_like(xs, dtype=bool)
    if eps <= 0:
        keep[1:] = dx > 0
    else:
        keep[1:] = dx > eps

    xs_new = xs[keep]

    if xs_new.size < 2:
        raise ValueError(f"{axis_name} grid becomes too small after enforcing strict monotonicity.")

    return xs_new, keep, order


def _reorder_field(field, axis, order, keep):
    """
    Reorder `field` along `axis` according to `order` (sort) then subset with `keep`.
    """
    f = np.take(field, order, axis=axis)
    idx = np.where(keep)[0]
    f = np.take(f, idx, axis=axis)
    return f


def _prepare_grids_and_field_for_interp(data, field_key="u_phi"):
    """
    Ensure r, theta, phi are strictly increasing, and reorder field accordingly.
    Also wrap phi into [0, 2pi) before sorting.
    """
    r = np.asarray(data["r"]).astype(float)
    theta = np.asarray(data["theta"]).astype(float)
    phi = np.asarray(data["phi"]).astype(float)
    field = np.asarray(data[field_key]).astype(float)

    if field.shape != (len(r), len(theta), len(phi)):
        raise ValueError(
            f"{field_key} shape {field.shape} != (Nr,Ntheta,Nphi)=({len(r)},{len(theta)},{len(phi)})"
        )

    # --- normalize phi into [0, 2pi) to avoid weird ordering across 2pi ---
    twopi = 2.0 * np.pi
    phi = np.mod(phi, twopi)

    # --- enforce strictly increasing r, theta, phi ---
    r_new, r_keep, r_order = _ensure_strictly_increasing_1d(r, "r")
    field = _reorder_field(field, axis=0, order=r_order, keep=r_keep)

    th_new, th_keep, th_order = _ensure_strictly_increasing_1d(theta, "theta")
    field = _reorder_field(field, axis=1, order=th_order, keep=th_keep)

    ph_new, ph_keep, ph_order = _ensure_strictly_increasing_1d(phi, "phi")
    field = _reorder_field(field, axis=2, order=ph_order, keep=ph_keep)

    return r_new, th_new, ph_new, field


def compute_column_averaged_geostrophic_flow(
    data,
    n_s=200,
    n_z=256,
    n_phi=None,     # None -> use (processed) phi grid; else uniform sampling
    s_max=None,
    ro=None,        # None -> max(r)
    ri=None,        # None -> min(r) (shell); full sphere typically ri~0
    field_key="u_phi",
):
    """
    Column-averaged geostrophic flow:
      Ug(s) = (1/L(s)) * ∫ <u_phi>_phi(s,z) dz   over the fluid column at cylindrical radius s.

    Robust to theta being decreasing / unsorted / having duplicates.
    """

    # --- preprocess grids so RegularGridInterpolator is happy ---
    r, theta, phi, u_phi = _prepare_grids_and_field_for_interp(data, field_key=field_key)

    if ro is None:
        ro = float(np.nanmax(r))
    if ri is None:
        ri = float(np.nanmin(r))

    # --- make phi periodic for interpolation ---
    # append 2pi endpoint and first slice
    phi_periodic = np.hstack([phi, 2*np.pi])
    u_phi_periodic = np.concatenate([u_phi, u_phi[:, :, 0:1]], axis=2)

    interp = RegularGridInterpolator(
        (r, theta, phi_periodic),
        u_phi_periodic,
        bounds_error=False,
        fill_value=np.nan,
    )

    # phi samples for phi-average
    if n_phi is None:
        phi_samp = phi  # already in [0,2pi), strictly increasing
    else:
        phi_samp = np.linspace(0.0, 2*np.pi, int(n_phi), endpoint=False)

    # s grid
    if s_max is None:
        s_max = ro
    s = np.linspace(0.0, float(s_max), int(n_s))
    Ug = np.full_like(s, np.nan, dtype=float)

    def uphi_phiavg_at_sz(sv, zv):
        rv = np.sqrt(sv * sv + zv * zv)
        if rv <= 0.0:
            return np.nan
        thetav = np.arccos(np.clip(zv / rv, -1.0, 1.0))
        pts = np.column_stack([
            np.full_like(phi_samp, rv),
            np.full_like(phi_samp, thetav),
            phi_samp
        ])
        vals = interp(pts)
        return np.nanmean(vals)

    for i, sv in enumerate(s):
        if sv > ro:
            continue

        Ho2 = ro * ro - sv * sv
        if Ho2 <= 0:
            continue
        Ho = np.sqrt(Ho2)

        # determine column segments
        if ri <= 0 or sv >= ri:
            segments = [(-Ho, Ho)]
            L = 2 * Ho
        else:
            Hi2 = ri * ri - sv * sv
            if Hi2 <= 0:
                segments = [(-Ho, Ho)]
                L = 2 * Ho
            else:
                Hi = np.sqrt(Hi2)
                segments = [(-Ho, -Hi), (Hi, Ho)]
                L = 2 * (Ho - Hi)

        integral = 0.0
        ok = False
        for za, zb in segments:
            zz = np.linspace(za, zb, int(n_z))
            uu = np.array([uphi_phiavg_at_sz(sv, zval) for zval in zz])
            if np.all(np.isnan(uu)):
                continue
            integral += np.trapz(uu, zz)
            ok = True

        if ok and L > 0:
            Ug[i] = integral / L

    return s, Ug


def plot_column_averaged_geostrophic_flow(
    data_sources,   # dict: {label: data}
    n_s=200,
    n_z=256,
    n_phi=None,
    s_max=None,
    ax=None,
    savefig=None,
    show=True,
    field_key="u_phi",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    out = {}
    for label, data in data_sources.items():
        s, Ug = compute_column_averaged_geostrophic_flow(
            data,
            n_s=n_s, n_z=n_z, n_phi=n_phi, s_max=s_max,
            field_key=field_key
        )
        out[label] = (s, Ug)
        ax.plot(s, Ug, lw=1.8, label=label)

    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$U_g(s)$")
    ax.set_title("Column-averaged geostrophic flow")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if savefig is not None:
        plt.savefig(savefig, dpi=200, bbox_inches="tight")
    if show and "fig" in locals():
        plt.show()

    return out





def plot_uphi_vs_s_fixed_z_phi(
    data_sources,
    z_list=(0.0, 0.5),
    phi0=0.0,
    n_s=200,
    ax=None,
    savefig=None,
    show=True,
    ylim=None
):
    """
    Plot u_phi(s) at fixed z and phi for multiple datasets.

    Parameters
    ----------
    data_sources : dict
        {"label": data}

    z_list : list
        heights z

    phi0 : float
        fixed phi

    n_s : int
        resolution in s
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,5))

    for label, data in data_sources.items():

        r = np.asarray(data["r"])
        theta = np.asarray(data["theta"])
        phi = np.asarray(data["phi"])
        u_phi = np.asarray(data["u_phi"])

        # ensure ascending theta
        if np.any(np.diff(theta) < 0):
            order = np.argsort(theta)
            theta = theta[order]
            u_phi = u_phi[:,order,:]

        # periodic phi
        phi_periodic = np.hstack([phi, 2*np.pi])
        u_phi_periodic = np.concatenate([u_phi, u_phi[:,:,0:1]], axis=2)

        interp = RegularGridInterpolator(
            (r, theta, phi_periodic),
            u_phi_periodic,
            bounds_error=False,
            fill_value=np.nan
        )

        ro = np.max(r)

        for z in z_list:

            s_max = np.sqrt(ro**2 - z**2)
            s = np.linspace(0, s_max, n_s)

            uphi = np.zeros_like(s)

            for i, s_val in enumerate(s):

                r_val = np.sqrt(s_val**2 + z**2)

                if r_val == 0:
                    uphi[i] = np.nan
                    continue

                theta_val = np.arccos(z / r_val)

                point = np.array([[r_val, theta_val, phi0]])
                uphi[i] = interp(point)[0]

            ax.plot(
                s,
                uphi,
                label=f"{label}, z={z}"
            )

    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$u_\phi$")
    ax.set_title(r"$u_\phi(s)$ at fixed $z$, $\phi=0$")
    ax.grid(alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()

  
    if savefig is not None:
        plt.savefig(savefig, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

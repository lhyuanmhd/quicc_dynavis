import h5py
import numpy as np
import matplotlib.pyplot as plt

def _read_scalar(ds):
    val = ds[()]
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    if isinstance(val, (np.generic,)):
        return val.item()
    return val

def _read_string_dataset(ds):
    val = ds[()]
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.shape == ():
            v = val.item()
            if isinstance(v, (bytes, np.bytes_)):
                return v.decode("utf-8", errors="replace")
            return str(v)
        if val.size == 1:
            v = val.ravel()[0]
            if isinstance(v, (bytes, np.bytes_)):
                return v.decode("utf-8", errors="replace")
            return str(v)
    return str(val)

def _d_dphi_periodic(a, dphi, axis=2):
    """Second-order centered periodic derivative in phi (uniform spacing)."""
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2.0 * dphi)

def extract_fields_with_curl_fixed(
    hdf5_file,
    output_file="vis_fields_fixed.npz",
    include_curl=True,
    include_magnetic_curl=False,
    include_udotgradu_mag=True,
    include_coriolis_mag=True,  # New option for Coriolis force
    save_udotgradu_components=False,
    save_coriolis_components=False,  # New option for Coriolis components
    verbose=True,
    r_cutoff=1e-2,
    theta_cutoff=1e-2,
):
    """
    Extract fields with proper handling of singularities at r=0 and poles.
    Now includes Coriolis force magnitude: |z_hat × u|
    """
    
    with h5py.File(hdf5_file, "r") as f:
        # ----------------------------
        # Mesh grids (1D)
        # ----------------------------
        r = f["/mesh/grid_r"][:]                 # (nr,)
        theta = f["/mesh/grid_theta"][:]         # colatitude (ntheta,)
        phi = f["/mesh/grid_phi"][:]             # (nphi,)

        latitude = np.pi / 2.0 - theta
        nr, ntheta, nphi = len(r), len(theta), len(phi)
        
        if verbose:
            print(f"Grid info:")
            print(f"  r: {nr} points, range [{r.min():.3e}, {r.max():.3e}]")
            print(f"  θ: {ntheta} points, range [{theta.min():.3f}, {theta.max():.3f}] rad")
            print(f"  φ: {nphi} points, range [{phi.min():.3f}, {phi.max():.3f}] rad")
            print(f"  r cutoff: {r_cutoff:.3e}")
            print(f"  θ cutoff: {theta_cutoff:.3e} rad")

        # ----------------------------
        # Fields
        # ----------------------------
        u_r     = f["/velocity/velocity_r"][:]
        u_theta = f["/velocity/velocity_theta"][:]
        u_phi   = f["/velocity/velocity_phi"][:]

        B_r     = f["/magnetic/magnetic_r"][:]
        B_theta = f["/magnetic/magnetic_theta"][:]
        B_phi   = f["/magnetic/magnetic_phi"][:]

        T = f["/temperature/temperature"][:]

        # Shape checks
        def _check_shape(name, arr):
            if arr.ndim != 3:
                raise ValueError(f"{name} should be 3D, got shape {arr.shape}")
            if arr.shape != (nr, ntheta, nphi):
                raise ValueError(
                    f"{name} shape mismatch. Expected ({nr}, {ntheta}, {nphi}), got {arr.shape}"
                )

        for nm, a in [
            ("u_r", u_r), ("u_theta", u_theta), ("u_phi", u_phi),
            ("B_r", B_r), ("B_theta", B_theta), ("B_phi", B_phi),
            ("T", T),
        ]:
            _check_shape(nm, a)

        # Time info
        time = _read_scalar(f["/run/time"])
        timestep = _read_scalar(f["/run/timestep"])

        out = dict(
            r=r, theta=theta, phi=phi, latitude=latitude,
            u_r=u_r, u_theta=u_theta, u_phi=u_phi,
            B_r=B_r, B_theta=B_theta, B_phi=B_phi,
            T=T, time=time, timestep=timestep,
        )

        # ----------------------------
        # Curl fields
        # ----------------------------
        if include_curl:
            w_r     = f["/velocity_curl/velocity_curl_r"][:]
            w_theta = f["/velocity_curl/velocity_curl_theta"][:]
            w_phi   = f["/velocity_curl/velocity_curl_phi"][:]

            cosT = np.cos(theta)[None, :, None]
            sinT = np.sin(theta)[None, :, None]
            w_z_theta = w_r * cosT - w_theta * sinT
            out["curl_u_axial"] = w_z_theta

        # Optional: magnetic curl
        if include_magnetic_curl:
            cBr     = f["/magnetic_curl/magnetic_curl_r"][:]
            cBtheta = f["/magnetic_curl/magnetic_curl_theta"][:]
            cBphi   = f["/magnetic_curl/magnetic_curl_phi"][:]
            for nm, a in [("curlB_r", cBr), ("curlB_theta", cBtheta), ("curlB_phi", cBphi)]:
                _check_shape(nm, a)
            out["curlB_r"] = cBr
            out["curlB_theta"] = cBtheta
            out["curlB_phi"] = cBphi

        # ----------------------------
        # | (u grad) u | in spherical coords
        # ----------------------------
        if include_udotgradu_mag:
            # Create grids
            rr = r[:, None, None]                         # (nr,1,1)
            theta_2d = theta[None, :, None]               # (1,ntheta,1)
            sinT = np.sin(theta_2d)
            cosT = np.cos(theta_2d)
            
            # Handle poles
            near_north_pole = (theta_2d < theta_cutoff)
            near_south_pole = (theta_2d > (np.pi - theta_cutoff))
            
            # Safe 1/sinθ
            safe_inv_sin = np.zeros_like(sinT)
            normal_region = ~(near_north_pole | near_south_pole)
            safe_inv_sin[normal_region] = 1.0 / np.abs(sinT[normal_region])
            safe_inv_sin[near_north_pole] = 1.0 / theta_cutoff
            safe_inv_sin[near_south_pole] = 1.0 / theta_cutoff
            safe_inv_sin = np.where(sinT >= 0, safe_inv_sin, -safe_inv_sin)
            
            # Safe cotθ (0 at poles)
            safe_cot = cosT * safe_inv_sin
            safe_cot = np.where(near_north_pole | near_south_pole, 0.0, safe_cot)
            
            # Handle r=0
            safe_inv_r = 1.0 / np.maximum(rr, r_cutoff)
            
            if verbose:
                print(f"\nSingularity handling:")
                print(f"  Near north pole: {np.sum(near_north_pole)} points")
                print(f"  Near south pole: {np.sum(near_south_pole)} points")
                print(f"  Near r=0: {np.sum(rr < r_cutoff)} points")
                print(f"  min sinθ (normal): {np.min(np.abs(sinT[normal_region])):.6e}")
                print(f"  max 1/r: {np.max(safe_inv_r):.6e}")

            # Compute derivatives
            du_r_dr, du_r_dtheta, du_r_dphi = np.gradient(u_r, r, theta, phi, edge_order=2)
            du_theta_dr, du_theta_dtheta, du_theta_dphi = np.gradient(u_theta, r, theta, phi, edge_order=2)
            du_phi_dr, du_phi_dtheta, du_phi_dphi = np.gradient(u_phi, r, theta, phi, edge_order=2)

            # Safe versions
            inv_r_safe = safe_inv_r
            inv_r_sin_safe = safe_inv_r * safe_inv_sin

            # Convective acceleration components
            adv_r = (
                u_r * du_r_dr
                + (u_theta * inv_r_safe) * du_r_dtheta
                + (u_phi * inv_r_sin_safe) * du_r_dphi
                - (u_theta**2 + u_phi**2) * inv_r_safe
            )

            adv_theta = (
                u_r * du_theta_dr
                + (u_theta * inv_r_safe) * du_theta_dtheta
                + (u_phi * inv_r_sin_safe) * du_theta_dphi
                + (u_r * u_theta) * inv_r_safe
                - (u_phi**2) * safe_cot * inv_r_safe
            )

            adv_phi = (
                u_r * du_phi_dr
                + (u_theta * inv_r_safe) * du_phi_dtheta
                + (u_phi * inv_r_sin_safe) * du_phi_dphi
                + (u_r * u_phi) * inv_r_safe
                + (u_theta * u_phi) * safe_cot * inv_r_safe
            )

            adv_mag = np.sqrt(adv_r**2 + adv_theta**2 + adv_phi**2)
            out["u_dot_grad_u_magnitude"] = adv_mag

            if save_udotgradu_components:
                out["u_dot_grad_u_r"] = adv_r
                out["u_dot_grad_u_theta"] = adv_theta
                out["u_dot_grad_u_phi"] = adv_phi

            if verbose:
                print("\nConvective acceleration statistics:")
                print(f"  |adv| mean={np.mean(adv_mag):.6e}, median={np.median(adv_mag):.6e}")

        # ----------------------------
        # Coriolis force magnitude: |z_hat × u|
        # In spherical coords (colatitude θ):
        # z_hat = cosθ e_r - sinθ e_θ
        # z_hat × u = (cosθ u_φ) e_θ + (sinθ u_φ) e_r? Wait, let's derive carefully
        # ----------------------------
        if include_coriolis_mag:
            # Create grids for broadcasting
            theta_2d = theta[None, :, None]  # (1, ntheta, 1)
            cosT = np.cos(theta_2d)
            sinT = np.sin(theta_2d)
            
            # Coriolis force components in spherical coordinates
            # z_hat = cosθ e_r - sinθ e_θ
            # z_hat × u = det[[e_r, e_θ, e_φ], [cosθ, -sinθ, 0], [u_r, u_θ, u_φ]]
            # = ( -sinθ * u_φ ) e_r? Let's compute properly:
            # e_r component: (z_hat_θ * u_φ - z_hat_φ * u_θ) = (-sinθ * u_φ - 0 * u_θ) = -sinθ * u_φ
            # e_θ component: (z_hat_φ * u_r - z_hat_r * u_φ) = (0 * u_r - cosθ * u_φ) = -cosθ * u_φ
            # e_φ component: (z_hat_r * u_θ - z_hat_θ * u_r) = (cosθ * u_θ - (-sinθ) * u_r) = cosθ * u_θ + sinθ * u_r
            
            coriolis_r = -sinT * u_phi
            coriolis_theta = -cosT * u_phi
            coriolis_phi = cosT * u_theta + sinT * u_r
            
            # Magnitude of Coriolis force
            coriolis_mag = np.sqrt(coriolis_r**2 + coriolis_theta**2 + coriolis_phi**2)
            
            # Alternative simpler expression: |z_hat × u| = sqrt(u² - (z_hat·u)²)
            # z_hat·u = cosθ * u_r - sinθ * u_theta
            z_dot_u = cosT * u_r - sinT * u_theta
            u_mag_sq = u_r**2 + u_theta**2 + u_phi**2
            coriolis_mag_alt = np.sqrt(np.maximum(u_mag_sq - z_dot_u**2, 0))  # max to avoid negative due to rounding
            
            # Check consistency (should be identical up to roundoff)
            max_diff = np.max(np.abs(coriolis_mag - coriolis_mag_alt))
            
            out["coriolis_magnitude"] = coriolis_mag
            
            if save_coriolis_components:
                out["coriolis_r"] = coriolis_r
                out["coriolis_theta"] = coriolis_theta
                out["coriolis_phi"] = coriolis_phi
                out["z_dot_u"] = z_dot_u
            
            if verbose:
                print("\nCoriolis force statistics:")
                print(f"  coriolis_r: mean={np.mean(np.abs(coriolis_r)):.6e}")
                print(f"  coriolis_theta: mean={np.mean(np.abs(coriolis_theta)):.6e}")
                print(f"  coriolis_phi: mean={np.mean(np.abs(coriolis_phi)):.6e}")
                print(f"  |z_hat × u|: mean={np.mean(coriolis_mag):.6e}, median={np.median(coriolis_mag):.6e}")
                print(f"  max diff between two methods: {max_diff:.6e}")
                
                # Compare with velocity magnitude
                u_mag = np.sqrt(u_mag_sq)
                print(f"  |u|: mean={np.mean(u_mag):.6e}")
                print(f"  |z_hat × u| / |u|: mean={np.mean(coriolis_mag/u_mag):.3f}")

    np.savez_compressed(output_file, **out)
    print(f"\nResults saved to: {output_file}")
    
    return out


def estimate_rms(field, name="field", verbose=True):
    """
    Estimate RMS value using robust statistics (percentiles).
    Returns dictionary with various estimates.
    """
    # Flatten the field
    data = field.flatten()
    
    # Basic statistics
    mean_val = np.mean(data)
    median_val = np.median(data)
    rms_val = np.sqrt(np.mean(data**2))
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    perc_vals = {p: np.percentile(data, p) for p in percentiles}
    
    if verbose:
        print(f"\n{name} RMS estimates:")
        print(f"  Mean: {mean_val:.6e}")
        print(f"  Median (50%): {median_val:.6e}")
        print(f"  75%: {perc_vals[75]:.6e}")
        print(f"  90%: {perc_vals[90]:.6e}")
        print(f"  95%: {perc_vals[95]:.6e}")
        print(f"  99%: {perc_vals[99]:.6e}")
        print(f"  True RMS: {rms_val:.6e}")
    
    return {
        'mean': mean_val,
        'median': median_val,
        'rms': rms_val,
        'percentiles': perc_vals
    }


# Example usage
if __name__ == "__main__":
    result = extract_fields_with_curl_fixed(
        "visState0000.hdf5",
        "vis_fields_0000.npz",
        include_curl=True,
        include_magnetic_curl=False,
        include_udotgradu_mag=True,
        include_coriolis_mag=True,
        save_udotgradu_components=False,
        save_coriolis_components=False,
        verbose=True,
        r_cutoff=1e-2,
        theta_cutoff=1e-2,
    )
    
    print("\n" + "="*60)
    print("RMS VALUE ESTIMATES")
    print("="*60)
    
    # Estimate RMS for convective acceleration
    if "u_dot_grad_u_magnitude" in result:
        adv_stats = estimate_rms(result["u_dot_grad_u_magnitude"], "Convective acceleration |(u·∇)u|")
    
    # Estimate RMS for Coriolis force
    if "coriolis_magnitude" in result:
        coriolis_stats = estimate_rms(result["coriolis_magnitude"], "Coriolis force |ẑ × u|")
        
        # Compare with velocity magnitude
        u_mag = np.sqrt(result["u_r"]**2 + result["u_theta"]**2 + result["u_phi"]**2)
        u_stats = estimate_rms(u_mag, "Velocity magnitude |u|")
        
        print(f"\nRatio |ẑ × u| / |u|:")
        print(f"  Mean ratio: {coriolis_stats['mean']/u_stats['mean']:.3f}")
        print(f"  Median ratio: {coriolis_stats['median']/u_stats['median']:.3f}")
        
        # In dimensionless units, Coriolis force magnitude should be comparable to |u|
        # since the prefactor is 1 (for unit rotation rate)
        print(f"\nNote: In dimensionless units with rotation rate Ω=1,")
        print(f"      Coriolis force magnitude should be comparable to |u|.")
        print(f"      Here it's about {coriolis_stats['median']/u_stats['median']:.2f} times |u|.")
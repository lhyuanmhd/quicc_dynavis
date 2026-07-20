#!/usr/bin/env python3
"""
extract_forces.py

Extract all forces from QUICC HDF5 output with proper volume-weighted RMS calculation
using Gauss-Legendre quadrature.

Forces:
- Inertia: (u·∇)u (inertia, no coefficient)
- Coriolis: ẑ × u  
- Viscous: E ∇²u
- Lorentz: (∇×B)×B
- Buoyancy: q Ra T r r̂

Note: When comparing with Coriolis, true inertia should be multiplied by E_eta = 1e-9
"""

import h5py
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore')

# Constants
E_ETA = 1e-9  # Magnetic Ekman number for inertia scaling

# ============================================================================
# Helper functions for HDF5 reading
# ============================================================================

def _read_scalar(ds):
    """Read scalar dataset from HDF5."""
    val = ds[()]
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    if isinstance(val, (np.generic,)):
        return val.item()
    return val

def _read_string_dataset(ds):
    """Read string dataset from HDF5."""
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

# ============================================================================
# Grid sorting and preparation
# ============================================================================

def sort_grid_and_fields(r, theta, phi, *fields):
    """
    Ensure grids are monotonically increasing and sort fields accordingly.
    
    Returns:
        r_sorted, theta_sorted, phi_sorted, and sorted fields
    """
    # Sort r
    r_idx = np.argsort(r)
    r_sorted = r[r_idx]
    
    # Sort theta
    theta_idx = np.argsort(theta)
    theta_sorted = theta[theta_idx]
    
    # Sort phi
    phi_idx = np.argsort(phi)
    phi_sorted = phi[phi_idx]
    
    # Sort all fields
    sorted_fields = []
    for field in fields:
        field_sorted = field[r_idx, :, :]
        field_sorted = field_sorted[:, theta_idx, :]
        field_sorted = field_sorted[:, :, phi_idx]
        sorted_fields.append(field_sorted)
    
    return (r_sorted, theta_sorted, phi_sorted, *sorted_fields)

# ============================================================================
# Spherical operators with singularity handling
# ============================================================================

def _safe_1_over_sin(theta, theta_cutoff=1e-2):
    """Compute 1/sinθ safely near poles."""
    sinT = np.sin(theta)
    near_north = (theta < theta_cutoff)
    near_south = (theta > (np.pi - theta_cutoff))
    
    result = np.zeros_like(theta)
    normal = ~(near_north | near_south)
    
    result[normal] = 1.0 / np.maximum(np.abs(sinT[normal]), 1e-10)
    result[near_north] = 1.0 / theta_cutoff
    result[near_south] = 1.0 / theta_cutoff
    
    return np.where(sinT >= 0, result, -result)

def _safe_cot(theta, theta_cutoff=1e-2):
    """Compute cotθ safely, returning 0 near poles."""
    sinT = np.sin(theta)
    near_pole = (theta < theta_cutoff) | (theta > (np.pi - theta_cutoff))
    
    cot = np.cos(theta) / np.maximum(np.abs(sinT), 1e-10)
    return np.where(near_pole, 0.0, cot)

def _vector_laplacian_curlcurl(v_r, v_theta, v_phi, r, theta, phi, 
                               r_cutoff=1e-2, theta_cutoff=1e-2):
    """
    Compute vector Laplacian using ∇²v = -∇×(∇×v) for incompressible flow.
    More stable than direct Laplacian.
    """
    nr, ntheta, nphi = v_r.shape
    
    # Create grids
    rr = r[:, None, None]
    theta_2d = theta[None, :, None]
    sinT = np.sin(theta_2d)
    
    # Safe inverses
    safe_inv_sin = _safe_1_over_sin(theta_2d, theta_cutoff)
    safe_inv_r = 1.0 / np.maximum(rr, r_cutoff)
    
    # ========== Step 1: Compute curl of v = ∇ × v ==========
    
    # Derivatives needed for curl
    dv_theta_dphi = np.gradient(v_theta, phi, axis=2, edge_order=2)
    dv_phi_dtheta = np.gradient(v_phi, theta, axis=1, edge_order=2)
    dv_r_dphi = np.gradient(v_r, phi, axis=2, edge_order=2)
    dv_r_dtheta = np.gradient(v_r, theta, axis=1, edge_order=2)
    
    d_rv_theta_dr = np.gradient(rr * v_theta, r, axis=0, edge_order=2)
    d_rv_phi_dr = np.gradient(rr * v_phi, r, axis=0, edge_order=2)
    
    # Curl components
    curl_r = safe_inv_r * safe_inv_sin * (
        np.gradient(sinT * v_phi, theta, axis=1, edge_order=2) - dv_theta_dphi
    )
    curl_theta = safe_inv_r * safe_inv_sin * (
        dv_r_dphi - sinT * d_rv_phi_dr
    )
    curl_phi = safe_inv_r * (
        d_rv_theta_dr - dv_r_dtheta
    )
    
    # ========== Step 2: Compute curl of curl = ∇ × (∇ × v) ==========
    
    # Derivatives of curl
    d_curl_r_dtheta = np.gradient(curl_r, theta, axis=1, edge_order=2)
    d_curl_r_dphi = np.gradient(curl_r, phi, axis=2, edge_order=2)
    d_curl_theta_dphi = np.gradient(curl_theta, phi, axis=2, edge_order=2)
    
    d_r_curl_theta_dr = np.gradient(rr * curl_theta, r, axis=0, edge_order=2)
    d_r_curl_phi_dr = np.gradient(rr * curl_phi, r, axis=0, edge_order=2)
    
    # Curl of curl components
    curl_curl_r = safe_inv_r * safe_inv_sin * (
        np.gradient(sinT * curl_phi, theta, axis=1, edge_order=2) - d_curl_theta_dphi
    )
    curl_curl_theta = safe_inv_r * safe_inv_sin * (
        d_curl_r_dphi - sinT * d_r_curl_phi_dr
    )
    curl_curl_phi = safe_inv_r * (
        d_r_curl_theta_dr - d_curl_r_dtheta
    )
    
    # Vector Laplacian: ∇²v = -∇ × (∇ × v) for incompressible flow
    lap_v_r = -curl_curl_r
    lap_v_theta = -curl_curl_theta
    lap_v_phi = -curl_curl_phi
    
    return lap_v_r, lap_v_theta, lap_v_phi

# ============================================================================
# Gauss-Legendre quadrature for volume integrals
# ============================================================================

class GaussLegendreIntegrator:
    """
    High-order integration in spherical coordinates using Gauss-Legendre quadrature.
    """
    
    def __init__(self, r, theta, phi, 
                 nr_gauss=None, ntheta_gauss=None, nphi_gauss=None):
        """
        Initialize integrator with Gauss nodes and weights.
        
        Parameters:
        -----------
        r, theta, phi : 1D arrays - original grid coordinates (must be monotonic)
        nr_gauss, ntheta_gauss, nphi_gauss : int - number of Gauss nodes
                                              (default: same as original)
        """
        # Ensure coordinates are monotonic
        self.r_orig = np.asarray(r)
        self.theta_orig = np.asarray(theta)
        self.phi_orig = np.asarray(phi)
        
        # Check monotonicity
        for name, arr in [('r', self.r_orig), ('theta', self.theta_orig), ('phi', self.phi_orig)]:
            if not np.all(np.diff(arr) > 0):
                print(f"Warning: {name} grid not strictly increasing. Sorting...")
                # This should not happen if sort_grid_and_fields was used
        
        # Set Gauss node counts (use original if not specified)
        self.nr = nr_gauss if nr_gauss is not None else len(r)
        self.ntheta = ntheta_gauss if ntheta_gauss is not None else len(theta)
        self.nphi = nphi_gauss if nphi_gauss is not None else len(phi)
        
        # Generate Gauss nodes and weights
        self._generate_gauss_nodes()
        
    def _generate_gauss_nodes(self):
        """Generate Gauss-Legendre nodes and weights for each direction."""
        # r direction
        xr, wr = leggauss(self.nr)
        r_min, r_max = self.r_orig.min(), self.r_orig.max()
        self.r_nodes = 0.5*(r_max - r_min)*xr + 0.5*(r_max + r_min)
        self.r_weights = 0.5*(r_max - r_min)*wr
        
        # theta direction
        xt, wt = leggauss(self.ntheta)
        theta_min, theta_max = self.theta_orig.min(), self.theta_orig.max()
        self.theta_nodes = 0.5*(theta_max - theta_min)*xt + 0.5*(theta_max + theta_min)
        self.theta_weights = 0.5*(theta_max - theta_min)*wt
        
        # phi direction (periodic, full 2π)
        xp, wp = leggauss(self.nphi)
        phi_min, phi_max = 0, 2*np.pi
        self.phi_nodes = 0.5*(phi_max - phi_min)*xp + 0.5*(phi_max + phi_min)
        self.phi_weights = 0.5*(phi_max - phi_min)*wp
        
        # Precompute sinθ at theta nodes
        self.sin_theta_nodes = np.sin(self.theta_nodes)
        
        # Create meshgrid for all nodes (sparse for memory efficiency)
        self.r_mesh, self.theta_mesh, self.phi_mesh = np.meshgrid(
            self.r_nodes, self.theta_nodes, self.phi_nodes,
            indexing='ij', sparse=True
        )
        
        # Compute weight tensor product
        r2_weights = self.r_nodes.reshape(-1, 1, 1) ** 2
        sin_theta_weights = self.sin_theta_nodes.reshape(1, -1, 1)
        
        self.vol_weights = (r2_weights * sin_theta_weights * 
                           self.r_weights.reshape(-1, 1, 1) *
                           self.theta_weights.reshape(1, -1, 1) *
                           self.phi_weights.reshape(1, 1, -1))
        
    def interpolate_to_gauss(self, field):
        """
        Interpolate field from original grid to Gauss nodes.
        Handles periodic boundary in phi.
        """
        # Extend phi for periodicity
        phi_extended = np.concatenate([self.phi_orig, [self.phi_orig[0] + 2*np.pi]])
        field_extended = np.concatenate([field, field[:, :, :1]], axis=2)
        
        # Ensure grids are strictly increasing for interpolator
        # (they should be after sorting)
        r_inc = np.ascontiguousarray(self.r_orig)
        theta_inc = np.ascontiguousarray(self.theta_orig)
        phi_inc = np.ascontiguousarray(phi_extended)
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (r_inc, theta_inc, phi_inc),
            field_extended,
            bounds_error=False,
            method='linear',
            fill_value=None
        )
        
        # Prepare points for interpolation
        # Need to create full grid for all points
        r_full, theta_full, phi_full = np.meshgrid(
            self.r_nodes, self.theta_nodes, self.phi_nodes,
            indexing='ij', sparse=False
        )
        
        # Wrap phi to [0, 2π) for interpolation
        phi_flat = phi_full.ravel() % (2*np.pi)
        r_flat = r_full.ravel()
        theta_flat = theta_full.ravel()
        
        points = np.column_stack([r_flat, theta_flat, phi_flat])
        
        # Interpolate and reshape
        field_gauss = interpolator(points).reshape(r_full.shape)
        
        return field_gauss
    
    def integrate(self, field, squared=True):
        """
        Compute volume integral ∫ f dV or ∫ f² dV.
        
        Parameters:
        -----------
        field : 3D array on original grid
        squared : bool - if True, integrate f², else integrate f
        
        Returns:
        --------
        integral : float - volume integral
        """
        # Interpolate to Gauss nodes
        f_gauss = self.interpolate_to_gauss(field)
        
        if squared:
            integrand = f_gauss ** 2
        else:
            integrand = f_gauss
        
        return np.sum(self.vol_weights * integrand)
    
    def compute_rms(self, field):
        """
        Compute properly volume-weighted RMS: √(∫ f² dV / ∫ dV)
        """
        integral_f2 = self.integrate(field, squared=True)
        volume = np.sum(self.vol_weights)  # ∫ dV
        
        rms = np.sqrt(integral_f2 / volume)
        return rms, integral_f2, volume
    
    def compute_force_balance(self, forces_dict, phys_params=None, scale_inertia=True):
        """
        Compute RMS for multiple forces and compare.
        
        Parameters:
        -----------
        forces_dict : dict - {name: field_array}
        phys_params : dict - physical parameters to display
        scale_inertia : bool - if True, multiply inertia by E_ETA for comparison
        
        Returns:
        --------
        results : dict - {name: {'rms': rms, 'integral': int, 'volume': vol,
                                 'scaled_rms': scaled_rms (if applicable)}}
        """
        results = {}
        volume = np.sum(self.vol_weights)
        
        print("\n" + "="*80)
        print("FORCE RMS (Gauss-Legendre quadrature)")
        print("="*80)
        print(f"Total volume: {volume:.6e}")
        
        # Display physical parameters if available
        if phys_params:
            print("-"*80)
            print("Physical Parameters:")
            for key, value in phys_params.items():
                if key in ['ekman', 'rayleigh', 'roberts', 'E_eta']:
                    print(f"  {key}: {value:.6e}")
                elif key in ['velocity_str', 'temperature_str', 'magnetic_str']:
                    print(f"  {key}: {value}")
        
        print(f"E_eta for inertia scaling: {E_ETA:.1e}")
        print("-"*80)
        print(f"{'Force':20s} {'RMS':15s} {'Scaled RMS':15s} {'∫|F|² dV':20s}")
        print("-"*80)
        
        for name, field in forces_dict.items():
            try:
                rms, integral, _ = self.compute_rms(field)
                
                # Scale inertia if requested
                if scale_inertia and name == 'Inertia':
                    scaled_rms = rms * E_ETA
                    print(f"{name:20s} {rms:15.6e} {scaled_rms:15.6e} {integral:20.6e}")
                    results[name] = {
                        'rms': rms, 
                        'scaled_rms': scaled_rms,
                        'integral': integral, 
                        'volume': volume
                    }
                else:
                    print(f"{name:20s} {rms:15.6e} {'-':15s} {integral:20.6e}")
                    results[name] = {
                        'rms': rms,
                        'integral': integral, 
                        'volume': volume
                    }
            except Exception as e:
                print(f"{name:20s} ERROR: {e}")
        
        # Compare with Coriolis
        if 'Coriolis' in results and len(results) > 1:
            print("-"*80)
            print("Ratios relative to Coriolis:")
            coriolis_rms = results['Coriolis']['rms']
            
            for name, res in results.items():
                if name != 'Coriolis':
                    if name == 'Inertia' and scale_inertia:
                        # Use scaled inertia for comparison
                        ratio = res['scaled_rms'] / coriolis_rms if coriolis_rms != 0 else float('inf')
                        print(f"  {name:20s} (scaled by E_eta) / Coriolis = {ratio:.6f}")
                    else:
                        ratio = res['rms'] / coriolis_rms if coriolis_rms != 0 else float('inf')
                        print(f"  {name:20s} / Coriolis = {ratio:.6f}")
        
        return results

# ============================================================================
# Main extraction function
# ============================================================================

def extract_forces(hdf5_file, output_file="vis_fields_forces.npz",
                   include_curl=True,
                   include_inertia=True,
                   include_coriolis=True,
                   include_viscous=True,
                   include_lorentz=True,
                   include_buoyancy=True,
                   verbose=True,
                   r_cutoff=1e-2,
                   theta_cutoff=1e-2):
    """
    Extract all forces from HDF5 file.
    """
    
    with h5py.File(hdf5_file, "r") as f:
        # ====================================================================
        # Read grid
        # ====================================================================
        r = f["/mesh/grid_r"][:]
        theta = f["/mesh/grid_theta"][:]  # colatitude
        phi = f["/mesh/grid_phi"][:]
        
        # ====================================================================
        # Read fields
        # ====================================================================
        u_r = f["/velocity/velocity_r"][:]
        u_theta = f["/velocity/velocity_theta"][:]
        u_phi = f["/velocity/velocity_phi"][:]
        
        B_r = f["/magnetic/magnetic_r"][:]
        B_theta = f["/magnetic/magnetic_theta"][:]
        B_phi = f["/magnetic/magnetic_phi"][:]
        
        T = f["/temperature/temperature"][:]

        print(f"  T: mean={np.mean(T):.3e}, median={np.median(T):.3e}")
        

        
        # ====================================================================
        # Sort grids and fields to ensure monotonicity
        # ====================================================================
        r, theta, phi, u_r, u_theta, u_phi, B_r, B_theta, B_phi, T = sort_grid_and_fields(
            r, theta, phi, u_r, u_theta, u_phi, B_r, B_theta, B_phi, T
        )
        
        latitude = np.pi/2 - theta
        nr, ntheta, nphi = len(r), len(theta), len(phi)
        
        if verbose:
            print("\n" + "="*80)
            print("GRID INFORMATION")
            print("="*80)
            print(f"r:     {nr} points, range [{r.min():.3e}, {r.max():.3e}]")
            print(f"θ:     {ntheta} points, range [{theta.min():.3f}, {theta.max():.3f}]")
            print(f"φ:     {nphi} points, range [{phi.min():.3f}, {phi.max():.3f}]")
        
        
        # ====================================================================
        # ZONAL FLOW: phi-average of u_phi (ADD THIS SECTION)
        # ====================================================================
        # Compute zonal flow (average over phi direction)
        u_phi_zonal = np.mean(u_phi, axis=2)  # Shape (nr, ntheta)
        u_phi_zonal_3d = u_phi_zonal[:, :, np.newaxis]  # Shape (nr, ntheta, 1) for broadcasting

        if verbose:
            print("\n" + "="*80)
            print("ZONAL FLOW COMPUTED")
            print("="*80)
            print(f"u_phi_zonal shape: {u_phi_zonal.shape}")
            print(f"u_phi_zonal range: [{np.min(u_phi_zonal):.3e}, {np.max(u_phi_zonal):.3e}]")
            print(f"u_phi_zonal mean: {np.mean(u_phi_zonal):.3e}")
                
        # Check shapes
        expected_shape = (nr, ntheta, nphi)
        for name, arr in [('u_r', u_r), ('u_theta', u_theta), ('u_phi', u_phi),
                          ('B_r', B_r), ('B_theta', B_theta), ('B_phi', B_phi),
                          ('T', T)]:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} shape {arr.shape} != {expected_shape}")
        
        # ====================================================================
        # Read physical parameters
        # ====================================================================
        phys = {}
        for key in ["ekman", "inertia", "rayleigh", "roberts"]:
            path = f"/physical/{key}"
            if path in f:
                phys[key] = _read_scalar(f[path])
        
        for key in ["velocity", "temperature", "magnetic"]:
            path = f"/physical/{key}"
            if path in f:
                phys[f"{key}_str"] = _read_string_dataset(f[path])
        
        if verbose:
            print("\n" + "="*80)
            print("PHYSICAL PARAMETERS")
            print("="*80)
            for k, v in phys.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6e}")
                else:
                    print(f"  {k}: {v}")
            print(f"  E_eta (for inertia scaling): {E_ETA:.1e}")
        
        # Time info
        time = _read_scalar(f["/run/time"]) if "/run/time" in f else np.nan
        timestep = _read_scalar(f["/run/timestep"]) if "/run/timestep" in f else np.nan
        
        # ====================================================================
        # Precompute common quantities for force calculations
        # ====================================================================
        rr = r[:, None, None]
        theta_2d = theta[None, :, None]
        sinT = np.sin(theta_2d)
        cosT = np.cos(theta_2d)
        
        # Safe versions for singularities
        safe_inv_sin = _safe_1_over_sin(theta_2d, theta_cutoff)
        safe_cot = _safe_cot(theta_2d, theta_cutoff)
        safe_inv_r = 1.0 / np.maximum(rr, r_cutoff)
        
        # ====================================================================
        # Initialize output dictionary
        # ====================================================================
        out = {
            'r': r, 'theta': theta, 'phi': phi, 'latitude': latitude,
            'u_r': u_r, 'u_theta': u_theta, 'u_phi': u_phi,
            'B_r': B_r, 'B_theta': B_theta, 'B_phi': B_phi,
            'T': T,
            'time': time, 'timestep': timestep,
            'E_eta': E_ETA,
            #'u_phi_zonal': u_phi_zonal,      # ADD THIS
            'u_phi_zonal_3d': u_phi_zonal_3d, # ADD THIS
            **phys
        }
        
        # ----------------------------
        # Curl fields (vorticity)
        # ----------------------------
        if include_curl:
            w_r     = f["/velocity_curl/velocity_curl_r"][:]
            w_theta = f["/velocity_curl/velocity_curl_theta"][:]
            w_phi   = f["/velocity_curl/velocity_curl_phi"][:]
            
            
            # Store full vorticity components
            #out["curl_u_r"] = w_r
            #out["curl_u_theta"] = w_theta
            #out["curl_u_phi"] = w_phi
            
            # Calculate vorticity magnitude
            #curl_u_mag = np.sqrt(w_r**2 + w_theta**2 + w_phi**2)
            #out["curl_u_magnitude"] = curl_u_mag
            
            # Calculate axial component of vorticity (along rotation axis)
            # z_hat = cosθ e_r - sinθ e_θ
            # ω_z = ω · z_hat = ω_r * cosθ - ω_θ * sinθ
            
            cosT = np.cos(theta)[None, :, None]
            sinT = np.sin(theta)[None, :, None]
            w_z = w_r * cosT - w_theta * sinT
            out["curl_u_axial"] = w_z  # This is the component along the rotation axis
            
            if verbose:
                 print("\nVorticity statistics:")
            #     print(f"  |∇×u| mean={np.mean(curl_u_mag):.6e}, median={np.median(curl_u_mag):.6e}")
                 print(f"  ω_z (axial) mean={np.mean(np.abs(w_z)):.6e}")
                
            #     # Compare with enstrophy
            #     enstrophy = np.mean(curl_u_mag**2)
            #     print(f"  Enstrophy (mean ω²): {enstrophy:.6e}")


        # ====================================================================
        # 1. Inertia force: |(u·∇)u| (inertia, no coefficient)
        # ====================================================================
        if include_inertia:
            if verbose:
                print("\n" + "="*80)
                print("COMPUTING INERTIA FORCE (no coefficient)")
                print("="*80)
            
            # Derivatives
            du_r_dr, du_r_dtheta, du_r_dphi = np.gradient(u_r, r, theta, phi, edge_order=2)
            du_theta_dr, du_theta_dtheta, du_theta_dphi = np.gradient(u_theta, r, theta, phi, edge_order=2)
            du_phi_dr, du_phi_dtheta, du_phi_dphi = np.gradient(u_phi, r, theta, phi, edge_order=2)
            
            # Components
            adv_r = (u_r * du_r_dr +
                    (u_theta * safe_inv_r) * du_r_dtheta +
                    (u_phi * safe_inv_r * safe_inv_sin) * du_r_dphi -
                    (u_theta**2 + u_phi**2) * safe_inv_r)
            
            adv_theta = (u_r * du_theta_dr +
                        (u_theta * safe_inv_r) * du_theta_dtheta +
                        (u_phi * safe_inv_r * safe_inv_sin) * du_theta_dphi +
                        (u_r * u_theta) * safe_inv_r -
                        (u_phi**2) * safe_cot * safe_inv_r)
            
            adv_phi = (u_r * du_phi_dr +
                      (u_theta * safe_inv_r) * du_phi_dtheta +
                      (u_phi * safe_inv_r * safe_inv_sin) * du_phi_dphi +
                      (u_r * u_phi) * safe_inv_r +
                      (u_theta * u_phi) * safe_cot * safe_inv_r)
            
            inertia_mag = np.sqrt(adv_r**2 + adv_theta**2 + adv_phi**2)
            out['inertia_magnitude'] = inertia_mag
            out['u_dot_grad_u_magnitude'] = inertia_mag  # backward compat
            
            if verbose:
                print(f"  True inertia RMS (unscaled): {np.mean(inertia_mag):.3e}")
                print(f"  Scaled by E_eta={E_ETA:.1e}: {np.mean(inertia_mag)*E_ETA:.3e}")
        
        # ====================================================================
        # 2. Coriolis force: |ẑ × u|
        # ====================================================================
        if include_coriolis:
            if verbose:
                print("\n" + "="*80)
                print("COMPUTING CORIOLIS FORCE")
                print("="*80)
            
            coriolis_r = -sinT * u_phi
            coriolis_theta = -cosT * u_phi
            coriolis_phi = cosT * u_theta + sinT * u_r
            
            coriolis_mag = np.sqrt(coriolis_r**2 + coriolis_theta**2 + coriolis_phi**2)
            out['coriolis_magnitude'] = coriolis_mag
        
        # ====================================================================
        # 3. Viscous force: |E ∇²u|
        # ====================================================================
        if include_viscous:
            if verbose:
                print("\n" + "="*80)
                print("COMPUTING VISCOUS FORCE")
                print("="*80)
            
            Ekman = phys.get('ekman', 1.0)
            
            # Vector Laplacian using curl-curl formulation
            lap_u_r, lap_u_theta, lap_u_phi = _vector_laplacian_curlcurl(
                u_r, u_theta, u_phi, r, theta, phi, r_cutoff, theta_cutoff
            )
            
            viscous_r = Ekman * lap_u_r
            viscous_theta = Ekman * lap_u_theta
            viscous_phi = Ekman * lap_u_phi
            
            viscous_mag = np.sqrt(viscous_r**2 + viscous_theta**2 + viscous_phi**2)
            out['viscous_magnitude'] = viscous_mag
            
            if verbose:
                u_mag = np.sqrt(u_r**2 + u_theta**2 + u_phi**2)
                lap_mag = np.sqrt(lap_u_r**2 + lap_u_theta**2 + lap_u_phi**2)
                print(f"  |u|:          {np.median(u_mag):.3e}")
                print(f"  |∇²u|:        {np.median(lap_mag):.3e}")
                print(f"  Implied scale: √(|u|/|∇²u|) = {np.sqrt(np.median(u_mag)/np.median(lap_mag)):.3e}")
        
        # ====================================================================
        # 4. Lorentz force: |(∇×B)×B|
        # ====================================================================
        if include_lorentz:
            if verbose:
                print("\n" + "="*80)
                print("COMPUTING LORENTZ FORCE")
                print("="*80)
            
            # Try to use pre-computed curl if available
            if "/magnetic_curl/magnetic_curl_r" in f:
                curlB_r = f["/magnetic_curl/magnetic_curl_r"][:]
                curlB_theta = f["/magnetic_curl/magnetic_curl_theta"][:]
                curlB_phi = f["/magnetic_curl/magnetic_curl_phi"][:]
                
                # Sort curl fields too
                _, _, _, curlB_r, curlB_theta, curlB_phi = sort_grid_and_fields(
                    r, theta, phi, curlB_r, curlB_theta, curlB_phi
                )
            else:
                # Compute curl of B
                dB_r_dr, dB_r_dtheta, dB_r_dphi = np.gradient(B_r, r, theta, phi, edge_order=2)
                dB_theta_dr, dB_theta_dtheta, dB_theta_dphi = np.gradient(B_theta, r, theta, phi, edge_order=2)
                dB_phi_dr, dB_phi_dtheta, dB_phi_dphi = np.gradient(B_phi, r, theta, phi, edge_order=2)
                
                curlB_r = (safe_inv_r * safe_inv_sin) * (
                    np.gradient(sinT * B_phi, theta, axis=1, edge_order=2) - dB_theta_dphi
                )
                curlB_theta = safe_inv_r * (
                    safe_inv_sin * dB_r_dphi - np.gradient(rr * B_phi, r, axis=0, edge_order=2)
                )
                curlB_phi = safe_inv_r * (
                    np.gradient(rr * B_theta, r, axis=0, edge_order=2) - dB_r_dtheta
                )
            
            # Lorentz force = J × B
            lorentz_r = curlB_theta * B_phi - curlB_phi * B_theta
            lorentz_theta = curlB_phi * B_r - curlB_r * B_phi
            lorentz_phi = curlB_r * B_theta - curlB_theta * B_r
            
            lorentz_mag = np.sqrt(lorentz_r**2 + lorentz_theta**2 + lorentz_phi**2)
            out['lorentz_magnitude'] = lorentz_mag
        
        # ====================================================================
        # 5. Buoyancy force: |q Ra T r|
        # ====================================================================
        if include_buoyancy:
            if verbose:
                print("\n" + "="*80)
                print("COMPUTING BUOYANCY FORCE")
                print("="*80)
            
            q = phys.get('roberts')
            Ra = phys.get('rayleigh')
            buoy_scale = q * Ra
            
            # Buoyancy = q * Ra * T * r (radial only)
            buoyancy_r = buoy_scale * T * rr.squeeze()[:, None, None]
            buoyancy_mag = np.abs(buoyancy_r)
            
            out['buoyancy_magnitude'] = buoyancy_mag
            
            if verbose:
                print(f"  q*Ra = {buoy_scale:.3e}")
                print(f"  T: mean={np.mean(T):.3e}, median={np.median(T):.3e}")
        
        # ====================================================================
        # Save results
        # ====================================================================
        np.savez_compressed(output_file, **out)
        if verbose:
            print(f"\nResults saved to: {output_file}")
        
        return out, phys  

# ============================================================================
# RMS analysis with Gauss-Legendre quadrature
# ============================================================================

def analyze_forces_rms(npz_file, phys_params=None, nr_gauss=None, ntheta_gauss=None, nphi_gauss=None,
                       scale_inertia=True):
    """
    Analyze force RMS using proper Gauss-Legendre quadrature.
    
    Parameters:
    -----------
    npz_file : str - input NPZ file
    phys_params : dict - physical parameters to display
    nr_gauss, ntheta_gauss, nphi_gauss : int - Gauss node counts
    scale_inertia : bool - if True, multiply inertia by E_ETA for comparison
    """
    # Load data
    data = np.load(npz_file)
    
    r = data['r']
    theta = data['theta']
    phi = data['phi']
    
    # Create integrator
    integrator = GaussLegendreIntegrator(
        r, theta, phi,
        nr_gauss=nr_gauss,
        ntheta_gauss=ntheta_gauss,
        nphi_gauss=nphi_gauss
    )
    
    # Collect forces
    forces = {}
    force_keys = [
        ('inertia_magnitude', 'Inertia'),
        ('coriolis_magnitude', 'Coriolis'),
        ('viscous_magnitude', 'Viscous'),
        ('lorentz_magnitude', 'Lorentz'),
        ('buoyancy_magnitude', 'Buoyancy'),
    ]
    
    for key, name in force_keys:
        if key in data:
            forces[name] = data[key]
    
    # Compute RMS with scaling and display physical parameters
    results = integrator.compute_force_balance(forces, phys_params=phys_params, scale_inertia=scale_inertia)
    
    return results, integrator

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract forces from QUICC HDF5")
    parser.add_argument("hdf5_file", help="Input HDF5 file (e.g., visState0000.hdf5)")
    parser.add_argument("--output", "-o", default="vis_fields_forces.npz",
                       help="Output NPZ file")
    parser.add_argument("--no-inertia", action="store_false", dest="inertia",
                       help="Skip inertia force")
    parser.add_argument("--no-coriolis", action="store_false", dest="coriolis",
                       help="Skip Coriolis force")
    parser.add_argument("--no-viscous", action="store_false", dest="viscous",
                       help="Skip viscous force")
    parser.add_argument("--no-lorentz", action="store_false", dest="lorentz",
                       help="Skip Lorentz force")
    parser.add_argument("--no-buoyancy", action="store_false", dest="buoyancy",
                       help="Skip buoyancy force")
    parser.add_argument("--nr-gauss", type=int, default=None,
                       help="Number of Gauss nodes in r (default: same as grid)")
    parser.add_argument("--ntheta-gauss", type=int, default=None,
                       help="Number of Gauss nodes in theta")
    parser.add_argument("--nphi-gauss", type=int, default=None,
                       help="Number of Gauss nodes in phi")
    parser.add_argument("--no-scale-inertia", action="store_false", dest="scale_inertia",
                       help="Do not scale inertia by E_eta in comparison")
    
    args = parser.parse_args()
    
    # Extract forces and get physical parameters
    result, phys_params = extract_forces(
        args.hdf5_file,
        output_file=args.output,
        include_curl=True,
        include_inertia=args.inertia,
        include_coriolis=args.coriolis,
        include_viscous=args.viscous,
        include_lorentz=args.lorentz,
        include_buoyancy=args.buoyancy,
        verbose=True
    )
    
    # Analyze RMS with Gauss quadrature
    print("\n" + "="*80)
    print("RMS ANALYSIS WITH GAUSS-LEGENDRE QUADRATURE")
    print("="*80)
    
    results, integrator = analyze_forces_rms(
        args.output,
        phys_params=phys_params,  
        nr_gauss=args.nr_gauss,
        ntheta_gauss=args.ntheta_gauss,
        nphi_gauss=args.nphi_gauss,
        scale_inertia=args.scale_inertia
    )
    
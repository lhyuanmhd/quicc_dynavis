import h5py
import numpy as np

def _read_scalar(ds):
    """Read an HDF5 scalar dataset safely into a Python scalar."""
    val = ds[()]
    # h5py may return numpy scalar types; convert to python scalar where possible
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    if isinstance(val, (np.generic,)):
        return val.item()
    return val

def _read_string_dataset(ds):
    """Read a string dataset (often stored as fixed-length ASCII) into a Python str."""
    val = ds[()]
    # Could be bytes, numpy bytes_, or array of length 1
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.shape == ():  # scalar array
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

def extract_fields_with_curl(
    hdf5_file,
    output_file="vis_fields.npz",
    include_curl=True,
    include_magnetic_curl=False,
):
    with h5py.File(hdf5_file, "r") as f:
        # ----------------------------
        # Mesh grids (1D)
        # ----------------------------
        r = f["/mesh/grid_r"][:]                 # (nr,)
        theta = f["/mesh/grid_theta"][:]         # colatitude (ntheta,)
        phi = f["/mesh/grid_phi"][:]             # (nphi,)

        latitude = np.pi / 2.0 - theta           # requested: latitude

        nr, ntheta, nphi = len(r), len(theta), len(phi)

        # ----------------------------
        # Fields (expecting shape (nr, ntheta, nphi))
        # ----------------------------
        u_r     = f["/velocity/velocity_r"][:]
        u_theta = f["/velocity/velocity_theta"][:]
        u_phi   = f["/velocity/velocity_phi"][:]

        B_r     = f["/magnetic/magnetic_r"][:]
        B_theta = f["/magnetic/magnetic_theta"][:]
        B_phi   = f["/magnetic/magnetic_phi"][:]

        T = f["/temperature/temperature"][:]

        # ----------------------------
        # Basic shape sanity checks
        # ----------------------------
        def _check_shape(name, arr):
            if arr.ndim != 3:
                raise ValueError(f"{name} should be 3D, got shape {arr.shape}")
            if arr.shape != (nr, ntheta, nphi):
                # Most common “silent bug”: swapped axes. Fail fast.
                raise ValueError(
                    f"{name} shape mismatch. Expected (nr, ntheta, nphi) = "
                    f"({nr}, {ntheta}, {nphi}), got {arr.shape}. "
                    f"If your data is stored with different axis order, transpose accordingly."
                )

        for nm, a in [
            ("u_r", u_r), ("u_theta", u_theta), ("u_phi", u_phi),
            ("B_r", B_r), ("B_theta", B_theta), ("B_phi", B_phi),
            ("T", T),
        ]:
            _check_shape(nm, a)

        # ----------------------------
        # Time / timestep and physical parameters
        # ----------------------------
        time = _read_scalar(f["/run/time"])
        timestep = _read_scalar(f["/run/timestep"])

        # Physical scalars (if present)
        phys = {}
        for key in ["ekman", "inertia", "rayleigh", "roberts"]:
            path = f"/physical/{key}"
            if path in f:
                phys[key] = _read_scalar(f[path])

        # Physical string “modes” (if present)
        phys_str = {}
        for key in ["velocity", "temperature", "magnetic"]:
            path = f"/physical/{key}"
            if path in f:
                phys_str[key] = _read_string_dataset(f[path])

        # Root attributes (header/type/version)
        root_attrs = {}
        for k in ["header", "type", "version"]:
            if k in f["/"].attrs:
                v = f["/"].attrs[k]
                if isinstance(v, (bytes, np.bytes_)):
                    v = v.decode("utf-8", errors="replace")
                root_attrs[k] = str(v)

        out = dict(
            # grids
            r=r, theta=theta, phi=phi,
            latitude=latitude,
            # fields
            u_r=u_r, u_theta=u_theta, u_phi=u_phi,
            B_r=B_r, B_theta=B_theta, B_phi=B_phi,
            T=T,
            # run info
            time=time,
            timestep=timestep,
        )

        # attach physical params
        for k, v in phys.items():
            out[f"physical_{k}"] = v
        for k, v in phys_str.items():
            out[f"physical_{k}_str"] = v
        for k, v in root_attrs.items():
            out[f"root_attr_{k}"] = v

        # ----------------------------
        # Curl fields and axial vorticity
        # ----------------------------
        if include_curl:
            w_r     = f["/velocity_curl/velocity_curl_r"][:]
            w_theta = f["/velocity_curl/velocity_curl_theta"][:]
            w_phi   = f["/velocity_curl/velocity_curl_phi"][:]

            for nm, a in [("w_r", w_r), ("w_theta", w_theta), ("w_phi", w_phi)]:
                _check_shape(nm, a)

            # Option A (colatitude theta): zhat = cos(theta) e_r - sin(theta) e_theta
            cosT = np.cos(theta)[None, :, None]
            sinT = np.sin(theta)[None, :, None]
            w_z_theta = w_r * cosT - w_theta * sinT

            # Option B (latitude lambda): cos(theta)=sin(lambda), sin(theta)=cos(lambda)
            sinLat = np.sin(latitude)[None, :, None]
            cosLat = np.cos(latitude)[None, :, None]
            w_z_lat = w_r * sinLat - w_theta * cosLat

            # Consistency check (should match to roundoff)
            #max_diff = np.max(np.abs(w_z_theta - w_z_lat))
            out["curl_u_axial"] = w_z_theta
            #out["curl_u_axial_consistency_maxabs"] = max_diff

            # If you also want to save the full curl components, uncomment:
            #out["curl_u_r"] = w_r
            #out["curl_u_theta"] = w_theta
            #out["curl_u_phi"] = w_phi

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

    np.savez_compressed(output_file, **out)
    print(f"Saved extracted fields to {output_file}")
    if "curl_u_axial_consistency_maxabs" in out:
        print("axial vorticity consistency max|diff| =", out["curl_u_axial_consistency_maxabs"])


# Example usage
extract_fields_with_curl(
    "visState0000.hdf5",
    "vis_fields_0000.npz",
    include_curl=True,
    include_magnetic_curl=False
)

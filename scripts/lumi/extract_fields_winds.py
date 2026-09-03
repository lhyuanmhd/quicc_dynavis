import h5py
import numpy as np


def extract_fields(
    hdf5_file,
    output_file="vis_fields.npz",
    include_curl=True,
    include_winds=True,
):
    with h5py.File(hdf5_file, "r") as f:

        # Mesh
        r = f["/mesh/grid_r"][:]
        theta = f["/mesh/grid_theta"][:]
        phi = f["/mesh/grid_phi"][:]
        latitude = np.pi / 2.0 - theta

        # Standard fields
        u_r = f["/velocity/velocity_r"][:]
        u_theta = f["/velocity/velocity_theta"][:]
        u_phi = f["/velocity/velocity_phi"][:]

        B_r = f["/magnetic/magnetic_r"][:]
        B_theta = f["/magnetic/magnetic_theta"][:]
        B_phi = f["/magnetic/magnetic_phi"][:]

        T = f["/temperature/temperature"][:]

        out = {
            "r": r,
            "theta": theta,
            "phi": phi,
            "latitude": latitude,

            "u_r": u_r,
            "u_theta": u_theta,
            "u_phi": u_phi,

            "B_r": B_r,
            "B_theta": B_theta,
            "B_phi": B_phi,

            "T": T,

            "time": f["/run/time"][()],
            "timestep": f["/run/timestep"][()],
        }

        # Velocity curl -> axial vorticity
        if include_curl:
            w_r = f["/velocity_curl/velocity_curl_r"][:]
            w_theta = f["/velocity_curl/velocity_curl_theta"][:]

            cos_theta = np.cos(theta)[None, :, None]
            sin_theta = np.sin(theta)[None, :, None]

            out["curl_u_axial"] = (
                w_r * cos_theta
                - w_theta * sin_theta
            )

        # Wind decomposition
        if include_winds:
            for name in [
                "thermal_wind",
                "magnetic_wind",
            ]:
                out[f"{name}_r"] = f[
                    f"/{name}/{name}_r"
                ][:]

                out[f"{name}_theta"] = f[
                    f"/{name}/{name}_theta"
                ][:]

                out[f"{name}_phi"] = f[
                    f"/{name}/{name}_phi"
                ][:]

    np.savez_compressed(
        output_file,
        **out,
    )

    print(f"Saved to {output_file}")


extract_fields(
    "visState0000.hdf5",
    "vis_fields_0000.npz",
    include_curl=True,
    include_winds=True,
)
"""
Extract certain physical fields information for visulization !
"""
import h5py
import numpy as np

def extract_fields(hdf5_file, output_file="fields_data.npz"):
    with h5py.File(hdf5_file, "r") as f:
        # Mesh grids
        r = f["/mesh/grid_r"][:]
        theta = f["/mesh/grid_theta"][:]
        phi = f["/mesh/grid_phi"][:]

        # Velocity components
        u_r = f["/velocity/velocity_r"][:]
        u_theta = f["/velocity/velocity_theta"][:]
        u_phi = f["/velocity/velocity_phi"][:]

        # Magnetic components
        B_r = f["/magnetic/magnetic_r"][:]
        B_theta = f["/magnetic/magnetic_theta"][:]
        B_phi = f["/magnetic/magnetic_phi"][:]

        # Temperature
        T = f["/temperature/temperature"][:]

    # Save everything into one compressed file
    np.savez_compressed(
        output_file,
        r=r, theta=theta, phi=phi,
        u_r=u_r, u_theta=u_theta, u_phi=u_phi,
        B_r=B_r, B_theta=B_theta, B_phi=B_phi,
        T=T
    )
    print(f"Saved extracted fields to {output_file}")

# Example usage
extract_fields("visState0000.hdf5", "vis_fields.npz")

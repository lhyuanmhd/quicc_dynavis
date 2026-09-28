from pathlib import Path

from quicc_dynavis.azimuthal import (
    discover_snapshots,
    analyze_snapshots,
    save_azimuthal_csv,
)


root = Path(
    "runs"
)

files = discover_snapshots(root)

print(f"Found {len(files)} snapshots.")


# diagnostics = {
#     "ur_abs": {
#         "quantity": "velocity_r",
#         "absolute": True,
#     },
#     "utheta_abs": {
#         "quantity": "velocity_theta",
#         "absolute": True,
#     },
#     "uphi_abs": {
#         "quantity": "velocity_phi",
#         "absolute": True,
#     },
#     "umag": {
#         "quantity": "velocity_magnitude",
#         "absolute": False,
#     },
#     "Hz_abs": {
#         "quantity": "axial_helicity",
#         "absolute": True,
#     },

#     "Hz": {
#     "quantity": "axial_helicity",
#     "absolute": False,
#     }
# }

diagnostics = {
    # ----------------------------------------------------------
    # Velocity
    # ----------------------------------------------------------
    "ur_abs": {
        "quantity": "velocity_r",
        "absolute": True,
    },
    "utheta_abs": {
        "quantity": "velocity_theta",
        "absolute": True,
    },
    "uphi_abs": {
        "quantity": "velocity_phi",
        "absolute": True,
    },
    "umag": {
        "quantity": "velocity_magnitude",
        "absolute": False,
    },

    # ----------------------------------------------------------
    # Vorticity
    # ----------------------------------------------------------
    "wr_abs": {
        "quantity": "vorticity_r",
        "absolute": True,
    },
    "wtheta_abs": {
        "quantity": "vorticity_theta",
        "absolute": True,
    },
    "wphi_abs": {
        "quantity": "vorticity_phi",
        "absolute": True,
    },
    "wmag": {
        "quantity": "vorticity_magnitude",
        "absolute": False,
    },

    # Axial vorticity
    "wz_abs": {
        "quantity": "vorticity_z",
        "absolute": True,
    },
    "wz": {
        "quantity": "vorticity_z",
        "absolute": False,
    },

    # ----------------------------------------------------------
    # Axial helicity
    # ----------------------------------------------------------
    "Hz_abs": {
        "quantity": "axial_helicity",
        "absolute": True,
    },
    "Hz": {
        "quantity": "axial_helicity",
        "absolute": False,
    },
}


df = analyze_snapshots(
    files,
    diagnostics,
    hemisphere="north",
    modes=(1,),
)


save_azimuthal_csv(
    df,
    "diagnostics/azimuthal_timeseries.csv",
    compact=True,
)
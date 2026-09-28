from pathlib import Path

from quicc_dynavis.azimuthal import (
    analyze_snapshots,
    save_azimuthal_csv,
)


snapshot_dir = Path(".")

files = sorted(
    snapshot_dir.glob("visState*.hdf5")
)


diagnostics = {
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
    "Hz_abs": {
        "quantity": "axial_helicity",
        "absolute": True,
    },
}


df = analyze_snapshots(
    files,
    diagnostics,
    hemisphere="north",
    modes=(1, 2, 3),
)


save_azimuthal_csv(
    df,
    "azimuthal_timeseries.csv",
)


print(df.head())
print()
print(f"Processed {len(df)} snapshots.")
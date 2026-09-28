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
    modes=(1,),
)


save_azimuthal_csv(
    df,
    "diagnostics/azimuthal_timeseries.csv",
    compact=True,
)
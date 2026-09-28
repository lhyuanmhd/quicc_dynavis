from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from quicc_dynavis.azimuthal import (
    read_grid,
    read_time,
    construct_scalar_field,
    analyze_azimuthal_structure,
)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})


# ============================================================
# Settings
# ============================================================

RUNS_DIR = Path("runs")
FIGURES_DIR = Path("figures")

HEMISPHERE = "north"
Z_MAX = 0.7
QUANTITY = "axial_helicity"


# ============================================================
# Find snapshots
# ============================================================

files = sorted(
    RUNS_DIR.glob("run*/visu*/visState0000.hdf5")
)

if not files:
    raise FileNotFoundError(
        f"No visState0000.hdf5 files found under {RUNS_DIR}"
    )


# ============================================================
# Analyze snapshots
# ============================================================

results = []

for filename in files:

    print(f"Processing {filename} ...")

    with h5py.File(filename, "r") as h5:

        time = read_time(h5)
        r, theta, phi = read_grid(h5)

        field = construct_scalar_field(
            h5,
            QUANTITY,
        )

        result = analyze_azimuthal_structure(
            field,
            r,
            theta,
            phi,
            hemisphere=HEMISPHERE,
            use_absolute=False,     # signed helicity
            max_m=None,             # no low-m truncation
            z_max=Z_MAX,
        )

    results.append({
        "time": time,
        "phi": result["phi"],
        "profile": result["profile"],
    })


results.sort(
    key=lambda x: x["time"]
)


# ============================================================
# Plot full azimuthal profiles
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

OFFSET = 1.5

for i, result in enumerate(results):

    phi_deg = np.degrees(
        result["phi"]
    )

    profile = result["profile"]

    # Remove azimuthal mean.
    y = profile - np.mean(profile)

    # Normalize each snapshot independently.
    # This emphasizes morphology and peak position.
    scale = np.max(np.abs(y))

    if scale > 0.0:
        y = y / scale

    # Vertical offset.
    y = y + i * OFFSET

    ax.plot(
        phi_deg,
        y,
        linewidth=1.2,
        label=rf"$t={result['time']:.3f}$",
    )


ax.set_xlim(0, 360)

ax.set_xlabel(
    r"$\phi$ (deg)"
)

ax.set_ylabel(
    r"$\langle H_z\rangle_{r,\theta}$"
)

ax.set_yticks([])

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
)

fig.tight_layout()


# ============================================================
# Save
# ============================================================

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    FIGURES_DIR
    / "signed_helicity_full_profiles.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nFigure saved to: {output_file}"
)

plt.show()
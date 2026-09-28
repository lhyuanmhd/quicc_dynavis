from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

CSV_FILE = Path("diagnostics/azimuthal_timeseries.csv")
FIGURES_DIR = Path("figures")

PHASE_COLUMN = "Hz_abs_m1_orientation_deg"


# ============================================================
# Read data
# ============================================================

df = pd.read_csv(CSV_FILE)

df = df[
    ["time", PHASE_COLUMN]
].dropna()

df = df.sort_values("time")

time = df["time"].to_numpy()
phi = df[PHASE_COLUMN].to_numpy()


# ============================================================
# Unwrap phase
# ============================================================

phi_unwrapped = np.degrees(
    np.unwrap(
        np.radians(phi)
    )
)


# ============================================================
# Print values
# ============================================================

print("\nPhase evolution:\n")

for t, p, pu in zip(
    time,
    phi,
    phi_unwrapped,
):
    print(
        f"t = {t:.3f}   "
        f"phi = {p:8.2f} deg   "
        f"unwrapped = {pu:8.2f} deg"
    )


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    time,
    phi_unwrapped,
    "o-",
    linewidth=1.8,
    markersize=7,
)

ax.set_xlabel("Time")

ax.set_ylabel(
    r"$\phi_{|H_z|,\,m=1}$ (deg)"
)

ax.grid(
    alpha=0.25
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
    / "Hz_abs_m1_phase_vs_time.png"
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
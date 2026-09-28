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
# Ask user for maximum m
# ============================================================

def ask_m_max():
    """Ask the user for the maximum azimuthal mode."""

    while True:
        value = input(
            "Enter maximum azimuthal mode m: "
        ).strip()

        try:
            m_max = int(value)

            if m_max < 1:
                raise ValueError

            return m_max

        except ValueError:
            print(
                "Please enter a positive integer, "
                "for example 1, 3, 5, or 20."
            )


# ============================================================
# Find snapshots
# ============================================================

def find_snapshots(runs_dir):
    """
    Find all visState0000.hdf5 files inside run folders.
    """

    files = sorted(
        runs_dir.glob(
            "run*/visu*/visState0000.hdf5"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No visState0000.hdf5 files found under {runs_dir}"
        )

    print(f"Found {len(files)} snapshots:")

    for filename in files:
        print(f"  {filename}")

    return files


# ============================================================
# Low-pass reconstruction
# ============================================================

def reconstruct_low_m(result, m_max):
    """
    Reconstruct azimuthal profile using Fourier modes
    0 <= m <= m_max.
    """

    phi = result["phi"]
    coeff = result["coefficient"]

    envelope = np.zeros(
        len(phi),
        dtype=float,
    )

    # m = 0
    envelope += coeff[0].real

    # m > 0
    for m in range(
        1,
        min(m_max + 1, len(coeff)),
    ):
        envelope += 2.0 * np.real(
            coeff[m] * np.exp(1j * m * phi)
        )

    return envelope


# ============================================================
# Analyze one snapshot
# ============================================================

def analyze_snapshot(filename, m_max):

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
            use_absolute=True,
            max_m=m_max,
            z_max=Z_MAX,
        )

    envelope = reconstruct_low_m(
        result,
        m_max,
    )

    return {
        "filename": filename,
        "time": time,
        "phi": result["phi"],
        "profile": result["profile"],
        "envelope": envelope,
    }


# ============================================================
# Main
# ============================================================

m_max = ask_m_max()

print(
    f"\nAnalyzing helicity envelope with m <= {m_max}\n"
)

files = find_snapshots(RUNS_DIR)

results = []

for filename in files:
    results.append(
        analyze_snapshot(
            filename,
            m_max,
        )
    )


# Sort by physical simulation time
results.sort(
    key=lambda x: x["time"]
)


# ============================================================
# Plot stacked envelopes
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

OFFSET = 1.5

for i, result in enumerate(results):

    phi_deg = np.degrees(
        result["phi"]
    )

    envelope = result["envelope"]

    # Remove axisymmetric mean.
    y = envelope - np.mean(envelope)

    # Normalize each snapshot so that we focus on shape/drift.
    scale = np.max(np.abs(y))

    if scale > 0.0:
        y = y / scale

    # Vertical offset.
    y = y + i * OFFSET

    ax.plot(
        phi_deg,
        y,
        label=rf"$t={result['time']:.3f}$",
    )


ax.set_xlim(
    0,
    360,
)

ax.set_xlabel(
    r"$\phi$ (deg)"
)

ax.set_ylabel(
    rf"$\langle |H_z| \rangle_{{r,\theta}},\ m\leq {m_max}$"
)

ax.set_yticks([])

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
)

fig.tight_layout()


# ============================================================
# Save figure
# ============================================================

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    FIGURES_DIR
    / f"helicity_m{m_max}_envelope.png"
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
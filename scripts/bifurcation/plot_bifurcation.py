
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
})


HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "bifurcation_summary.csv")

COLORS = {
    "tor": "#3268A8",
    "pol": "#D17A36",
}

MARKERS = {
    "ED": "o",
    "AD": "s",
}


def plot_energy_components(ax, prefix, ylabel):
    for _, row in df.iterrows():
        q = row["q"]
        marker = MARKERS.get(row["state"], "D")
        is_transient = row["phase"] == "pretransition"

        for component in ("tor", "pol"):
            value = row[f"{prefix}_{component}_mean"]

            if not np.isfinite(value) or value <= 0:
                continue

            color = COLORS[component]

            ax.scatter(
                q,
                value,
                marker=marker,
                s=52,
                facecolors="none" if is_transient else color,
                edgecolors=color,
                linewidths=1.3,
                zorder=3,
            )

    ax.set_xlabel(r"Roberts number $q$")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(alpha=0.15)


fig, axes = plt.subplots(
    1, 3, figsize=(14.5, 4.4),
    constrained_layout=True,
)

plot_energy_components(
    axes[0],
    "Emag",
    r"Magnetic energy",
)

plot_energy_components(
    axes[1],
    "Ekin",
    r"Kinetic energy",
)

# g10 growth/decay rate
valid = df.dropna(subset=["sigma_g10"]).sort_values("q")

axes[2].axhline(
    0, color="0.4", linestyle="--", linewidth=1
)

for _, row in valid.iterrows():
    marker = MARKERS.get(row["state"], "D")

    axes[2].scatter(
        row["q"],
        row["sigma_g10"],
        marker=marker,
        color=COLORS["tor"],
        s=54,
        zorder=3,
    )

    error = row.get("sigma_fit_se", np.nan)

    if np.isfinite(error) and error > 0:
        axes[2].errorbar(
            row["q"],
            row["sigma_g10"],
            yerr=error,
            fmt="none",
            ecolor=COLORS["tor"],
            capsize=2,
            alpha=0.55,
        )

axes[2].set_xlabel(r"Roberts number $q$")
axes[2].set_ylabel(r"$\sigma_{g_{10}}$")
axes[2].grid(alpha=0.15)

for label, ax in zip("(a)(b)(c)", axes):
    ax.text(
        0.04, 0.95, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontweight="bold",
    )

legend_handles = [
    Line2D(
        [], [], marker="o", linestyle="none",
        color="black", label="ED",
    ),
    Line2D(
        [], [], marker="s", linestyle="none",
        color="black", label="AD",
    ),
    Line2D(
        [], [], marker="o", linestyle="none",
        markerfacecolor="none", markeredgecolor="black",
        label="Pre-transition",
    ),
    Line2D(
        [], [], color=COLORS["tor"], lw=2,
        label="Toroidal",
    ),
    Line2D(
        [], [], color=COLORS["pol"], lw=2,
        label="Poloidal",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="outside lower center",
    ncol=5,
    frameon=False,
)

save_path = HERE / "bifurcation_energy_and_sigma.pdf"
fig.savefig(save_path, bbox_inches="tight")
plt.show()

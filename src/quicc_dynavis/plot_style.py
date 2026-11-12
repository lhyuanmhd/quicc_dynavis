
"""
Centralized plotting styles for quicc_dynavis.

Provides:
    DEFAULT_COLORS : consistent colors for physical quantities
    FIELD_LABELS   : LaTeX labels for physical quantities
    set_plot_style : configure Matplotlib defaults
    get_color      : retrieve a field color
    get_label      : retrieve a LaTeX field label
"""

import matplotlib as mpl


# ============================================================
# Default colors
# ============================================================

DEFAULT_COLORS = {
    "kinetic": "royalblue",
    "magnetic": "darkorange",
    "dipolarity": "red",
    "dipole_angle": "gray",
    "nusselt": "purple",
    "temperature": "brown",
    "viscous": "teal",
    "ohmic": "peru",
    "fohm": "dimgray",
    "ED_orientation": "darkgreen",
    "buoyancy": "mediumvioletred",
}


# ============================================================
# LaTeX labels
# ============================================================

FIELD_LABELS = {
    # Energies
    "kinetic": r"$E_{\mathrm{kin}}$",
    "magnetic": r"$E_{\mathrm{mag}}$",
    "kinE": r"$E_{\mathrm{kin}}$",
    "magE": r"$E_{\mathrm{mag}}$",

    # Magnetic diagnostics
    "Lambda": r"$\Lambda$",
    "Rm": r"$Rm$",
    "dipolarity": r"$f_{\mathrm{dip}}$",
    "fdip": r"$f_{\mathrm{dip}}$",
    "dipole_angle": r"$\theta$",
    "tilt": r"$\theta$",
    "ED_orientation": r"$B_{\mathrm{ED}}$",

    # Flow diagnostics
    "Ro": r"$Ro$",
    "local_Ro": r"$Ro_{\ell}$",
    "flow_degree": r"$\ell_u$",
    "Pe": r"$Pe$",

    # Thermal diagnostics
    "temperature": r"$T$",
    "nusselt": r"$Nu$",
    "Nu": r"$Nu$",

    # Dissipation and power
    "viscous": r"$D_{\nu}$",
    "visDis": r"$D_{\nu}$",
    "ohmic": r"$D_{\eta}$",
    "ohmDis": r"$D_{\eta}$",
    "fohm": r"$f_{\mathrm{ohm}}$",
    "buoyancy": r"$P_{\mathrm{buoy}}$",

    # Control parameters
    "q": r"$q$",
    "Ra": r"$Ra$",
    "E": r"$E$",
    "Ek": r"$E$",
    "Pm": r"$Pm$",
    "Pr": r"$Pr$",

    # Time
    "time": r"$t/\tau_{\eta}$",
}


# ============================================================
# Helper functions
# ============================================================

def get_color(field, default=None):
    """Return the default color for a physical quantity."""
    return DEFAULT_COLORS.get(field, default)


def get_label(field, default=None):
    """Return the LaTeX label for a physical quantity."""
    return FIELD_LABELS.get(
        field,
        field if default is None else default,
    )


# ============================================================
# Matplotlib style
# ============================================================

def set_plot_style(
    fontsize=12,
    labelsize=13,
    titlesize=14,
    legendsize=11,
    linewidth=1.5,
    use_tex=False,
):
    """
    Configure Matplotlib plotting defaults.

    Parameters
    ----------
    fontsize : int
        Base font size.
    labelsize : int
        Axis label font size.
    titlesize : int
        Title font size.
    legendsize : int
        Legend font size.
    linewidth : float
        Default line width.
    use_tex : bool
        Whether to use external LaTeX rendering.
    """

    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        "axes.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": legendsize,
        "lines.linewidth": linewidth,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "text.usetex": use_tex,
    })

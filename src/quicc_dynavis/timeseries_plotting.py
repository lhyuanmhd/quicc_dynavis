"""Plotting helpers for QuICC dynamo time series."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .timeseries_data import TimeseriesData, HydroTimeseriesData 
from .timeseries_diagnostics import DynamoDiagnostics, HydroDiagnostics


DEFAULT_COLORS = {
    "kinetic": "royalblue",
    "magnetic": "orange",
    "dipolarity": "coral",
    "dipole_angle": "gray",
    "nusselt": "purple",
    "temperature": "brown",
    "viscous": "teal",
    "ohmic": "peru",
    "fohm": "dimgray",
}

def resolve_time_limits(
    time: np.ndarray,
    xlim=None,
    padding_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return explicit time-axis limits.

    When xlim is supplied, it must contain exactly two values. Otherwise,
    limits are inferred from the supplied time array with a small padding.
    """
    if xlim is not None:
        if len(xlim) != 2:
            raise ValueError(
                "xlim must contain exactly two values: (xmin, xmax)"
            )
        return float(xlim[0]), float(xlim[1])

    if len(time) == 0:
        raise ValueError("Cannot infer x-axis limits from an empty time array")

    time_min = float(np.nanmin(time))
    time_max = float(np.nanmax(time))
    time_span = time_max - time_min

    if time_span == 0:
        padding = max(abs(time_min) * padding_fraction, 1.0)
    else:
        padding = padding_fraction * time_span

    return time_min - padding, time_max + padding


def create_timeseries_figure(
    has_dissipation: bool,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Create the standard dynamo time-series figure."""
    plt.close("all")

    panel_count = 5 if has_dissipation else 4
    figure_height = 12 if has_dissipation else 10

    fig, axes_array = plt.subplots(
        panel_count,
        1,
        figsize=(8, figure_height),
        sharex=True,
        dpi=180,
        squeeze=False,
    )

    axes = tuple(axes_array[:, 0])
    return fig, axes


def create_dipolarity_figure():
    """Create the compact magnetic-energy and dipole diagnostics figure."""
    plt.close("all")

    fig, axes_array = plt.subplots(
        4,
        1,
        figsize=(12, 7),
        sharex=True,
        dpi=180,
        squeeze=False,
    )

    axes = tuple(axes_array[:, 0])
    return fig, axes

def create_equatorial_dipole_figure():
    """Create the equatorial-dipole evolution figure."""

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    fig.subplots_adjust(
        hspace=0.08,
    )

    return fig, axes

def create_hydro_timeseries_figure(
    has_dissipation: bool,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Create the standard hydro time-series figure."""
    plt.close("all")

    panel_count = 3 if has_dissipation else 2
    figure_height = 8 if has_dissipation else 6

    fig, axes_array = plt.subplots(
        panel_count,
        1,
        figsize=(8, figure_height),
        sharex=True,
        dpi=180,
        squeeze=False,
    )

    return fig, tuple(axes_array[:, 0])


def plot_kinetic_energy_panel(
    ax: Axes,
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Plot physical kinetic energy."""
    ax.plot(
        data.tkin,
        diagnostics.physical_kinetic_energy,
        color=DEFAULT_COLORS["kinetic"],
        label=r"$\mathcal{E}_{kin}$",
    )

    ax.set_title(
        rf"$E: {data.Ek:.1e}, Ra: {data.Ra:.2e}, "
        rf"q: {data.q:.2f}, "
        rf"(N,L,M): ({data.N},{data.L},{data.M})$"
    )
    ax.set_ylabel("Energy density")
    ax.set_yscale("log")
    ax.set_xlim(time_limits)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

def plot_magnetic_energy_panel(
    ax: Axes,
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Plot magnetic energy only."""
    ax.plot(
        data.tmag,
        data.mag_total,
        color=DEFAULT_COLORS["magnetic"],
        label=r"$\mathcal{E}_{mag}$", #fixed y label 
    )

    ax.set_title(
        rf"$E: {data.Ek:.1e}, Ra: {data.Ra:.2e}, "
        rf"q: {data.q:.2f}, "
        rf"(N,L,M): ({data.N},{data.L},{data.M})$"
    )
    ax.set_ylabel(r"$\mathcal{E}_{mag}$") 
    ax.set_yscale("log")
    ax.set_xlim(time_limits)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

def plot_magnetic_thermal_panel(
    ax: Axes,
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Plot magnetic energy and scaled thermal perturbation energy."""
    ax.plot(
        data.ttem,
        data.Ek * diagnostics.thermal_perturbation,
        color=DEFAULT_COLORS["temperature"],
        label=r"$E \mathcal{E}_{t}$",
    )

    ax.plot(
        data.tmag,
        data.mag_total,
        color=DEFAULT_COLORS["magnetic"],
        label=r"$\mathcal{E}_{mag}$",
    )

    ax.set_ylabel("Energy density")
    ax.set_yscale("log")
    ax.set_xlim(time_limits)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()


def plot_dipolarity_panel(
    ax: Axes,
    data: TimeseriesData,
    time_limits: tuple[float, float],
    ylabel: str = "Dipolarity",
) -> None:
    """Plot dipolarity."""
    if len(data.tdip) > 0 and len(data.fdip) > 0:
        ax.plot(
            data.tdip,
            data.fdip,
            color=DEFAULT_COLORS["dipolarity"],
            label=r"$f_{\mathrm{dip}}$",
            alpha=0.6,
        )

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_xlim(time_limits)


def plot_dipole_angle_panel(
    ax: Axes,
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
    time_limits: tuple[float, float],
    *,
    set_xlabel: bool,
    ylabel: str = "Dipole angle (deg)",
) -> None:
    """Plot dipole tilt angle."""
    if len(data.tdip) > 0 and len(diagnostics.dipole_angle) > 0:
        ax.plot(
            data.tdip,
            diagnostics.dipole_angle,
            color=DEFAULT_COLORS["dipole_angle"],
            alpha=0.7,
        )

    ax.axhline(
        90,
        linestyle="--",
        alpha=0.4,
        label=r"$90^{\circ}$",
    )

    ax.set_ylabel(ylabel)
    ax.set_ylim(-5, 185)
    ax.set_yticks([0, 90, 180])
    ax.set_xlim(time_limits)

    if set_xlabel:
        ax.set_xlabel("Time")

    ax.legend()

def plot_dipole_components_panel(
    ax: Axes,
    data: TimeseriesData,
    time_limits: tuple[float, float],
    *,
    set_xlabel: bool,
    ylabel: str = "Dipole Gauss coefficients",
) -> None:
    """Plot axial and equatorial dipole Gauss coefficients."""

    if (
        len(data.tdip) > 0
        and len(data.g10) > 0
        and len(data.g11) > 0
        and len(data.h11) > 0
    ):
        ed_amp = np.sqrt(data.g11**2 + data.h11**2)

        ax.semilogy(
            data.tdip,
            abs(data.g10),
            label=r"$|g_1^0|$",
            alpha=0.7,
        )

        ax.semilogy(
            data.tdip,
            abs(data.g11),
            label=r"$|g_1^1|$",
            alpha=0.7,
        )

        ax.semilogy(
            data.tdip,
            abs(data.h11),
            label=r"$|h_1^1|$",
            alpha=0.7,
        )

        ax.semilogy(
            data.tdip,
            ed_amp,
            label=r"$\sqrt{(g_1^1)^2+(h_1^1)^2}$",
            linestyle="--",
            linewidth=1.5,
        )

    ax.axhline(
        0,
        linestyle="--",
        alpha=0.4,
    )

    ax.set_ylabel(ylabel)
    ax.set_xlim(time_limits)

    if set_xlabel:
        ax.set_xlabel("Time")

    ax.legend()


# def plot_equatorial_dipole_orientation_panel(
#     ax: Axes,
#     data: TimeseriesData,
#     time_limits: tuple[float, float],
#     *,
#     set_xlabel: bool,
#     ylabel: str = r"$\phi_{\mathrm{ED}}$ (deg)",
# ) -> None:
#     """Plot the azimuthal orientation of the equatorial dipole."""

#     if (
#         len(data.tdip) > 0
#         and len(data.g11) > 0
#         and len(data.h11) > 0
#     ):
#         phi_ed = np.degrees(
#             np.arctan2(data.h11, data.g11)
#         )

#         # Map orientation to [0, 360) degrees
#         phi_ed = np.mod(phi_ed, 360.0)

#         ax.plot(
#             data.tdip,
#             phi_ed,
#             alpha=0.7,
#         )

#     ax.set_ylabel(ylabel)
#     ax.set_xlim(time_limits)
#     ax.set_ylim(0, 360)
#     ax.set_yticks([0, 90, 180, 270, 360])

#     if set_xlabel:
#         ax.set_xlabel("Time")
def plot_equatorial_dipole_orientation_panel(
    ax: Axes,
    data: TimeseriesData,
    time_limits: tuple[float, float],
    *,
    set_xlabel: bool,
    ylabel: str = r"$\phi_{\mathrm{ED}}$ (deg)",
) -> None:
    """Plot the azimuthal orientation of the equatorial dipole."""

    if (
        len(data.tdip) > 0
        and len(data.g11) > 0
        and len(data.h11) > 0
    ):
        phi_ed = np.degrees(
            np.arctan2(data.h11, data.g11)
        )

        # Map orientation to [0, 360) degrees
        phi_ed = np.mod(phi_ed, 360.0)

        ax.plot(
            data.tdip,
            phi_ed,
            alpha=0.7,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlim(time_limits)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])

    if set_xlabel:
        ax.set_xlabel("Time")

def plot_dipole_amplitudes_panel(
    ax: Axes,
    data: TimeseriesData,
    time_limits: tuple[float, float],
    *,
    set_xlabel: bool,
    ylabel: str = "Dipole amplitude",
) -> None:
    """Plot axial- and equatorial-dipole amplitudes."""

    if (
        len(data.tdip) > 0
        and len(data.g10) > 0
        and len(data.g11) > 0
        and len(data.h11) > 0
    ):
        axial_dipole = np.abs(data.g10)
        equatorial_dipole = np.sqrt(
            data.g11**2 + data.h11**2
        )

        ax.semilogy(
            data.tdip,
            axial_dipole,
            label=r"$|g_1^0|$",
            alpha=0.7,
        )

        ax.semilogy(
            data.tdip,
            equatorial_dipole,
            label=r"$D_{\mathrm{ED}}$",
            alpha=0.7,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlim(time_limits)

    if set_xlabel:
        ax.set_xlabel("Time")

    ax.legend()

def plot_dissipation_panel(
    ax: Axes,
    data: TimeseriesData,
    time_limits: tuple[float, float],
) -> None:
    """Plot available viscous and ohmic dissipation series."""
    if len(data.tkin_dis) > 0:
        ax.plot(
            data.tkin_dis,
            data.Ek * data.kin_dis_total,
            alpha=0.7,
            color=DEFAULT_COLORS["viscous"],
            label="Viscous dissipation",
        )

    if len(data.tmag_dis) > 0:
        ax.plot(
            data.tmag_dis,
            data.mag_dis_total,
            alpha=0.7,
            color=DEFAULT_COLORS["ohmic"],
            label="Ohmic dissipation",
        )

    ax.set_ylabel("Dissipation")
    ax.set_xlabel("Time")
    ax.set_yscale("log")
    ax.set_xlim(time_limits)
    ax.legend()



def plot_hydro_kinetic_panel(
    ax: Axes,
    data: HydroTimeseriesData,
    diagnostics: HydroDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Plot hydro kinetic energy."""
    ax.plot(
        data.tkin,
        diagnostics.physical_kinetic_energy,
        color=DEFAULT_COLORS["kinetic"],
        label=r"$\mathcal{E}_{kin}$",
    )

    ax.set_title(
        rf"Hydro, $E: {data.Ek:.1e}, Ra: {data.Ra:.2e}, "
        rf"(N,L,M): ({data.N},{data.L},{data.M})$"
    )
    ax.set_ylabel("Energy density")
    ax.set_yscale("log")
    ax.set_xlim(time_limits)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

def plot_hydro_thermal_panel(
    ax: Axes,
    data: HydroTimeseriesData,
    diagnostics: HydroDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
    *,
    set_xlabel: bool,
) -> None:
    """Plot scaled thermal perturbation energy."""
    ax.plot(
        data.ttem,
        data.Ek * diagnostics.thermal_perturbation,
        color=DEFAULT_COLORS["temperature"],
        label=r"$E\mathcal{E}_{t}$",
    )

    ax.set_ylabel("Energy density")
    ax.set_yscale("log")
    ax.set_xlim(time_limits)

    if ylim is not None:
        ax.set_ylim(ylim)

    if set_xlabel:
        ax.set_xlabel("Time")

    ax.legend()

def plot_hydro_dissipation_panel(
    ax: Axes,
    data: HydroTimeseriesData,
    time_limits: tuple[float, float],
) -> None:
    """Plot viscous dissipation."""
    if len(data.tkin_dis) > 0:
        ax.plot(
            data.tkin_dis,
            data.Ek * data.kin_dis_total,
            alpha=0.7,
            color=DEFAULT_COLORS["viscous"],
            label="Viscous dissipation",
        )

    ax.set_ylabel("Dissipation")
    ax.set_xlabel("Time")
    ax.set_yscale("log")
    ax.set_xlim(time_limits)
    ax.legend()


# populate figures

def populate_timeseries_figure(
    axes: Sequence[Axes],
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Populate every panel in the standard time-series figure."""
    expected_count = 5 if data.has_dissipation else 4

    if len(axes) != expected_count:
        raise ValueError(
            f"Expected {expected_count} axes, received {len(axes)}"
        )

    plot_kinetic_energy_panel(
        axes[0],
        data,
        diagnostics,
        time_limits,
        ylim=ylim,
    )

    plot_magnetic_energy_panel(
        axes[1],
        data,
        diagnostics,
        time_limits,
        ylim=ylim,
    )

    plot_dipolarity_panel(
        axes[2],
        data,
        time_limits,
    )

    plot_dipole_angle_panel(
        axes[3],
        data,
        diagnostics,
        time_limits,
        set_xlabel=not data.has_dissipation,
    )

    if data.has_dissipation:
        plot_dissipation_panel(
            axes[4],
            data,
            time_limits,
        )



# def populate_dipolarity_figure(
#     axes: Sequence[Axes],
#     data: TimeseriesData,
#     diagnostics: DynamoDiagnostics,
#     time_limits: tuple[float, float],
#     ylim=None,
# ) -> None:
#     """Populate the compact magnetic-energy and dipole figure."""
#     if len(axes) != 3:
#         raise ValueError(
#             f"Expected 3 axes, received {len(axes)}"
#         )

#     plot_magnetic_energy_panel(
#         axes[0],
#         data,
#         diagnostics,
#         time_limits,
#         ylim=ylim,
#     )

#     plot_dipolarity_panel(
#         axes[1],
#         data,
#         time_limits,
#         ylabel=r"$f_{\mathrm{dip}}$",
#     )

#     plot_dipole_angle_panel(
#         axes[2],
#         data,
#         diagnostics,
#         time_limits,
#         set_xlabel=True,
#         ylabel=r"$\theta$ (deg)",
#     )

def populate_dipolarity_figure(
    axes: Sequence[Axes],
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Populate the compact magnetic-energy and dipole figure."""
    if len(axes) != 4:
        raise ValueError(
            f"Expected 4 axes, received {len(axes)}"
        )

    plot_magnetic_energy_panel(
        axes[0],
        data,
        diagnostics,
        time_limits,
        ylim=ylim,
    )

    plot_dipolarity_panel(
        axes[1],
        data,
        time_limits,
        ylabel=r"$f_{\mathrm{dip}}$",
    )

    plot_dipole_components_panel(
        axes[2],
        data,
        time_limits,
        set_xlabel=False,
        ylabel="Dipole coefficients",
    )

    plot_dipole_angle_panel(
        axes[3],
        data,
        diagnostics,
        time_limits,
        set_xlabel=True,
        ylabel=r"$\theta$ (deg)",
    )


def populate_equatorial_dipole_figure(
    axes: Sequence[Axes],
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Populate the equatorial-dipole evolution figure."""
    if len(axes) != 4:
        raise ValueError(
            f"Expected 4 axes, received {len(axes)}"
        )

    # (a) Magnetic energy
    plot_magnetic_energy_panel(
        axes[0],
        data,
        diagnostics,
        time_limits,
        ylim=ylim,
    )

    # (b) Axial vs equatorial dipole amplitudes
    plot_dipole_amplitudes_panel(
        axes[1],
        data,
        time_limits,
        set_xlabel=False,
    )

    # (c) Dipole tilt angle
    plot_dipole_angle_panel(
        axes[2],
        data,
        diagnostics,
        time_limits,
        set_xlabel=False,
        ylabel=r"$\theta$ (deg)",
    )

    # (d) Equatorial-dipole orientation
    plot_equatorial_dipole_orientation_panel(
        axes[3],
        data,
        time_limits,
        set_xlabel=True,
        ylabel=r"$\phi_{\mathrm{ED}}$ (deg)",
    )   

def populate_hydro_timeseries_figure(
    axes: Sequence[Axes],
    data: HydroTimeseriesData,
    diagnostics: HydroDiagnostics,
    time_limits: tuple[float, float],
    ylim=None,
) -> None:
    """Populate the standard hydro time-series figure."""
    expected_count = 3 if data.has_dissipation else 2

    if len(axes) != expected_count:
        raise ValueError(
            f"Expected {expected_count} axes, received {len(axes)}"
        )

    plot_hydro_kinetic_panel(
        axes[0],
        data,
        diagnostics,
        time_limits,
        ylim=ylim,
    )

    plot_hydro_thermal_panel(
        axes[1],
        data,
        diagnostics,
        time_limits,
        ylim=ylim,
        set_xlabel=not data.has_dissipation,
    )

    if data.has_dissipation:
        plot_hydro_dissipation_panel(
            axes[2],
            data,
            time_limits,
        )

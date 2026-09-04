"""Diagnostic calculations for QuICC dynamo time series."""

from dataclasses import dataclass

import numpy as np

from .timeseries_data import TimeseriesData, HydroTimeseriesData


@dataclass
class DynamoDiagnostics:
    """Time-averaged diagnostics derived from one simulation case."""

    averaging_start_index: int
    
    mean_rossby: float

    physical_kinetic_energy: np.ndarray
    thermal_perturbation: np.ndarray
    dipole_angle: np.ndarray

    initial_magnetic_energy: float
    mean_kinetic_energy: float
    mean_magnetic_energy: float
    mean_thermal_perturbation: float
    mean_nusselt: float

    magnetic_reynolds_number: float
    elsasser_number: float

    mean_dipolarity: float
    relative_std_dipolarity: float

    dynamo_active: bool
    excursion: bool
    reversal: bool

    mean_viscous_dissipation: float
    mean_ohmic_dissipation: float
    ohmic_fraction: float

    vel_dis_length_scale: float
    mag_dis_length_scale: float




def _averaging_start_index(time: np.ndarray, fraction: float = 0.3) -> int:
    """Return the first index after the initial fraction of a time interval."""
    if len(time) == 0:
        raise ValueError("Cannot determine averaging interval from an empty time array")

    if len(time) == 1:
        return 0

    threshold = time[0] + fraction * (time[-1] - time[0])
    indices = np.flatnonzero(time >= threshold)

    if len(indices) == 0:
        return 0

    return int(indices[0])


def _mean_after_start(values: np.ndarray, start_index: int) -> float:
    """Return the mean from start_index onward, or NaN for missing data."""
    if values is None or len(values) == 0:
        return float("nan")

    return float(np.mean(values[start_index:]))


def _compute_dipole_angle(
    g10: np.ndarray,
    g11: np.ndarray,
    h11: np.ndarray,
) -> np.ndarray:
    """Compute dipole tilt angle in degrees."""
    denominator = np.sqrt(g10**2 + g11**2 + h11**2)

    ratio = np.divide(
        g10,
        denominator,
        out=np.full_like(g10, np.nan, dtype=float),
        where=denominator > 0,
    )

    ratio = np.clip(ratio, -1.0, 1.0)
    return np.degrees(np.arccos(ratio))


def _detect_excursion(dipole_angle: np.ndarray) -> bool:
    """Return whether the dipole crosses 90 degrees."""
    valid = dipole_angle[np.isfinite(dipole_angle)]

    if len(valid) < 2:
        return False

    below = valid[:-1] < 90.0
    above_next = valid[1:] >= 90.0

    above = valid[:-1] > 90.0
    below_next = valid[1:] <= 90.0

    return bool(np.any((below & above_next) | (above & below_next)))


def _detect_reversal(
    dipole_angle: np.ndarray,
    mean_dipolarity: float,
    dipolarity_threshold: float = 0.30,
) -> bool:
    """Apply the current empirical reversal criterion."""
    valid = dipole_angle[np.isfinite(dipole_angle)]

    if len(valid) == 0:
        return False

    excursion = _detect_excursion(valid)

    return bool(
        excursion
        and np.max(valid) > 150.0
        and np.min(valid) < 30.0
        and mean_dipolarity > dipolarity_threshold
    )


def compute_dynamo_diagnostics(
    data: TimeseriesData,
    averaging_fraction: float = 0.3,
    dynamo_energy_threshold: float = 6e-4,
    dynamo_rm_threshold: float = 30.0,
) -> DynamoDiagnostics:
    """Compute dynamo diagnostics from raw time-series data."""
    start_index = _averaging_start_index(
        data.tkin,
        fraction=averaging_fraction,
    )

    physical_kinetic_energy = np.asarray(
        data.kin_total,
        dtype=float,
    ).copy()

    if data.Pm != 0 and data.Pr != 0:
        physical_kinetic_energy *= data.Ek / data.Pm

    thermal_perturbation = (
        3.0 / 5.0
        * (data.q * data.Ra) ** 2
        * data.tem_total
    )

    dipole_angle = _compute_dipole_angle(
        data.g10,
        data.g11,
        data.h11,
    )

    initial_magnetic_energy = (
        float(data.mag_total[0])
        if len(data.mag_total) > 0
        else float("nan")
    )

    mean_kinetic_energy = _mean_after_start(
        physical_kinetic_energy,
        start_index,
    )
    mean_magnetic_energy = _mean_after_start(
        data.mag_total,
        start_index,
    )
    mean_thermal_perturbation = _mean_after_start(
        thermal_perturbation,
        start_index,
    )
    mean_nusselt = _mean_after_start(
        data.nusselt,
        start_index,
    )
    mean_dipolarity = _mean_after_start(
        data.fdip,
        start_index,
    )

    if np.isfinite(mean_dipolarity) and mean_dipolarity != 0:
        relative_std_dipolarity = float(
            np.std(data.fdip[start_index:]) / mean_dipolarity
        )
    else:
        relative_std_dipolarity = float("nan")
    

    # Rm is sqrt of Ekin independent of Ek and Pm, so we can use the raw kinetic energy time series
    magnetic_reynolds_number = float(
        np.sqrt(np.mean(data.kin_total[start_index:]))
    )
     
    # compue Rossby number based on mean Rm
    if data.Pm != 0 and data.Pr != 0:
        mean_rossby = 2*magnetic_reynolds_number * data.Ek / data.Pm
    
    else:
        #else return nan
        mean_rossby = np.nan

    elsasser_number = 2.0 * mean_magnetic_energy

    final_magnetic_energy = (
        float(data.mag_total[-1])
        if len(data.mag_total) > 0
        else float("nan")
    )

    dynamo_active = bool(
        np.isfinite(final_magnetic_energy)
        and final_magnetic_energy >= dynamo_energy_threshold
        and magnetic_reynolds_number >= dynamo_rm_threshold
    )

    dipole_angle_subset = dipole_angle[start_index:]
    excursion = _detect_excursion(dipole_angle_subset)
    reversal = _detect_reversal(
        dipole_angle_subset,
        mean_dipolarity,
    )

    if len(data.kin_dis_total) > 0:
        mean_viscous_dissipation = float(
            data.Ek * np.mean(data.kin_dis_total)
        )
    else:
        mean_viscous_dissipation = float("nan")

    if len(data.mag_dis_total) > 0:
        mean_ohmic_dissipation = float(
            2 *4/3 * np.pi * np.mean(data.mag_dis_total)
        )
    else:
        mean_ohmic_dissipation = float("nan")

    total_dissipation = (
        mean_viscous_dissipation
        + mean_ohmic_dissipation
    )

    if np.isfinite(total_dissipation) and total_dissipation > 0:
        ohmic_fraction = mean_ohmic_dissipation / total_dissipation
    else:
        ohmic_fraction = float("nan")

    if (
        np.isfinite(mean_viscous_dissipation)
        and mean_viscous_dissipation > 0
        and np.isfinite(mean_kinetic_energy)
    ):
        vel_dis_length_scale = float(
            # sqrt(2*V) 
            np.sqrt(
                 (2*4/3*np/pi)*_mean_after_start(data.kin_total,start_index,)
                / _mean_after_start(data.kin_dis_total,start_index,)
            )
        )
    else:
        vel_dis_length_scale = float("nan")

    if (
        np.isfinite(mean_ohmic_dissipation)
        and mean_ohmic_dissipation > 0
        and np.isfinite(mean_magnetic_energy)
    ):
        mag_dis_length_scale = float(
            np.sqrt(
                 np.sqrt(
                 (2*4/3*np/pi)*_mean_after_start(data.mag_total,start_index,)
                / _mean_after_start(data.mag_dis_total,start_index,)
            )
            )
        )
    else:
        mag_dis_length_scale = float("nan")


    return DynamoDiagnostics(
        averaging_start_index=start_index,
        mean_rossby=mean_rossby,
        physical_kinetic_energy=physical_kinetic_energy,
        thermal_perturbation=thermal_perturbation,
        dipole_angle=dipole_angle,
        initial_magnetic_energy=initial_magnetic_energy,
        mean_kinetic_energy=mean_kinetic_energy,
        mean_magnetic_energy=mean_magnetic_energy,
        mean_thermal_perturbation=mean_thermal_perturbation,
        mean_nusselt=mean_nusselt,
        magnetic_reynolds_number=magnetic_reynolds_number,
        elsasser_number=elsasser_number,
        mean_dipolarity=mean_dipolarity,
        relative_std_dipolarity=relative_std_dipolarity,
        dynamo_active=dynamo_active,
        excursion=excursion,
        reversal=reversal,
        mean_viscous_dissipation=mean_viscous_dissipation,
        mean_ohmic_dissipation=mean_ohmic_dissipation,
        ohmic_fraction=ohmic_fraction,
        vel_dis_length_scale=vel_dis_length_scale,
        mag_dis_length_scale=mag_dis_length_scale,
    )


def print_dynamo_diagnostics(
    data: TimeseriesData,
    diagnostics: DynamoDiagnostics,
) -> None:
    """Print a readable diagnostics report."""
    print("Input parameters:")
    print(f"Ek: {data.Ek}")
    print(f"Ra: {data.Ra}")
    print(f"q: {data.q}")
    print("-------------------------------")
    print("Output diagnostics:")
    print(
        "Magnetic Reynolds number Rm = "
        f"{diagnostics.magnetic_reynolds_number:.2f}"
    )
    print(
        "Time-averaged Rossby number = "
        f"{diagnostics.mean_rossby:.2e}"
    )
    print(
        "Time-averaged thermal perturbation = "
        f"{diagnostics.mean_thermal_perturbation:.2e}"
    )
    print(
        "Time-averaged kinetic energy = "
        f"{diagnostics.mean_kinetic_energy:.2e}"
    )
    print(
        "Time-averaged magnetic energy = "
        f"{diagnostics.mean_magnetic_energy:.2e}"
    )
    print(
        "Elsasser number Lambda = "
        f"{diagnostics.elsasser_number:.2e}"
    )
    print(
        "Time-averaged Nusselt number = "
        f"{diagnostics.mean_nusselt:.2e}"
    )
    print(
        "Time-averaged dipolarity = "
        f"{diagnostics.mean_dipolarity:.2e}"
    )
    print(
        "std(fdip)/mean(fdip) = "
        f"{diagnostics.relative_std_dipolarity:.2e}"
    )
    print(f"Dynamo active: {int(diagnostics.dynamo_active)}")
    print(f"Excursion: {int(diagnostics.excursion)}")
    print(f"Reversal: {int(diagnostics.reversal)}")
    print(
        "Viscous dissipation = "
        f"{diagnostics.mean_viscous_dissipation:.2e}"
    )
    print(
        "Ohmic dissipation = "
        f"{diagnostics.mean_ohmic_dissipation:.2e}"
    )
    print(
        "Fraction of ohmic dissipation = "
        f"{diagnostics.ohmic_fraction:.2e}"
    )
    print(
        "Typical velocity dissipation length scale = "
        f"{diagnostics.vel_dis_length_scale:.2e}"
    )
    print(
        "Typical magnetic disspation length scale = "
        f"{diagnostics.mag_dis_length_scale:.2e}"
    )






@dataclass
class HydroDiagnostics:
    """Time-averaged diagnostics for a hydro simulation."""

    averaging_start_index: int
    physical_kinetic_energy: np.ndarray
    thermal_perturbation: np.ndarray

    mean_kinetic_energy: float
    mean_thermal_perturbation: float
    mean_nusselt: float
    mean_viscous_dissipation: float


def compute_hydro_diagnostics(
    data: HydroTimeseriesData,
    averaging_fraction: float = 0.3,
) -> HydroDiagnostics:
    """Compute hydro time-series diagnostics."""
    start_index = _averaging_start_index(
        data.tkin,
        fraction=averaging_fraction,
    )

    physical_kinetic_energy = np.asarray(
        data.kin_total,
        dtype=float,
    ).copy()
    
    if data.Pm != 0 and data.Pr != 0:
        physical_kinetic_energy *= data.Ek / data.Pm

    thermal_perturbation = (
        3.0 / 5.0
        * (data.q * data.Ra) ** 2
        * data.tem_total
    )

    mean_kinetic_energy = _mean_after_start(
        physical_kinetic_energy,
        start_index,
    )

    mean_thermal_perturbation = _mean_after_start(
        thermal_perturbation,
        start_index,
    )

    mean_nusselt = _mean_after_start(
        data.nusselt,
        start_index,
    )

    if len(data.kin_dis_total) > 0:
        mean_viscous_dissipation = float(
            2 * 4/3*np.pi * data.Ek * np.mean(data.kin_dis_total)
        )
    else:
        mean_viscous_dissipation = float("nan")

    return HydroDiagnostics(
        averaging_start_index=start_index,
        physical_kinetic_energy=physical_kinetic_energy,
        thermal_perturbation=thermal_perturbation,
        mean_kinetic_energy=mean_kinetic_energy,
        mean_thermal_perturbation=mean_thermal_perturbation,
        mean_nusselt=mean_nusselt,
        mean_viscous_dissipation=mean_viscous_dissipation,
    )


def print_hydro_diagnostics(
    data: HydroTimeseriesData,
    diagnostics: HydroDiagnostics,
) -> None:
    """Print hydro diagnostics."""
    print("Input parameters:")
    print(f"Ek: {data.Ek}")
    print(f"Ra: {data.Ra}")
    print("-------------------------------")
    print("Output diagnostics:")
    print(
        "Time-averaged kinetic energy = "
        f"{diagnostics.mean_kinetic_energy:.2e}"
    )
    print(
        "Time-averaged thermal perturbation = "
        f"{diagnostics.mean_thermal_perturbation:.2e}"
    )
    print(
        "Time-averaged Nusselt number = "
        f"{diagnostics.mean_nusselt:.2e}"
    )
    print(
        "Viscous dissipation = "
        f"{diagnostics.mean_viscous_dissipation:.2e}"
    )
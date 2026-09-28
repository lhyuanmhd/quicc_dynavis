# field construction
#     ↓
# single-snapshot analysis
#     ↓
# multi-snapshot / CSV

"""
Azimuthal structure analysis for QUICC visualization snapshots.

This module analyzes the longitudinal organization of scalar quantities
constructed from QUICC HDF5 visualization files.

Main workflow
-------------
QUICC snapshot
    -> construct scalar field F(r, theta, phi)
    -> volume-weighted average over (r, theta)
    -> Fbar(phi)
    -> azimuthal Fourier decomposition
    -> amplitudes and phases of azimuthal modes
    -> multi-snapshot time series
    -> CSV output

Coordinate convention
---------------------
theta is colatitude:
    theta = 0       north pole
    theta = pi / 2  equator
    theta = pi      south pole

Fields are assumed to have shape:
    (nr, ntheta, nphi)

Fourier convention
------------------
NumPy FFT uses

    F(phi) = sum_m Fhat_m exp(+i m phi)

with

    Fhat_m = (1/N) sum_j F(phi_j) exp(-i m phi_j).

For a real mode written as

    F_m(phi) = A_m cos[m(phi - phi_m)],

the orientation is therefore

    phi_m = -arg(Fhat_m) / m.

For m > 1, phi_m is defined modulo 2*pi/m.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


# ============================================================================
# Supported scalar quantities
# ============================================================================

SUPPORTED_QUANTITIES = (
    "velocity_r",
    "velocity_theta",
    "velocity_phi",
    "velocity_magnitude",
    "velocity_z",
    "velocity_s",
    "vorticity_r",
    "vorticity_theta",
    "vorticity_phi",
    "vorticity_magnitude",
    "vorticity_z",
    "axial_helicity",
    "kinetic_helicity",
    "magnetic_r",
    "magnetic_theta",
    "magnetic_phi",
    "magnetic_magnitude",
    "magnetic_energy",
    "temperature",
)

def _trapezoid(y, *, x=None, axis=-1):
    """NumPy-version-compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)

    return _trapezoid(y, x=x, axis=axis)

def available_quantities() -> tuple[str, ...]:
    """Return scalar quantities supported by the analyzer."""
    return SUPPORTED_QUANTITIES

def snapshot_metadata(
    filename: str | Path,
) -> dict[str, str]:
    """
    Extract run and visualization-directory information
    from a snapshot path.
    """
    path = Path(filename)

    run = ""
    visu = ""

    for parent in path.parents:
        if parent.name.startswith("visu_") and not visu:
            visu = parent.name

        if parent.name.startswith("run") and not run:
            run = parent.name

    return {
        "run": run,
        "visu": visu,
        "filename": path.name,
        "path": str(path),
    }

def discover_snapshots(
    root: str | Path,
    *,
    pattern: str = "visState*.hdf5",
) -> list[Path]:
    """
    Recursively discover QUICC visualization snapshots.

    Examples
    --------
    root/
        runs/
            run001/
                visu_0001/
                    visState0000.hdf5
            run002/
                visu_0035/
                    visState0000.hdf5

    Notes
    -----
    The filename itself is not assumed to uniquely identify a snapshot.
    Physical simulation time stored in /run/time is used later for
    chronological ordering.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"Snapshot root does not exist: {root}"
        )

    files = list(root.rglob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found under {root}"
        )

    return sorted(files)

# ============================================================================
# Basic HDF5 utilities
# ============================================================================


def _read_scalar(dataset):
    """Read a scalar HDF5 dataset as a Python scalar."""
    value = dataset[()]

    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()

    if isinstance(value, np.generic):
        return value.item()

    return value


def read_time(h5: h5py.File) -> float:
    """Read simulation time from a QUICC visualization snapshot."""
    if "/run/time" not in h5:
        return np.nan

    return float(_read_scalar(h5["/run/time"]))


def read_timestep(h5: h5py.File) -> float:
    """Read simulation timestep from a QUICC visualization snapshot."""
    if "/run/timestep" not in h5:
        return np.nan

    return float(_read_scalar(h5["/run/timestep"]))


def read_grid(
    h5: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the physical grid.

    Returns
    -------
    r, theta, phi : ndarray
        One-dimensional coordinate arrays.
    """
    r = np.asarray(h5["/mesh/grid_r"][:])
    theta = np.asarray(h5["/mesh/grid_theta"][:])
    phi = np.asarray(h5["/mesh/grid_phi"][:])

    return r, theta, phi


# ============================================================================
# Grid sorting
# ============================================================================


def sort_grid_and_field(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort r, theta and phi into monotonically increasing order.

    The field is reordered consistently.

    Parameters
    ----------
    r, theta, phi
        One-dimensional coordinate arrays.

    field
        Scalar field with shape (nr, ntheta, nphi).

    Returns
    -------
    r_sorted, theta_sorted, phi_sorted, field_sorted
    """
    expected_shape = (len(r), len(theta), len(phi))

    if field.shape != expected_shape:
        raise ValueError(
            f"Field shape {field.shape} does not match "
            f"grid shape {expected_shape}."
        )

    r_idx = np.argsort(r)
    theta_idx = np.argsort(theta)
    phi_idx = np.argsort(phi)

    r_sorted = r[r_idx]
    theta_sorted = theta[theta_idx]
    phi_sorted = phi[phi_idx]

    field_sorted = field[
        np.ix_(r_idx, theta_idx, phi_idx)
    ]

    return (
        r_sorted,
        theta_sorted,
        phi_sorted,
        field_sorted,
    )


# ============================================================================
# Primitive-field readers
# ============================================================================


def _read_velocity(
    h5: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read spherical velocity components."""
    u_r = np.asarray(h5["/velocity/velocity_r"][:])
    u_theta = np.asarray(h5["/velocity/velocity_theta"][:])
    u_phi = np.asarray(h5["/velocity/velocity_phi"][:])

    return u_r, u_theta, u_phi


def _read_vorticity(
    h5: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read spherical vorticity components."""
    omega_r = np.asarray(
        h5["/velocity_curl/velocity_curl_r"][:]
    )
    omega_theta = np.asarray(
        h5["/velocity_curl/velocity_curl_theta"][:]
    )
    omega_phi = np.asarray(
        h5["/velocity_curl/velocity_curl_phi"][:]
    )

    return omega_r, omega_theta, omega_phi


def _read_magnetic(
    h5: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read spherical magnetic-field components."""
    B_r = np.asarray(h5["/magnetic/magnetic_r"][:])
    B_theta = np.asarray(h5["/magnetic/magnetic_theta"][:])
    B_phi = np.asarray(h5["/magnetic/magnetic_phi"][:])

    return B_r, B_theta, B_phi


# ============================================================================
# Scalar-field construction
# ============================================================================


def construct_scalar_field(
    h5: h5py.File,
    quantity: str,
) -> np.ndarray:
    """
    Construct a scalar diagnostic from a QUICC visualization snapshot.

    Parameters
    ----------
    h5
        Open HDF5 file.

    quantity
        Name of scalar quantity.

    Returns
    -------
    field : ndarray
        Scalar field with shape (nr, ntheta, nphi).

    Notes
    -----
    This function always returns the signed physical quantity where
    applicable. Absolute values are applied later by the azimuthal
    analyzer through ``use_absolute=True``.
    """
    if quantity not in SUPPORTED_QUANTITIES:
        available = ", ".join(SUPPORTED_QUANTITIES)
        raise ValueError(
            f"Unknown quantity '{quantity}'. "
            f"Available quantities: {available}"
        )

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    if quantity.startswith("velocity_"):
        u_r, u_theta, u_phi = _read_velocity(h5)

        if quantity == "velocity_r":
            return u_r

        if quantity == "velocity_theta":
            return u_theta

        if quantity == "velocity_phi":
            return u_phi
        
        if quantity == "velocity_s":
            ur, utheta, _ = _read_velocity(h5)

            theta = np.asarray(h5["/mesh/grid_theta"])
            sin_theta = np.sin(theta)[None, :, None]
            cos_theta = np.cos(theta)[None, :, None]

            return ur * sin_theta + utheta * cos_theta

        if quantity == "velocity_magnitude":
            return np.sqrt(
                u_r**2
                + u_theta**2
                + u_phi**2
            )

        if quantity == "velocity_z":
            theta = h5["/mesh/grid_theta"][:]
            cos_theta = np.cos(theta)[None, :, None]
            sin_theta = np.sin(theta)[None, :, None]

            return (
                u_r * cos_theta
                - u_theta * sin_theta
            )

    # ------------------------------------------------------------------
    # Vorticity
    # ------------------------------------------------------------------

    if quantity.startswith("vorticity_"):
        omega_r, omega_theta, omega_phi = _read_vorticity(h5)

        if quantity == "vorticity_r":
            return omega_r

        if quantity == "vorticity_theta":
            return omega_theta

        if quantity == "vorticity_phi":
            return omega_phi

        if quantity == "vorticity_magnitude":
            return np.sqrt(
                omega_r**2
                + omega_theta**2
                + omega_phi**2
            )

        if quantity == "vorticity_z":
            theta = h5["/mesh/grid_theta"][:]
            cos_theta = np.cos(theta)[None, :, None]
            sin_theta = np.sin(theta)[None, :, None]

            return (
                omega_r * cos_theta
                - omega_theta * sin_theta
            )

    # ------------------------------------------------------------------
    # Helicity
    # ------------------------------------------------------------------

    if quantity == "axial_helicity":
        u_r, u_theta, _ = _read_velocity(h5)

        omega_r, omega_theta, _ = _read_vorticity(h5)

        theta = h5["/mesh/grid_theta"][:]

        cos_theta = np.cos(theta)[None, :, None]
        sin_theta = np.sin(theta)[None, :, None]

        u_z = (
            u_r * cos_theta
            - u_theta * sin_theta
        )

        omega_z = (
            omega_r * cos_theta
            - omega_theta * sin_theta
        )

        return u_z * omega_z

    if quantity == "kinetic_helicity":
        u_r, u_theta, u_phi = _read_velocity(h5)

        omega_r, omega_theta, omega_phi = _read_vorticity(h5)

        return (
            u_r * omega_r
            + u_theta * omega_theta
            + u_phi * omega_phi
        )

    # ------------------------------------------------------------------
    # Magnetic field
    # ------------------------------------------------------------------

    if quantity.startswith("magnetic_"):
        B_r, B_theta, B_phi = _read_magnetic(h5)

        if quantity == "magnetic_r":
            return B_r

        if quantity == "magnetic_theta":
            return B_theta

        if quantity == "magnetic_phi":
            return B_phi

        if quantity == "magnetic_magnitude":
            return np.sqrt(
                B_r**2
                + B_theta**2
                + B_phi**2
            )

        if quantity == "magnetic_energy":
            return 0.5 * (
                B_r**2
                + B_theta**2
                + B_phi**2
            )

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    if quantity == "temperature":
        return np.asarray(
            h5["/temperature/temperature"][:]
        )

    raise RuntimeError(
        f"Quantity '{quantity}' is listed as supported "
        "but has no implementation."
    )


# ============================================================================
# Hemisphere selection
# ============================================================================


def _hemisphere_mask(
    theta: np.ndarray,
    hemisphere: str,
) -> np.ndarray:
    """
    Return theta mask for selected hemisphere.

    theta is colatitude.
    """
    hemisphere = hemisphere.lower()

    if hemisphere == "full":
        return np.ones(theta.shape, dtype=bool)

    if hemisphere == "north":
        return theta <= np.pi / 2

    if hemisphere == "south":
        return theta >= np.pi / 2

    raise ValueError(
        "hemisphere must be 'north', 'south', or 'full'."
    )


# ============================================================================
# Volume-weighted azimuthal profile
# ============================================================================


# def azimuthal_profile(
#     field: np.ndarray,
#     r: np.ndarray,
#     theta: np.ndarray,
#     *,
#     hemisphere: str = "north",
#     use_absolute: bool = False,
# ) -> np.ndarray:
    
def azimuthal_profile(
    field: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    *,
    hemisphere: str = "north",
    use_absolute: bool = False,
    z_max: float | None = None,
) -> np.ndarray:
    """
    Compute volume-weighted (r, theta) average at every longitude.

    For each phi,

        Fbar(phi) =
            integral F r^2 sin(theta) dr dtheta
            ---------------------------------
            integral r^2 sin(theta) dr dtheta

    Parameters
    ----------
    field
        Scalar field, shape (nr, ntheta, nphi).

    r, theta
        Monotonically increasing coordinates.

    hemisphere
        'north', 'south', or 'full'.

    use_absolute
        If True, analyze |field| instead of field.

    Returns
    -------
    profile : ndarray
        Volume-weighted azimuthal profile with shape (nphi,).
    """
    expected_shape = (
        len(r),
        len(theta),
        field.shape[2],
    )

    if field.shape[:2] != expected_shape[:2]:
        raise ValueError(
            "Field dimensions are inconsistent with r and theta."
        )

    mask = _hemisphere_mask(theta, hemisphere)

    theta_selected = theta[mask]
    field_selected = field[:, mask, :]

    if use_absolute:
        field_selected = np.abs(field_selected)

    rr = r[:, None, None]
    tt = theta_selected[None, :, None]

    jacobian = rr**2 * np.sin(tt)

    if z_max is not None:
        z = rr * np.cos(tt)
        spatial_mask = np.abs(z) <= z_max
        jacobian = jacobian * spatial_mask

    weighted_field = field_selected * jacobian

    # Integrate over theta first.
    numerator_theta = _trapezoid(
        weighted_field,
        x=theta_selected,
        axis=1,
    )

    denominator_theta = _trapezoid(
        jacobian,
        x=theta_selected,
        axis=1,
    )

    # Then integrate over radius.
    numerator = _trapezoid(
        numerator_theta,
        x=r,
        axis=0,
    )

    denominator = _trapezoid(
        denominator_theta,
        x=r,
        axis=0,
    )

    denominator = float(np.asarray(denominator).squeeze())

    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            "Invalid integration volume."
        )

    return numerator / denominator


# ============================================================================
# Fourier analysis
# ============================================================================

def fourier_decomposition(
    profile: np.ndarray,
    *,
    max_m: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Fourier-decompose an azimuthal profile.

    Parameters
    ----------
    profile
        One-dimensional uniformly sampled periodic profile.

    max_m
        Maximum azimuthal wavenumber returned.
        If None, return all rFFT modes.

    Returns
    -------
    result : dict
        Contains:

        m
            Azimuthal wavenumber.

        coefficient
            Complex Fourier coefficient.

        amplitude
            Real cosine-mode amplitude A_m.

        relative_to_mean
            Relative Fourier amplitude A_m / A_0.
            Returns NaN if the axisymmetric amplitude A_0
            is too small for the ratio to be meaningful.

        phase
            Complex phase arg(Fhat_m), radians.

        orientation
            Physical orientation -phase/m, radians.
            For m > 0, defined modulo 2*pi/m.

        orientation_deg
            Physical orientation in degrees.

        power_fraction
            Fraction of total non-axisymmetric Fourier power.
            The denominator includes all resolved m > 0 modes,
            even when max_m truncates the returned arrays.
    """
    profile = np.asarray(profile)

    if profile.ndim != 1:
        raise ValueError(
            "profile must be one-dimensional."
        )

    nphi = len(profile)

    if nphi < 2:
        raise ValueError(
            "At least two longitude samples are required."
        )

    # --------------------------------------------------------------
    # Fourier coefficients
    # --------------------------------------------------------------

    coeff_all = np.fft.rfft(profile) / nphi
    m_all = np.arange(len(coeff_all))

    # Real cosine-mode amplitudes.
    amplitude_all = 2.0 * np.abs(coeff_all)

    # m = 0 is not doubled.
    amplitude_all[0] = np.abs(coeff_all[0])

    # Nyquist mode is also not doubled for even nphi.
    if nphi % 2 == 0:
        amplitude_all[-1] = np.abs(coeff_all[-1])

    # --------------------------------------------------------------
    # Relative amplitude A_m / A_0
    # --------------------------------------------------------------

    relative_to_mean_all = np.full(
        len(coeff_all),
        np.nan,
        dtype=float,
    )

    amplitude_scale = np.max(amplitude_all)

    # Avoid meaningless ratios when the mean of a signed field
    # is zero or numerically very small.
    if (
        amplitude_scale > 0.0
        and amplitude_all[0] > 1.0e-12 * amplitude_scale
    ):
        relative_to_mean_all = (
            amplitude_all / amplitude_all[0]
        )

    # --------------------------------------------------------------
    # Phase and physical orientation
    # --------------------------------------------------------------

    phase_all = np.angle(coeff_all)

    orientation_all = np.full(
        len(coeff_all),
        np.nan,
        dtype=float,
    )

    if len(coeff_all) > 1:
        orientation_all[1:] = (
            -phase_all[1:] / m_all[1:]
        )

        # Each m-mode orientation is defined modulo 2*pi/m.
        orientation_all[1:] = np.mod(
            orientation_all[1:],
            2.0 * np.pi / m_all[1:],
        )

    orientation_deg_all = np.degrees(
        orientation_all
    )

    # --------------------------------------------------------------
    # Non-axisymmetric Fourier power
    # --------------------------------------------------------------

    power_fraction_all = np.full(
        len(coeff_all),
        np.nan,
        dtype=float,
    )

    if len(coeff_all) > 1:
        nonaxis_power = (
            np.abs(coeff_all[1:]) ** 2
        )

        total_nonaxis_power = np.sum(
            nonaxis_power
        )

        if total_nonaxis_power > 0.0:
            power_fraction_all[1:] = (
                nonaxis_power
                / total_nonaxis_power
            )

    # --------------------------------------------------------------
    # Truncate returned arrays if requested
    # --------------------------------------------------------------

    if max_m is None:
        stop = len(coeff_all)

    else:
        if max_m < 0:
            raise ValueError(
                "max_m must be non-negative."
            )

        stop = min(
            max_m + 1,
            len(coeff_all),
        )

    return {
        "m": m_all[:stop],
        "coefficient": coeff_all[:stop],
        "amplitude": amplitude_all[:stop],
        "relative_to_mean": relative_to_mean_all[:stop],
        "phase": phase_all[:stop],
        "orientation": orientation_all[:stop],
        "orientation_deg": orientation_deg_all[:stop],
        "power_fraction": power_fraction_all[:stop],
    }


# ============================================================================
# General scalar-field analysis
# ============================================================================


# def analyze_azimuthal_structure(
#     field: np.ndarray,
#     r: np.ndarray,
#     theta: np.ndarray,
#     phi: np.ndarray,
#     *,
#     hemisphere: str = "north",
#     use_absolute: bool = False,
#     max_m: int | None = 10,
# ) -> dict:
    
def analyze_azimuthal_structure(
    field: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    hemisphere: str = "north",
    use_absolute: bool = False,
    max_m: int | None = 10,
    z_max: float | None = None,
) -> dict:
    """
    Analyze azimuthal organization of an arbitrary scalar field.

    This is the central field-independent analysis routine.
    """
    (
        r,
        theta,
        phi,
        field,
    ) = sort_grid_and_field(
        r,
        theta,
        phi,
        field,
    )

    # Check that phi is approximately uniformly spaced.
    dphi = np.diff(phi)

    if len(dphi) > 1:
        if not np.allclose(
            dphi,
            np.mean(dphi),
            rtol=1.0e-5,
            atol=1.0e-10,
        ):
            raise ValueError(
                "phi grid is not uniformly spaced; "
                "FFT analysis is not valid."
            )

    # profile = azimuthal_profile(
    #     field,
    #     r,
    #     theta,
    #     hemisphere=hemisphere,
    #     use_absolute=use_absolute,
    # )

    profile = azimuthal_profile(
        field,
        r,
        theta,
        hemisphere=hemisphere,
        use_absolute=use_absolute,
        z_max=z_max,
    )

    fourier = fourier_decomposition(
        profile,
        max_m=max_m,
    )

    return {
        "r": r,
        "theta": theta,
        "phi": phi,
        "profile": profile,
        "hemisphere": hemisphere,
        "use_absolute": use_absolute,
        **fourier,
    }


# ============================================================================
# Single-snapshot analysis
# ============================================================================

def analyze_snapshots(
    filenames: Iterable[str | Path],
    diagnostics: dict,
    *,
    hemisphere: str = "north",
    modes: Iterable[int] = (1,),
    z_max: float | None = None,
) -> pd.DataFrame:
    """
    Analyze multiple diagnostics over multiple QUICC snapshots.

    Parameters
    ----------
    filenames
        Snapshot filenames.

    diagnostics
        Dictionary defining the diagnostics to analyze.

        Example
        -------
        diagnostics = {
            "ur_abs": {
                "quantity": "velocity_r",
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

        The dictionary key is used as the output-column prefix.

    hemisphere
        'north', 'south', or 'full'.

    modes
        Non-axisymmetric Fourier modes to save.

    Returns
    -------
    pandas.DataFrame
        One row per snapshot.
    """
    filenames = [Path(filename) for filename in filenames]
    modes = tuple(sorted(set(modes)))

    if not filenames:
        raise ValueError("No snapshot files supplied.")

    if not diagnostics:
        raise ValueError("No diagnostics supplied.")

    if not modes:
        raise ValueError("No Fourier modes supplied.")

    if min(modes) < 1:
        raise ValueError(
            "modes should contain positive "
            "non-axisymmetric mode numbers."
        )

    # Validate diagnostics before starting expensive HDF5 reads.
    for name, config in diagnostics.items():

        if "quantity" not in config:
            raise ValueError(
                f"Diagnostic '{name}' has no 'quantity'."
            )

        quantity = config["quantity"]

        if quantity not in SUPPORTED_QUANTITIES:
            raise ValueError(
                f"Diagnostic '{name}' uses unsupported "
                f"quantity '{quantity}'."
            )

    max_m = max(modes)

    rows = []

    for filename in filenames:

        print(f"Processing {filename} ...")

        # row = {
        #     "filename": filename.name,
        # }

        row = snapshot_metadata(filename)

        # Open each snapshot only once.
        with h5py.File(filename, "r") as h5:

            time = read_time(h5)
            timestep = read_timestep(h5)

            r, theta, phi = read_grid(h5)

            row["time"] = time
            row["timestep"] = timestep
        
            row["z_max"] = z_max

            for name, config in diagnostics.items():

                quantity = config["quantity"]
                use_absolute = config.get(
                    "absolute",
                    False,
                )

                field = construct_scalar_field(
                    h5,
                    quantity,
                )

                # result = analyze_azimuthal_structure(
                #     field,
                #     r,
                #     theta,
                #     phi,
                #     hemisphere=hemisphere,
                #     use_absolute=use_absolute,
                #     max_m=max_m,
                # )

                result = analyze_azimuthal_structure(
                    field,
                    r,
                    theta,
                    phi,
                    hemisphere=hemisphere,
                    use_absolute=use_absolute,
                    max_m=max_m,
                    z_max=z_max,
                )

                for mode in modes:

                    mode_data = _extract_mode_row(
                        result,
                        mode,
                        prefix=name,
                    )

                    row.update(mode_data)

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    # Sort chronologically using the physical simulation time.
    dataframe = dataframe.sort_values(
        "time",
        kind="stable",
    ).reset_index(drop=True)

    # metadata_columns = [
    #     "filename",
    #     "time",
    #     "timestep",
    # ]

    metadata_columns = [
        "run",
        "visu",
        "filename",
        "path",
        "time",
        "timestep",
        "hemisphere",
        "z_max",
    ]

    remaining_columns = [
        column
        for column in dataframe.columns
        if column not in metadata_columns
    ]

    return dataframe[
        metadata_columns
        + remaining_columns
    ]

# ============================================================================
# Extract one mode into a flat row
# ============================================================================


# def _extract_mode_row(
#     result: dict,
#     mode: int,
#     *,
#     prefix: str,
# ) -> dict:
#     """Extract one Fourier mode into flat CSV-friendly columns."""
#     m = result["m"]

#     index = np.where(m == mode)[0]

#     if len(index) == 0:
#         raise ValueError(
#             f"Mode m={mode} is not available."
#         )

#     i = index[0]

#     return {
#         f"{prefix}_m{mode}_amplitude":
#             float(result["amplitude"][i]),

#         f"{prefix}_m{mode}_power_fraction":
#             float(result["power_fraction"][i]),

#         f"{prefix}_m{mode}_phase_rad":
#             float(result["phase"][i]),

#         f"{prefix}_m{mode}_orientation_rad":
#             float(result["orientation"][i]),

#         f"{prefix}_m{mode}_orientation_deg":
#             float(result["orientation_deg"][i]),
#     }

def _extract_mode_row(
    result: dict,
    mode: int,
    *,
    prefix: str,
) -> dict:
    """Extract one Fourier mode into flat output columns."""

    index = np.where(result["m"] == mode)[0]

    if len(index) == 0:
        raise ValueError(
            f"Mode m={mode} is not available."
        )

    i = index[0]

    return {
        f"{prefix}_m{mode}_amplitude":
            float(result["amplitude"][i]),

        f"{prefix}_m{mode}_relative":
            float(result["relative_to_mean"][i]),

        f"{prefix}_m{mode}_power_fraction":
            float(result["power_fraction"][i]),

        f"{prefix}_m{mode}_phase_rad":
            float(result["phase"][i]),

        f"{prefix}_m{mode}_orientation_rad":
            float(result["orientation"][i]),

        f"{prefix}_m{mode}_orientation_deg":
            float(result["orientation_deg"][i]),
    }

# ============================================================================
# CSV output
# ============================================================================

def save_azimuthal_csv(
    dataframe: pd.DataFrame,
    filename: str | Path,
    *,
    compact: bool = True,
) -> None:
    """Save azimuthal time-series results to CSV."""

    filename = Path(filename)

    if compact:
        metadata = [
            "run",
            "visu",
            "time",
            "z_max",
        ]

        # diagnostics = [
        #     "ur_abs",
        #     "utheta_abs",
        #     "uphi_abs",
        #     "umag",
        #     "Hz_abs",
        #     "Hz",
        # ]
        # diagnostics = [
        #     "ur_abs",
        #     "utheta_abs",
        #     "uphi_abs",
        #     "umag",
        #     "wz_abs",
        #     "wz",
        #     "Hz_abs",
        #     "Hz",
        # ]

        # diagnostics = [
        #     "ur_abs",
        #     "utheta_abs",
        #     "uphi_abs",
        #     "umag",
        #     "wr_abs",
        #     "wtheta_abs",
        #     "wphi_abs",
        #     "wmag",
        #     "wz_abs",
        #     "wz",
        #     "Hz_abs",
        #     "Hz",
        # ]
        diagnostics = [
            # Velocity
            "ur_abs",
            "ur",
            "utheta_abs",
            "uphi_abs",
            "umag",

            # Cylindrical / axial velocity
            "us_abs",
            "us",
            "uz_abs",
            "uz",

            # Vorticity
            "wr_abs",
            "wtheta_abs",
            "wphi_abs",
            "wmag",
            "wz_abs",
            "wz",

            # Helicity
            "Hz_abs",
            "Hz",
        ]

        columns = metadata.copy()

        for diagnostic in diagnostics:
            columns.extend([
                f"{diagnostic}_m1_relative",
                f"{diagnostic}_m1_orientation_deg",
            ])

        columns = [
            column
            for column in columns
            if column in dataframe.columns
        ]

        output = dataframe[columns]

    else:
        output = dataframe

    output.to_csv(
        filename,
        index=False,
        float_format="%.6e",
    )

    print(
        f"Saved azimuthal analysis to {filename}"
    )

    
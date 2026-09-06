"""Diagnostics derived from QuICC spectral data."""

from dataclasses import dataclass

import numpy as np

from .spectra_utils import calculate_flow_degree


@dataclass
class SpectraDiagnostics:
    """Diagnostics derived from kinetic spectra."""

    flow_degree: float = np.nan
    degree_over_pi: float = np.nan
    local_rossby: float = np.nan


def compute_spectra_diagnostics(
    kinetic_spectrum,
    rossby_number=np.nan,
) -> SpectraDiagnostics:
    """Compute diagnostics derived from a kinetic l-spectrum."""

    kinetic_spectrum = np.asarray(
        kinetic_spectrum,
        dtype=float,
    )

    degrees = np.arange(
        len(kinetic_spectrum),
        dtype=float,
    )

    flow_degree = calculate_flow_degree(
        degrees,
        kinetic_spectrum,
    )

    degree_over_pi = (
        flow_degree / np.pi
    )

    if np.isfinite(rossby_number):
        local_rossby = (
            rossby_number
            * degree_over_pi
        )
    else:
        local_rossby = np.nan

    print("degrees.shape =", degrees.shape)
    print("kinetic_spectrum.shape =", kinetic_spectrum.shape)
    print("degrees[:5] =", degrees[:5])
    print("kinetic_spectrum[:5] =", kinetic_spectrum[:5])

    return SpectraDiagnostics(
        flow_degree=flow_degree,
        degree_over_pi=degree_over_pi,
        local_rossby=local_rossby,
    )
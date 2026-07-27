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
    degrees,
    kinetic_spectrum,
    rossby_number=np.nan,
) -> SpectraDiagnostics:
    """Compute diagnostics derived from a kinetic l-spectrum."""
    flow_degree = calculate_flow_degree(
        degrees,
        kinetic_spectrum,
    )

    degree_over_pi = flow_degree / np.pi

    if np.isfinite(rossby_number):
        local_rossby = rossby_number * degree_over_pi
    else:
        local_rossby = np.nan

    return SpectraDiagnostics(
        flow_degree=flow_degree,
        degree_over_pi=degree_over_pi,
        local_rossby=local_rossby,
    )


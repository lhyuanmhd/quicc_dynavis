import os
import matplotlib.pyplot as plt
import numpy as np
from .io import read_single_spectrum, avgSpectra_new 
import re


def calculate_flow_degree(
    degrees,
    kinetic_spectrum,
):
    """
    Calculate the energy-weighted spherical-harmonic degree.

    l_u = sum_l(l * E_l) / sum_l(E_l)
    """
    degrees = np.asarray(degrees, dtype=float)
    kinetic_spectrum = np.asarray(kinetic_spectrum, dtype=float)

    if degrees.shape != kinetic_spectrum.shape:
        raise ValueError(
            "degrees and kinetic_spectrum must have the same shape."
        )

    valid = (
        np.isfinite(degrees)
        & np.isfinite(kinetic_spectrum)
        & (kinetic_spectrum >= 0.0)
    )

    degrees = degrees[valid]
    kinetic_spectrum = kinetic_spectrum[valid]

    total_energy = np.sum(kinetic_spectrum)

    if total_energy <= 0.0:
        raise ValueError(
            "The kinetic spectrum must have positive total energy."
        )

    return float(
        np.sum(degrees * kinetic_spectrum) / total_energy
    )

def calculate_local_rossby(
    rossby_number,
    flow_degree,
):
    """
    Calculate the local Rossby number.

    Ro_l = Ro * l_u / pi
    """
    if rossby_number < 0.0:
        raise ValueError("rossby_number must be non-negative.")

    if flow_degree <= 0.0:
        raise ValueError("flow_degree must be positive.")

    return float(
        rossby_number * flow_degree / np.pi
    )


def calculate_spectral_flow_diagnostics(
    folder_file,
    rossby_number,
    *,
    mode="single",
    which="last",
    start_time=None,
    stop_time=None,
):
    """
    Calculate flow degree and local Rossby number from the kinetic spectrum.
    """
    if mode == "single":
        (
            degrees,
            _,
            total_l_spectrum,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = read_single_spectrum(
            folder_file,
            "kinetic",
            which=which,
        )

    elif mode == "average":
        if start_time is None or stop_time is None:
            raise ValueError(
                "start_time and stop_time are required "
                "for an averaged spectrum."
            )

        (
            degrees,
            _,
            total_l_spectrum,
            _,
            _,
            _,
            _,
            _,
        ) = avgSpectra_new(
            folder_file,
            "kinetic",
            start_time,
            stop_time,
        )

    else:
        raise ValueError("mode must be 'single' or 'average'")

    flow_degree = calculate_flow_degree(
        degrees,
        total_l_spectrum,
    )

    local_rossby = calculate_local_rossby(
        rossby_number,
        flow_degree,
    )

    return flow_degree, local_rossby

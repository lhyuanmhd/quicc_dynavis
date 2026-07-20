"""Data loading utilities for QuICC time-series diagnostics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import (
    F_conc_timeseries,
    get_boundary_conditions,
    get_parameters,
    get_resolution,
)
from .timeseries_utils import discover_run_folders, safe_conc_timeseries


@dataclass
class TimeseriesData:
    """Raw time-series data and simulation metadata for one QuICC case."""

    run_folders: list[str]

    tkin: np.ndarray
    kin_total: np.ndarray
    kin_tor: np.ndarray
    kin_pol: np.ndarray

    tmag: np.ndarray
    mag_total: np.ndarray
    mag_tor: np.ndarray
    mag_pol: np.ndarray

    ttem: np.ndarray
    tem_total: np.ndarray

    tnusselt: np.ndarray
    nusselt: np.ndarray

    tdip: np.ndarray
    fdip: np.ndarray
    g10: np.ndarray
    g11: np.ndarray
    h11: np.ndarray

    tkin_dis: np.ndarray
    kin_dis_total: np.ndarray
    kin_dis_tor: np.ndarray
    kin_dis_pol: np.ndarray

    tmag_dis: np.ndarray
    mag_dis_total: np.ndarray
    mag_dis_tor: np.ndarray
    mag_dis_pol: np.ndarray

    Ek: float
    Pm: float
    Pr: float
    q: float
    Ra: float
    Ro_input: float

    N: int
    M: int
    L: int

    bc_mag: str
    bc_temp: str
    bc_vel: str

    @property
    def has_dissipation(self) -> bool:
        """Return whether at least one dissipation time series is available."""
        return len(self.tkin_dis) > 0 or len(self.tmag_dis) > 0


def load_timeseries_data(case_dir) -> TimeseriesData:
    """Load all time-series files and metadata for one simulation case."""
    case_path = Path(case_dir)
    run_folders = discover_run_folders(case_path)

    if not run_folders:
        raise FileNotFoundError(
            f"No numeric run directories were found under {case_path}"
        )

    parameter_file = Path(run_folders[0]) / "parameters.cfg"

    if not parameter_file.is_file():
        raise FileNotFoundError(
            f"Parameter file does not exist: {parameter_file}"
        )

    tkin, kin_total, kin_tor, kin_pol = F_conc_timeseries(
        run_folders,
        "kinE",
    )
    tmag, mag_total, mag_tor, mag_pol = F_conc_timeseries(
        run_folders,
        "magE",
    )
    ttem, tem_total = F_conc_timeseries(
        run_folders,
        "temE",
    )
    tnusselt, nusselt = F_conc_timeseries(
        run_folders,
        "Nusselt",
    )
    tdip, fdip, g10, g11, h11 = F_conc_timeseries(
        run_folders,
        "Dip",
    )

    (
        tkin_dis,
        kin_dis_total,
        kin_dis_tor,
        kin_dis_pol,
    ) = safe_conc_timeseries(run_folders, "kinDis")

    (
        tmag_dis,
        mag_dis_total,
        mag_dis_tor,
        mag_dis_pol,
    ) = safe_conc_timeseries(run_folders, "magDis")

    Ek, Pm, Pr, q, Ra, Ro_input = get_parameters(
        str(parameter_file),
        "no",
    )

    N, M, L = get_resolution(
        str(parameter_file),
        "no",
    )

    bc_mag, bc_temp, bc_vel = get_boundary_conditions(
        str(parameter_file),
        "no",
    )

    return TimeseriesData(
        run_folders=run_folders,
        tkin=tkin,
        kin_total=kin_total,
        kin_tor=kin_tor,
        kin_pol=kin_pol,
        tmag=tmag,
        mag_total=mag_total,
        mag_tor=mag_tor,
        mag_pol=mag_pol,
        ttem=ttem,
        tem_total=tem_total,
        tnusselt=tnusselt,
        nusselt=nusselt,
        tdip=tdip,
        fdip=fdip,
        g10=g10,
        g11=g11,
        h11=h11,
        tkin_dis=tkin_dis,
        kin_dis_total=kin_dis_total,
        kin_dis_tor=kin_dis_tor,
        kin_dis_pol=kin_dis_pol,
        tmag_dis=tmag_dis,
        mag_dis_total=mag_dis_total,
        mag_dis_tor=mag_dis_tor,
        mag_dis_pol=mag_dis_pol,
        Ek=Ek,
        Pm=Pm,
        Pr=Pr,
        q=q,
        Ra=Ra,
        Ro_input=Ro_input,
        N=int(N),
        M=int(M),
        L=int(L),
        bc_mag=bc_mag,
        bc_temp=bc_temp,
        bc_vel=bc_vel,
    )

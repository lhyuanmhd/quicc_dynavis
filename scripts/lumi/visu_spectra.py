#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from quicc_dynavis import spectra
from quicc_dynavis.summary import (
    update_dynamo_summary_spectra,
)
from quicc_dynavis.timeseries_utils import (
    extract_ek_root,
    input_params_from_path,
)

from quicc_dynavis.io import (
    get_parameters,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_dir",
        nargs="?",
        default=".",
        help="Case dir (default: current directory)",
    )

    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()

    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Single spectra at the last timestep
    # ---------------------------------------------------------
    spectra.plot_spectra_km(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        mode="single",
        which="last",
        show=False,
        ref_scaling=True,
    )

    # ---------------------------------------------------------
    # Average spectra
    # ---------------------------------------------------------
    (
        _,
        _,
        _,
        spectra_diagnostics,
    ) = spectra.plot_spectra_km(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        mode="average",
        which="last",
        show=False,
        ref_scaling=True,
    )

    # ---------------------------------------------------------
    # Update spectral diagnostics in the existing summary CSV
    # ---------------------------------------------------------
    #Ek, q, Ra = input_params_from_path(str(case_dir))


    parameter_file = case_dir / "runs/run000/parameters.cfg"
    Ek, Pm, Pr, q, Ra, _ = get_parameters(
        filepath=str(parameter_file),
        output=None,
    )

    #Ek_root = extract_ek_root(case_dir)
    Ek_root = Path(extract_ek_root(case_dir))

    csv_path = (
        Ek_root
        / "diagnostics"
        / f"data_E_{Ek:.1e}.csv"
    )

    try:
        update_dynamo_summary_spectra(
            csv_path=csv_path,
            q=q,
            Ra=Ra,
            Ek=Ek,
            Pm=Pm,
            Pr=Pr,
            flow_degree=spectra_diagnostics.flow_degree,
        )

    except (FileNotFoundError, ValueError) as exc:
        print(
            "[WARNING] Could not update spectral diagnostics: "
            f"{exc}",
            file=sys.stderr,
        )

    print(f"[OK] spectra saved under {fig_dir}")


if __name__ == "__main__":
    main()
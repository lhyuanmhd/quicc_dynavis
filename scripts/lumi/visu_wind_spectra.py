#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from quicc_dynavis import spectra


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_dir",
        nargs="?",
        default=".",
        help="Case directory (default: current directory)",
    )

    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time for spectral averaging (default: 0.0)",
    )

    parser.add_argument(
        "--stop-time",
        type=float,
        default=10.0,
        help="Stop time for spectral averaging (default: 10.0)",
    )

    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()

    fig_dir = case_dir / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[INFO] Plotting averaged wind spectra "
        f"for t = [{args.start_time}, {args.stop_time}]"
    )

    _, _, save_path = spectra.plot_wind_spectra(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        start_time=args.start_time,
        stop_time=args.stop_time,
        show=False,
    )

    print(
        f"[OK] wind spectrum saved to {save_path}"
    )


if __name__ == "__main__":
    main()
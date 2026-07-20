#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import contextlib
import io as pyio  
import re


import matplotlib
matplotlib.use("Agg")


from quicc_dynavis.timeseries import plot_timeseries, plot_timeseries_dipolarity
from quicc_dynavis import io 

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "case_dir",
        nargs="?",
        default=".",
        help="Case dir (default: current directory)",
    )
    args = p.parse_args()

    case_dir = Path(args.case_dir).resolve()

    # ---- figures ----
    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_timeseries(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        show=False,
    )

    plot_timeseries_dipolarity(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        show=False,
    ) 
    
    print(f"[OK] timeseries saved under {fig_dir}")

    # ---- diagnostics / run_summary ----
    diag_dir = case_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
   
    runs_dir = case_dir / "runs"

    candidates = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and re.fullmatch(r'run0+', d.name)
    ])

    if not candidates:
        raise FileNotFoundError("No run0-style directory found")

    run0_dir = candidates[0]

    param_file = run0_dir / "parameters.cfg"
    summary_file = diag_dir / "run_summary.txt"

    buf = pyio.StringIO()
    with contextlib.redirect_stdout(buf):
        io.print_simulation_summary(str(param_file))

    summary_file.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[OK] simulation summary written to {summary_file}")


if __name__ == "__main__":
    main()


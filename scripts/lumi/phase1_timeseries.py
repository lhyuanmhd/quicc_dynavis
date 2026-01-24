#!/usr/bin/env python3
import argparse
from pathlib import Path

# Headless backend for LUMI/batch
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append('/scratch/project_465001528/lhyuan/codes/quicc_dynavis/src')
from quicc_dynavis.timeseries import plot_timeseries

def main():
    p = argparse.ArgumentParser(description="Phase1: plot timeseries for a single case directory (LUMI-safe).")
    p.add_argument("case_dir", type=str, help="Path to a case directory (e.g., .../N81L161)")
    args = p.parse_args()

    case_dir = Path(args.case_dir).resolve()
    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Your updated plot_timeseries should auto-discover runs/ or run*
    plot_timeseries(folderFile=str(case_dir), save_dir=str(fig_dir), show=False)

    print(f"[OK] Timeseries saved under: {fig_dir}")

if __name__ == "__main__":
    main()


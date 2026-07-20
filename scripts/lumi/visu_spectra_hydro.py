#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append('/scratch/project_465001528/lhyuan/codes/quicc_dynavis/src')
from quicc_dynavis import io, spectra, timeseries, fields_snapshot 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("case_dir", nargs="?", default=".", help="Case dir (default: current directory)")
    args = p.parse_args()

    case_dir = Path(args.case_dir).resolve()
    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    #single spectra  at last timestep
    spectra.plot_spectra_kt(folderFile=str(case_dir), save_dir = str(fig_dir),
        mode="single",   # "single" or "average": #mode = 'average', start_time=0, stop_time=1,
        #mode = "average",
        which="last",    # if single: "first", "last", or index (e.g. 10  
        show=False,
        ref_scaling= True                            
    )

    #Average spectra
    spectra.plot_spectra_kt(folderFile=str(case_dir), save_dir = str(fig_dir),
        #mode="single",   # "single" or "average": #mode = 'average', start_time=0, stop_time=1,
        mode = "average",
        which="last",    # if single: "first", "last", or index (e.g. 10  
        show=False,
        ref_scaling= True                        
    )
    

    print(f"[OK] timeseries saved under {fig_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import glob
import sys

# ---- user input ----
CASE_DIR = "/scratch/project_465001528/lhyuan/IlessDyn/CattaneoHuges/E1e-5/q_1/Ra1e3/stfromE1e-4Ra1e3q5/N81L161"

# ---- find runs ----
#run_dirs = sorted(glob.glob(os.path.join(CASE_DIR, "runs", "run*")))
from quicc_dynavis.utils.lumi import discover_valid_runs
run_dirs = discover_valid_runs(CASE_DIR)


print("Found runs:")
for r in run_dirs:
    print("  ", r)

if len(run_dirs) == 0:
    raise RuntimeError("No run directories found")

# ---- import dynavis tools ----
from quicc_dynavis.timeseries import F_conc_timeseries

# ---- test read ----
t, kinE, _, _ = F_conc_timeseries(run_dirs, "kinE")

print("Timeseries loaded:")
print("  t range :", t.min(), t.max())
print("  kinE min/max :", kinE.min(), kinE.max())


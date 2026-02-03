#!/usr/bin/env bash

#remove unnecessary files
#rm run.log
#rm slurm*
rm *gxl
rm OUT_stdout
rm state4Visu.hdf5  


python - <<'PY'
import re
from pathlib import Path
import h5py

d = Path(".").resolve()
vis = d / "visState0000.hdf5"
if not vis.exists():
    raise SystemExit(0)

states = sorted(d.glob("state*.hdf5"))
if not states:
    raise SystemExit(0)

def key(p):
    m = re.search(r"state(\d+)\.hdf5$", p.name)
    return int(m.group(1)) if m else -1

st = max(states, key=key)

with h5py.File(st, "r") as fs:
    t = fs["/run/time"][()]
with h5py.File(vis, "r+") as fv:
    old = fv["/run/time"][()]
    if float(old) == 0.0 and float(t) != 0.0:
        fv["/run/time"][...] = t
        print(f"[PATCH] visState /run/time: {old} -> {t} (from {st.name})")
PY


# extract fields we are intertested
python /scratch/project_465001528/lhyuan/codes/quicc_dynavis/scripts/lumi/extract_fields_curl.py

#remove 
#rm visState0000.hdf5

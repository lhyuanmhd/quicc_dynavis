#!/usr/bin/env bash
set -euo pipefail

LOG="viz_$(date +%Y%m%d_%H%M%S).log"
#LOG="viz_$(date).log"
exec > >(tee -a "$LOG") 2>&1

# Path to your quicc_dynavis repo
DYNAVIS="/scratch/project_465001528/lhyuan/codes/quicc_dynavis"

#activate environment

#visu timeseries 
python "$DYNAVIS/scripts/lumi/visu_timeseries.py" .

#visu spectra single latest
python "$DYNAVIS/scripts/lumi/visu_spectra.py" .

echo "== **************************** =="
echo "== $(date) running viz in $(pwd) =="


#visu snapshots
python $DYNAVIS/scripts/lumi/visu_snapshots.py --latest .

# other options
#python $DYNAVIS/scripts/lumi/visu_snapshots.py .
#python $DYNAVIS/scripts/lumi/visu_snapshots.py --nlatest 3 .
#python $DYNAVIS/scripts/lumi/visu_snapshots.py --run run3 --tags 0040,0027,0011 .
#python "$DYNAVIS/scripts/lumi/visu_snapshots.py" --run run1 --tag 0003  .
#python "$DYNAVIS/scripts/lumi/visu_snapshots.py" --run run3 --tag 0011  .

#!/usr/bin/env bash
# viz.sh
# Batch visualization driver for LUMI:
# - Timeseries and spectra are run once for the case directory.
# - Snapshots can be generated for missing visu_* folders across run*/visu_****.
# - You can also target a specific run/visu index.
#
# New:
# - --udotgradu : plot |u·∇u| magnitude (equatorial + meridional)
# - --coriolis  : plot |ẑ × u| magnitude (Coriolis force)
# - --both-forces: plot both quantities in a combined 2x2 panel
#
# Examples:
#   bash viz.sh --timeseries --spectra
#   bash viz.sh --snapshots --missing
#   bash viz.sh --udotgradu --missing
#   bash viz.sh --coriolis --missing
#   bash viz.sh --both-forces --missing
#   bash viz.sh --snapshots --udotgradu --missing
#   bash viz.sh --udotgradu --run 3 --visu 0020 --force

set -euo pipefail
#set -x

#LOG="viz_$(date +%Y%m%d_%H%M%S).log"
LOG="viz.log"
exec > >(tee -a "$LOG") 2>&1

# Path to your quicc_dynavis repo
DYNAVIS="/scratch/project_465001528/lhyuan/codes/quicc_dynavis"

DO_TIMESERIES=0
DO_SPECTRA=0
DO_SNAPSHOTS=0
DO_UDOTGRADU=0
DO_CORIOLIS=0
DO_BOTH_FORCES=0

MODE_MISSING=0          # plot only missing figures (recommended)
FORCE=0                 # overwrite existing figures
RUN_FILTER=""           # e.g. "3" for run3
VISU_FILTER=""          # e.g. "0020" for visu_0020
SLEEP_SEC="${SLEEP_SEC:-0}"  # optional tiny sleep between plots
MODE="udotgradu"        # default mode for the plotting script (udotgradu, coriolis, combined)

usage() {
  cat <<EOF
Usage:
  bash viz.sh [--timeseries] [--spectra] [--snapshots] [--udotgradu] [--coriolis] [--both-forces] [--missing] [--force] [--run N] [--visu YYYY]

Examples:
  # Run timeseries + spectra for the current case directory
  bash viz.sh --timeseries --spectra

  # Plot snapshots for all missing visu folders under run*/visu_****
  bash viz.sh --snapshots --missing

  # Plot |u dot grad u| for all missing visu folders
  bash viz.sh --udotgradu --missing

  # Plot Coriolis force |ẑ × u| for all missing visu folders
  bash viz.sh --coriolis --missing

  # Plot both forces in a combined 2x2 panel
  bash viz.sh --both-forces --missing

  # Plot both snapshots AND |u dot grad u| for all missing visu folders
  bash viz.sh --snapshots --udotgradu --missing

  # Plot a single snapshot: run3/visu_0020
  bash viz.sh --snapshots --run 3 --visu 0020

  # Plot |u dot grad u| for a single snapshot: run3/visu_0020
  bash viz.sh --udotgradu --run 3 --visu 0020

  # Plot Coriolis force for a single snapshot: run3/visu_0020
  bash viz.sh --coriolis --run 3 --visu 0020

  # Force re-plot (overwrite) for run3/visu_0020 (multiple if requested)
  bash viz.sh --snapshots --udotgradu --coriolis --run 3 --visu 0020 --force
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeseries) DO_TIMESERIES=1; shift ;;
    --spectra)    DO_SPECTRA=1; shift ;;
    --snapshots)  DO_SNAPSHOTS=1; shift ;;
    --udotgradu)  DO_UDOTGRADU=1; shift ;;
    --coriolis)   DO_CORIOLIS=1; shift ;;
    --both-forces) DO_BOTH_FORCES=1; shift ;;
    --missing)    MODE_MISSING=1; shift ;;
    --force)      FORCE=1; shift ;;
    --run)        RUN_FILTER="$2"; shift 2 ;;
    --visu)       VISU_FILTER="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
  esac
done

echo "== $(date) running viz in $(pwd) =="

CASE_DIR="$(pwd)"

# Determine where run* directories live:
# - If CASE_DIR has a "runs/" directory, use CASE_DIR as case root (python will find runs/run*).
# - If CASE_DIR itself already contains run* directories, treat CASE_DIR as the run-root (old layout).
RUNS_ROOT=""
if [[ -d "$CASE_DIR/runs" ]]; then
  RUNS_ROOT="$CASE_DIR/runs"
elif compgen -G "$CASE_DIR/run[0-9]*" > /dev/null; then
  RUNS_ROOT="$CASE_DIR"
else
  RUNS_ROOT="$CASE_DIR"
fi

# Timeseries and spectra (run once at case level)
if [[ "$DO_TIMESERIES" == "1" ]]; then
  python "$DYNAVIS/scripts/lumi/visu_timeseries.py" "$CASE_DIR"
fi

if [[ "$DO_SPECTRA" == "1" ]]; then
  python "$DYNAVIS/scripts/lumi/visu_spectra.py" "$CASE_DIR"
fi

# Nothing to do?
if [[ "$DO_SNAPSHOTS" == "0" && "$DO_UDOTGRADU" == "0" && "$DO_CORIOLIS" == "0" && "$DO_BOTH_FORCES" == "0" ]]; then
  echo "[INFO] Nothing selected. Use --snapshots, --udotgradu, --coriolis, or --both-forces."
  exit 0
fi

# Check if the new plotting script exists
PLOT_SCRIPT="$DYNAVIS/scripts/lumi/visu_udotgradu_coriolis.py"
if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "[ERROR] Plotting script not found: $PLOT_SCRIPT"
  echo "Please ensure the visu_udotgradu_coriolis.py script exists."
  exit 1
fi

# Snapshots / u·∇u / Coriolis: plot missing (or selected) visu folders
mkdir -p "$CASE_DIR/figures"
shopt -s nullglob

# Iterate runs
for run_dir in "$RUNS_ROOT"/run[0-9]*; do
  [[ -d "$run_dir" ]] || continue
  run_base="$(basename "$run_dir")"     # run3
  run_num="${run_base#run}"            # 3

  if [[ -n "$RUN_FILTER" && "$run_num" != "$RUN_FILTER" ]]; then
    continue
  fi

  # Iterate visu directories
  for visu_dir in "$run_dir"/visu_[0-9][0-9][0-9][0-9]; do
    [[ -d "$visu_dir" ]] || continue
    visu_base="$(basename "$visu_dir")"     # visu_0020
    visu_idx="${visu_base#visu_}"           # 0020

    if [[ -n "$VISU_FILTER" && "$visu_idx" != "$VISU_FILTER" ]]; then
      continue
    fi

    # Require the extracted visualization data
    if [[ ! -f "$visu_dir/vis_fields_0000.npz" ]] && ! compgen -G "$visu_dir/vis_fields*.npz" > /dev/null; then
      echo "[SKIP] $visu_dir (missing vis_fields*.npz)"
      continue
    fi

    # -----------------------------
    # 1) Original snapshots figure
    # -----------------------------
    if [[ "$DO_SNAPSHOTS" == "1" ]]; then
      pattern_snap="$CASE_DIR/figures/*_${run_base}_${visu_idx}_snapshots.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && compgen -G "$pattern_snap" > /dev/null; then
        echo "[SKIP] $visu_dir (snapshots exist: $pattern_snap)"
      else
        echo "[PLOT] $run_base/$visu_base -> snapshots"
        python "$DYNAVIS/scripts/lumi/visu_snapshots.py" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    # -----------------------------
    # 2) |u·∇u| figure
    # -----------------------------
    if [[ "$DO_UDOTGRADU" == "1" ]]; then
      pattern_adv="$CASE_DIR/figures/*_${run_base}_${visu_idx}_udotgradu.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && compgen -G "$pattern_adv" > /dev/null; then
        echo "[SKIP] $visu_dir (udotgradu exist: $pattern_adv)"
      else
        echo "[PLOT] $run_base/$visu_base -> |u·∇u|"
        python "$PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --mode udotgradu \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    # -----------------------------
    # 3) Coriolis force figure
    # -----------------------------
    if [[ "$DO_CORIOLIS" == "1" ]]; then
      pattern_cor="$CASE_DIR/figures/*_${run_base}_${visu_idx}_coriolis.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && compgen -G "$pattern_cor" > /dev/null; then
        echo "[SKIP] $visu_dir (coriolis exist: $pattern_cor)"
      else
        echo "[PLOT] $run_base/$visu_base -> |ẑ × u| (Coriolis)"
        python "$PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --mode coriolis \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    # -----------------------------
    # 4) Combined forces figure
    # -----------------------------
    if [[ "$DO_BOTH_FORCES" == "1" ]]; then
      pattern_comb="$CASE_DIR/figures/*_${run_base}_${visu_idx}_combined.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && compgen -G "$pattern_comb" > /dev/null; then
        echo "[SKIP] $visu_dir (combined forces exist: $pattern_comb)"
      else
        echo "[PLOT] $run_base/$visu_base -> Combined forces (|u·∇u| + |ẑ × u|)"
        python "$PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --mode combined \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$SLEEP_SEC" != "0" ]]; then
      sleep "$SLEEP_SEC"
    fi

  done
done

echo "[DONE] All requested plots processed."
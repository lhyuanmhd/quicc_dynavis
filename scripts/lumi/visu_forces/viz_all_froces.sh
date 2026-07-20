#!/usr/bin/env bash
# viz.sh
# Batch visualization driver for LUMI:
# - Timeseries and spectra are run once for the case directory.
# - Forces (inertia, Coriolis, viscous, Lorentz, buoyancy) can be plotted.
#
# Examples:
#   bash viz.sh --timeseries --spectra
#   bash viz.sh --all-forces --missing
#   bash viz.sh --inertia --missing
#   bash viz.sh --coriolis --run 3 --visu 0020 --force
#   bash viz.sh --all-forces --snapshots --missing

set -euo pipefail

LOG="viz.log"
exec > >(tee -a "$LOG") 2>&1

# Resolve the repository root from this script's location
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DYNAVIS="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

# Flags
DO_TIMESERIES=0
DO_SPECTRA=0
DO_SNAPSHOTS=0
DO_ALL_FORCES=0
DO_INERTIA=0
DO_CORIOLIS=0
DO_VISCOUS=0
DO_LORENTZ=0
DO_BUOYANCY=0

MODE_MISSING=0          # plot only missing figures
FORCE=0                 # overwrite existing figures
RUN_FILTER=""           # e.g. "3" for run3
VISU_FILTER=""          # e.g. "0020" for visu_0020
SLEEP_SEC="${SLEEP_SEC:-0}"  # optional tiny sleep between plots

usage() {
  cat <<EOF
Usage:
  bash viz.sh [OPTIONS]

Options for what to plot:
  --timeseries          Plot timeseries (run once per case)
  --spectra             Plot spectra (run once per case)
  --snapshots           Plot original snapshot figures
  --all-forces          Plot all forces in 2x3 panel
  --inertia             Plot inertia force |(u·∇)u| only
  --coriolis            Plot Coriolis force |ẑ×u| only
  --viscous             Plot viscous force |E∇²u| only
  --lorentz             Plot Lorentz force |(∇×B)×B| only
  --buoyancy            Plot buoyancy force |qRaTr| only

Selection options:
  --missing             Plot only missing figures (recommended)
  --force               Overwrite existing figures
  --run N               Filter by run number (e.g., 3 for run3)
  --visu YYYY           Filter by visu tag (e.g., 0020)

Examples:
  # Run timeseries + spectra
  bash viz.sh --timeseries --spectra

  # Plot all forces for all missing snapshots
  bash viz.sh --all-forces --missing

  # Plot inertia force for run3/visu_0020
  bash viz.sh --inertia --run 3 --visu 0020

  # Plot all forces and original snapshots
  bash viz.sh --all-forces --snapshots --missing
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeseries) DO_TIMESERIES=1; shift ;;
    --spectra)    DO_SPECTRA=1; shift ;;
    --snapshots)  DO_SNAPSHOTS=1; shift ;;
    --all-forces) DO_ALL_FORCES=1; shift ;;
    --inertia)    DO_INERTIA=1; shift ;;
    --coriolis)   DO_CORIOLIS=1; shift ;;
    --viscous)    DO_VISCOUS=1; shift ;;
    --lorentz)    DO_LORENTZ=1; shift ;;
    --buoyancy)   DO_BUOYANCY=1; shift ;;
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

# Determine where run* directories live
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
ANY_FORCE=$((DO_ALL_FORCES + DO_INERTIA + DO_CORIOLIS + DO_VISCOUS + DO_LORENTZ + DO_BUOYANCY))
if [[ "$DO_SNAPSHOTS" == "0" && "$ANY_FORCE" == "0" ]]; then
  echo "[INFO] Nothing selected. Use --snapshots and/or force options."
  exit 0
fi

# Check if forces plotting script exists
FORCE_PLOT_SCRIPT="$DYNAVIS/scripts/lumi/visu_forces.py"
if [[ ! -f "$FORCE_PLOT_SCRIPT" && "$ANY_FORCE" -gt 0 ]]; then
  echo "[ERROR] Forces plotting script not found: $FORCE_PLOT_SCRIPT"
  exit 1
fi

mkdir -p "$CASE_DIR/figures"
shopt -s nullglob

# Function to check if figure exists
figure_exists() {
  local pattern="$1"
  compgen -G "$pattern" > /dev/null
}

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

    # Check for extracted data
    if [[ ! -f "$visu_dir/vis_fields_forces.npz" ]] && \
       [[ ! -f "$visu_dir/vis_fields_0000.npz" ]] && \
       ! compgen -G "$visu_dir/vis_fields*.npz" > /dev/null; then
      echo "[SKIP] $visu_dir (missing vis_fields*.npz)"
      continue
    fi

    # -----------------------------
    # Original snapshots figure
    # -----------------------------
    if [[ "$DO_SNAPSHOTS" == "1" ]]; then
      pattern_snap="$CASE_DIR/figures/*_${run_base}_${visu_idx}_snapshots.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern_snap"; then
        echo "[SKIP] $visu_dir (snapshots exist)"
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
    # All forces in one panel
    # -----------------------------
    if [[ "$DO_ALL_FORCES" == "1" ]]; then
      pattern_all="$CASE_DIR/figures/*_${run_base}_${visu_idx}_all_forces.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern_all"; then
        echo "[SKIP] $visu_dir (all forces exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> all forces"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --all \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    # -----------------------------
    # Individual forces
    # -----------------------------
    if [[ "$DO_INERTIA" == "1" ]]; then
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_inertia.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (inertia exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> inertia"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --inertia \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_CORIOLIS" == "1" ]]; then
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_coriolis.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (coriolis exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> coriolis"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --coriolis \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_VISCOUS" == "1" ]]; then
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_viscous.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (viscous exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> viscous"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --viscous \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_LORENTZ" == "1" ]]; then
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_lorentz.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (lorentz exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> lorentz"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --lorentz \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_BUOYANCY" == "1" ]]; then
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_buoyancy.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (buoyancy exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> buoyancy"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --buoyancy \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$SLEEP_SEC" != "0" ]]; then
      sleep "$SLEEP_SEC"
    fi

  done
done

echo "[DONE]"

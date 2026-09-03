#!/usr/bin/env bash
# viz_forces.sh
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
#   bash viz.sh --all-forces --common-scale 1e-3     # all forces with same vmax
#   bash viz.sh --coriolis --scale-from lorentz      # coriolis with lorentz scale

set -euo pipefail

LOG="viz.log"
exec > >(tee -a "$LOG") 2>&1

# Path to your quicc_dynavis repo
#DYNAVIS="/scratch/project_465001528/lhyuan/codes/quicc_dynavis"

DYNAVIS="/users/yuanlong/quicc_dynavis"
# Flags
DO_TIMESERIES=0
DO_SPECTRA=0
DO_SNAPSHOTS=0
DO_WINDS=0
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

# New scale options
COMMON_SCALE=""         # e.g. "1e-3" for fixed vmax
SCALE_FROM=""           # e.g. "coriolis" to use coriolis scale

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

Scale options (for force plots):
  --common-scale VALUE  Use fixed vmax for comparison (e.g., 1e-3)
  --scale-from FORCE    Use vmax from this force (inertia/coriolis/viscous/lorentz/buoyancy)

Examples:
  # Run timeseries + spectra
  bash viz.sh --timeseries --spectra

  # Plot all forces for all missing snapshots
  bash viz.sh --all-forces --missing

  # Plot all forces with same scale
  bash viz.sh --all-forces --common-scale 1e-3
  bash viz.sh --all-forces --scale-from coriolis

  # Plot inertia force with Coriolis scale
  bash viz.sh --inertia --scale-from coriolis

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
    --winds)      DO_WINDS=1; shift ;;
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
    --common-scale) COMMON_SCALE="$2"; shift 2 ;;
    --scale-from) SCALE_FROM="$2"; shift 2 ;;
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
#ANY_FORCE=$((DO_ALL_FORCES + DO_INERTIA + DO_CORIOLIS + DO_VISCOUS + DO_LORENTZ + DO_BUOYANCY))
#if [[ "$DO_SNAPSHOTS" == "0" && "$ANY_FORCE" == "0" ]]; then
#  echo "[INFO] Nothing selected. Use --snapshots and/or force options."
#  exit 0
#fi
# Nothing to do?
ANY_FORCE=$((DO_ALL_FORCES + DO_INERTIA + DO_CORIOLIS + DO_VISCOUS + DO_LORENTZ + DO_BUOYANCY))

if [[ "$DO_SNAPSHOTS" == "0" \
   && "$DO_WINDS" == "0" \
   && "$ANY_FORCE" == "0" ]]; then

  echo "[INFO] Nothing selected. Use --snapshots, --winds and/or force options."
  exit 0
fi

# Check if forces plotting script exists
FORCE_PLOT_SCRIPT="$DYNAVIS/scripts/lumi/visu_forces/visu_forces.py"
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

# Build scale arguments for force plots
build_scale_args() {
  local args=""
  if [[ -n "$COMMON_SCALE" ]]; then
    args="--common-vmax $COMMON_SCALE"
  elif [[ -n "$SCALE_FROM" ]]; then
    # Convert short name to full force name
    case "$SCALE_FROM" in
      inertia)   args="--vmax-from inertia_magnitude" ;;
      coriolis)  args="--vmax-from coriolis_magnitude" ;;
      viscous)   args="--vmax-from viscous_magnitude" ;;
      lorentz)   args="--vmax-from lorentz_magnitude" ;;
      buoyancy)  args="--vmax-from buoyancy_magnitude" ;;
      *) echo "[WARN] Unknown force: $SCALE_FROM, ignoring scale"; args="" ;;
    esac
  fi
  echo "$args"
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
    # Wind visualization
    # -----------------------------
    if [[ "$DO_WINDS" == "1" ]]; then
      pattern_winds="$CASE_DIR/figures/*_${run_base}_${visu_idx}_winds.png"

      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern_winds"; then
        echo "[SKIP] $visu_dir (wind snapshots exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> wind snapshots"
        python "$DYNAVIS/scripts/lumi/winds/visu_snapshots_winds.py" \
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
      # Build scale suffix for pattern matching
      scale_suffix=""
      if [[ -n "$COMMON_SCALE" ]]; then
        scale_suffix="_vmax${COMMON_SCALE}"
      elif [[ -n "$SCALE_FROM" ]]; then
        scale_suffix="_scale_from_${SCALE_FROM}"
      fi
      
      pattern_all="$CASE_DIR/figures/*_${run_base}_${visu_idx}_all_forces${scale_suffix}.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern_all"; then
        echo "[SKIP] $visu_dir (all forces${scale_suffix} exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> all forces${scale_suffix}"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --all \
          $(build_scale_args) \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    # -----------------------------
    # Individual forces
    # -----------------------------
    if [[ "$DO_INERTIA" == "1" ]]; then
      scale_suffix=""
      if [[ -n "$COMMON_SCALE" ]]; then
        scale_suffix="_vmax${COMMON_SCALE}"
      elif [[ -n "$SCALE_FROM" ]]; then
        scale_suffix="_scale_from_${SCALE_FROM}"
      fi
      
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_inertia${scale_suffix}.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (inertia${scale_suffix} exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> inertia${scale_suffix}"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --inertia \
          $(build_scale_args) \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_CORIOLIS" == "1" ]]; then
      scale_suffix=""
      if [[ -n "$COMMON_SCALE" ]]; then
        scale_suffix="_vmax${COMMON_SCALE}"
      elif [[ -n "$SCALE_FROM" ]]; then
        scale_suffix="_scale_from_${SCALE_FROM}"
      fi
      
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_coriolis${scale_suffix}.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (coriolis${scale_suffix} exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> coriolis${scale_suffix}"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --coriolis \
          $(build_scale_args) \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_VISCOUS" == "1" ]]; then
      scale_suffix=""
      if [[ -n "$COMMON_SCALE" ]]; then
        scale_suffix="_vmax${COMMON_SCALE}"
      elif [[ -n "$SCALE_FROM" ]]; then
        scale_suffix="_scale_from_${SCALE_FROM}"
      fi
      
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_viscous${scale_suffix}.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (viscous${scale_suffix} exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> viscous${scale_suffix}"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --viscous \
          $(build_scale_args) \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_LORENTZ" == "1" ]]; then
      scale_suffix=""
      if [[ -n "$COMMON_SCALE" ]]; then
        scale_suffix="_vmax${COMMON_SCALE}"
      elif [[ -n "$SCALE_FROM" ]]; then
        scale_suffix="_scale_from_${SCALE_FROM}"
      fi
      
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_lorentz${scale_suffix}.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (lorentz${scale_suffix} exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> lorentz${scale_suffix}"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --lorentz \
          $(build_scale_args) \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$DO_BUOYANCY" == "1" ]]; then
      scale_suffix=""
      if [[ -n "$COMMON_SCALE" ]]; then
        scale_suffix="_vmax${COMMON_SCALE}"
      elif [[ -n "$SCALE_FROM" ]]; then
        scale_suffix="_scale_from_${SCALE_FROM}"
      fi
      
      pattern="$CASE_DIR/figures/*_${run_base}_${visu_idx}_buoyancy${scale_suffix}.png"
      if [[ "$FORCE" == "0" && "$MODE_MISSING" == "1" ]] && figure_exists "$pattern"; then
        echo "[SKIP] $visu_dir (buoyancy${scale_suffix} exist)"
      else
        echo "[PLOT] $run_base/$visu_base -> buoyancy${scale_suffix}"
        python "$FORCE_PLOT_SCRIPT" \
          "$CASE_DIR" \
          --run "$run_base" \
          --tag "$visu_idx" \
          --buoyancy \
          $(build_scale_args) \
          $([[ "$FORCE" == "1" ]] && echo "--force")
      fi
    fi

    if [[ "$SLEEP_SEC" != "0" ]]; then
      sleep "$SLEEP_SEC"
    fi

  done
done
echo "[DONE]"
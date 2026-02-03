#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./prepare_visu_batches.sh [STATE_GLOB]
# Example:
#   ./prepare_visu_batches.sh "state*.hdf5"
#   ./prepare_visu_batches.sh "state00[1-4][0-9].hdf5"   # subset
STATE_GLOB="${1:-state*.hdf5}"

REQ_FILES=("Lumi_visu.slurm" "parameters.cfg" "postprocess_visu.sh")

# sanity check required files
for f in "${REQ_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "[ERROR] missing required file: $f"; exit 2; }
done

shopt -s nullglob
states=($STATE_GLOB)
shopt -u nullglob

[[ ${#states[@]} -gt 0 ]] || { echo "[ERROR] no states matched: $STATE_GLOB"; exit 3; }

echo "[INFO] Found ${#states[@]} states."

for s in "${states[@]}"; do
  # extract tag from stateXXXX.hdf5
  # supports state0001.hdf5
  if [[ "$s" =~ state([0-9]+)\.hdf5$ ]]; then
    tag="${BASH_REMATCH[1]}"
    tag=$(printf "%04d" "$((10#$tag))")
  else
    echo "[WARN] skip unrecognized state name: $s"
    continue
  fi

  d="visu_${tag}"
  mkdir -p "$d"

  # copy (not symlink) to keep each visu dir self-contained
  for f in "${REQ_FILES[@]}"; do
    cp -f "$f" "$d/"
  done
  cp -f "$s" "$d/"

  echo "[OK] prepared $d (state=$s)"
done

echo "[DONE] Prepared visu directories."
echo "Next: submit jobs, e.g.: ./submit_visu.sh"



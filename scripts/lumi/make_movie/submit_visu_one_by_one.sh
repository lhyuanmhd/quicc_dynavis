#!/usr/bin/env bash
set -euo pipefail

PATTERN="${PATTERN:-visu_[0-9][0-9][0-9][0-9]}"
SLURM="${SLURM:-Lumi_visu.slurm}"
SLEEP_SEC="${SLEEP_SEC:-30}"
DRYRUN="${DRYRUN:-0}"

is_done() {
  local d="$1"
  [[ -f "$d/visState0000.hdf5" && -f "$d/vis_fields_0000.npz" ]]
}

wait_job_finish() {
  local jid="$1"
  echo "[WAIT] Job $jid running/queued; waiting until it finishes..." >&2

  # 先等 squeue 里消失
  while squeue -j "$jid" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "$jid" -h || true)" ]]; do
    sleep "$SLEEP_SEC"
  done

  # 再用 sacct 取最终状态（有时会有一点延迟）
  local st=""
  for _ in {1..10}; do
    st="$(sacct -j "$jid" --format=State -n | head -n 1 | awk '{print $1}' || true)"
    [[ -n "$st" ]] && break
    sleep 2
  done

  echo "[DONE] Job $jid state=${st:-UNKNOWN}" >&2
}

main() {
  shopt -s nullglob
  local dirs=($PATTERN)
  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "[INFO] No directories matched: $PATTERN"
    exit 0
  fi

  IFS=$'\n' dirs=($(printf "%s\n" "${dirs[@]}" | sort))
  unset IFS

  echo "[INFO] Found ${#dirs[@]} dirs matching $PATTERN"
  echo "[INFO] Using slurm script: $SLURM"
  echo ""

  local n_submit=0 n_skip=0 n_warn=0

  for d in "${dirs[@]}"; do
    [[ -d "$d" ]] || continue

    if is_done "$d"; then
      echo "[SKIP] $d (already done)"
      n_skip=$((n_skip + 1))
      continue
    fi

    if [[ ! -f "$d/$SLURM" ]]; then
      echo "[WARN] $d missing $SLURM, skipping."
      n_warn=$((n_warn + 1))
      continue
    fi

    if [[ "$DRYRUN" == "1" ]]; then
      echo "[DRYRUN] would submit: (cd $d && sbatch --parsable $SLURM)"
      continue
    fi

    # --parsable 确保只返回 jobid（或 jobid;cluster）
    jid="$(cd "$d" && sbatch --parsable "$SLURM" | cut -d';' -f1)"
    echo "[SUBMITTED] $d -> JobID=$jid"
    n_submit=$((n_submit + 1))

    # 严格串行：等这个 job 完全结束再继续
    wait_job_finish "$jid"
  done

  echo ""
  echo "[SUMMARY] submitted=$n_submit, skipped_done=$n_skip, skipped_missing_slurm=$n_warn"
}

main "$@"

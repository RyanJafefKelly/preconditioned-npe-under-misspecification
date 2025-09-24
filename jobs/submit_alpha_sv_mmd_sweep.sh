#!/usr/bin/env bash
# Submit alpha-SV NPE/PNPE RS jobs over a grid of MMD_WEIGHT values.
set -euo pipefail

DEFAULT_WEIGHTS="0.1 1.0 10.0 100.0"
DEFAULT_METHODS="npe_rs pnpe_rs"

read -r -a WEIGHTS <<< "${MMD_WEIGHTS:-$DEFAULT_WEIGHTS}"
read -r -a METHODS <<< "${METHODS:-$DEFAULT_METHODS}"

EXTRA_ARGS=("$@")

declare -A JOB_SCRIPTS=(
  [npe_rs]="jobs/alpha_sv_npe_rs.pbs"
  [pnpe_rs]="jobs/alpha_sv_pnpe_rs.pbs"
)

for method in "${METHODS[@]}"; do
  job_script=${JOB_SCRIPTS[$method]:-}
  if [ -z "${job_script}" ]; then
    echo "Unknown method: ${method}" >&2
    exit 2
  fi
  for weight in "${WEIGHTS[@]}"; do
    vars="MMD_WEIGHT=${weight}"
    if [ -n "${QSUB_VARS:-}" ]; then
      vars+="${QSUB_VARS:+,}${QSUB_VARS}"
    fi
    echo "[submit] method=${method} mmd_weight=${weight}"
    job_id=$(qsub "${EXTRA_ARGS[@]}" -v "$vars" "$job_script")
    echo "  job_id=${job_id}"
  done
done

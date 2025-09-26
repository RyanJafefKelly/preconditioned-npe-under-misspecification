#!/usr/bin/env bash
# Submit one array per method, 32 CPUs each task.
set -euo pipefail

METHODS=(${METHODS:-npe pnpe prnpe rnpe})
R=${R:-1}                      # reps per method
EXTRA_QSUB_ARGS=("$@")         # e.g., -l mem=96GB

for m in "${METHODS[@]}"; do
  vars="METHODS_STR=${m},R=${R},SMC_N_PARTICLES=2000,T=32,START_VOLUME=50.0,PAGE=5,SUMMARY=identity,OBS_MODEL=real"
  echo "[submit] method=${m} R=${R}"
  qsub "${EXTRA_QSUB_ARGS[@]}" -v "$vars" -J 0-$((R-1)) jobs/bvcbm_worker.pbs
done

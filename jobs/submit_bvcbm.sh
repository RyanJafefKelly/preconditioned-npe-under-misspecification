#!/usr/bin/env bash
# Submit one array per method, 32 CPUs each task.
set -euo pipefail

METHODS=(${METHODS:-npe pnpe prnpe rnpe})
R=${R:-5}                      # reps per method
EXTRA_QSUB_ARGS=("$@")         # e.g., -l mem=96GB

for m in "${METHODS[@]}"; do
  vars="METHODS_STR=${m},R=${R},SMC_N_PARTICLES=2000,T=19,START_VOLUME=50.0,PAGE=5,SUMMARY=log,OBS_MODEL=synthetic"
  echo "[submit] method=${m} R=${R}"
  qsub "${EXTRA_QSUB_ARGS[@]}" -v "$vars" -J 0-$((R-1)) jobs/bvcbm_worker.pbs
done

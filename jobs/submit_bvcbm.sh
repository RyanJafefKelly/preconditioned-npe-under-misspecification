#!/usr/bin/env bash
# Submit one job per method (no reps).
set -euo pipefail

METHODS=(${METHODS:-npe pnpe prnpe rnpe rf_abc_npe rf_abc_rnpe})
EXTRA_QSUB_ARGS=("$@")   # e.g., -l ncpus=32 -l mem=96GB

for m in "${METHODS[@]}"; do
  # R=1 → single run; worker derives METHOD from METHODS_STR and sets SEED=0
  vars=(
    "METHODS_STR=${m}"
    "R=1"
    "SMC_N_PARTICLES=2000"
    "SMC_INITIAL_R=4"
    "SMC_ALPHA=0.4"
    "T=32"
    "START_VOLUME=50.0"
    "PAGE=5"
    "SUMMARY=identity"
    "OBS_MODEL=real"
    "BVCBM_START_METHOD=spawn"
  )
  csv=$(IFS=,; echo "${vars[*]}")
  echo "[submit] method=${m}"
  qsub "${EXTRA_QSUB_ARGS[@]}" -v "$csv" jobs/bvcbm_worker.pbs
done

#!/usr/bin/env bash
set -euo pipefail

METHODS=(npe pnpe prnpe rnpe rf_abc_npe rf_abc_rnpe)
T_BY_PATIENT=(19 26 32 32)

for pid in 0 1 2 3; do
  T=${T_BY_PATIENT[$pid]}
  for m in "${METHODS[@]}"; do
    vars=(
      "METHODS_STR=${m}" "R=1"
      "SMC_N_PARTICLES=2000" "SMC_INITIAL_R=4" "SMC_ALPHA=0.4"
      "T=${T}" "START_VOLUME=50.0" "PAGE=5" "SUMMARY=identity"
      "OBS_MODEL=real" "BVCBM_START_METHOD=spawn"
      "DATASET=pancreatic" "PATIENT_IDX=${pid}"
    )
    csv=$(IFS=,; echo "${vars[*]}")
    echo "[submit] ${m} ds=pancreatic p=${pid} T=${T}"
    qsub -l ncpus=32 -l mem=64GB -v "$csv" jobs/bvcbm_worker.pbs
  done
done

#!/usr/bin/env bash
set -euo pipefail

METHODS=(npe pnpe prnpe rnpe rf_abc_npe rf_abc_rnpe)
T_BY_PATIENT=(19 26 32 32)

for pid in 0 1 2 3; do
  T=${T_BY_PATIENT[$pid]}
  for m in "${METHODS[@]}"; do
    csv=$(IFS=,; echo "METHOD=$m,DATASET=pancreatic,PATIENT_IDX=$pid,T=$T,PAGE=5,SUMMARY=identity,OBS_MODEL=real,PPD_N=2000,SEED=0")
    echo "[submit] ppd ${m} p=${pid} T=${T}"
    qsub -l ncpus=1 -l mem=8GB -v "$csv" jobs/bvcbm_ppd_worker.pbs
  done
done

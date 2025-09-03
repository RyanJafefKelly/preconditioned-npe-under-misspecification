#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64=${JAX_ENABLE_X64:-1}

DATE=$(date +"%Y%m%d-%H%M%S")
OUTDIR="results/contaminated_normal_nle_abc/${DATE}"
mkdir -p "$OUTDIR"

: "${SEED:=0}"
: "${THETA:=2.0}"
: "${STDEV_ERR:=2.0}"
: "${N_OBS:=100}"
: "${N_TRAIN:=5000}"
: "${N_PROPS:=400_000}"
: "${Q_ACCEPT:=0.005}"

CMD="uv run python -m precond_npe_misspec.pipelines.contaminated_normal_nle_abc \
  --seed ${SEED} --theta_true ${THETA} --stdev_err ${STDEV_ERR} \
  --n_obs ${N_OBS} --n_train ${N_TRAIN} \
  --n_props ${N_PROPS} --q_accept ${Q_ACCEPT}"

echo "$CMD" | tee "${OUTDIR}/cmd.txt"
$CMD 2>&1 | tee "${OUTDIR}/stdout.log"

#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DATE=$(date +"%Y%m%d-%H%M%S")

: "${METHOD:=npe}"

: "${SEED:=0}"
: "${T:=32}"
: "${START_VOLUME:=100.0}"
: "${PAGE:=5}"
: "${SUMMARY:=identity}"           # log|identity
: "${OBS_MODEL:=real}"   # synthetic|real
: "${EMBEDDER:=asv_tcn}"
: "${DATASET:=pancreatic}"
: "${PATIENT_IDX:=0}"

THETA_DEFAULT="200.0 12.0 50.0"
THETA="${THETA:-$THETA_DEFAULT}"; read -r -a THETA_ARR <<< "$THETA"
(( ${#THETA_ARR[@]} == 3 )) || { echo "need 3 values for THETA"; exit 1; }

: "${N_SIMS:=20000}"
: "${N_POSTERIOR_DRAWS:=20000}"

: "${FLOW_LAYERS:=8}"; : "${NN_WIDTH:=128}"; : "${KNOTS:=10}"; : "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"; : "${MAX_EPOCHS:=500}"; : "${MAX_PATIENCE:=10}"; : "${BATCH_SIZE:=512}"
th_parts=(); for i in "${!THETA_ARR[@]}"; do th_parts+=("p$((i+1))${THETA_ARR[$i]}"); done
THETA_TAG=$(IFS=_; echo "${th_parts[*]}")
RUN_TAG="ds_${DATASET}-p${PATIENT_IDX}-T_${T}"
GROUP="${RUN_TAG}-th_${THETA_TAG}-sum_${SUMMARY}-page_${PAGE}-obs_${OBS_MODEL}-n_sims_${N_SIMS}"
OUTDIR="results/bvcbm/${METHOD}/${GROUP}/seed-${SEED}/${DATE}"
mkdir -p "$OUTDIR"

cmd=(uv run python -m precond_npe_misspec.pipelines.bvcbm
  --seed "$SEED" --obs_seed "$((10#$SEED + 1234))" --outdir "$OUTDIR"
  --T "$T" --start_volume "$START_VOLUME" --page "$PAGE"
  --summary "$SUMMARY" --embedder "$EMBEDDER"
  --theta_true "${THETA_ARR[@]}" --obs_model "$OBS_MODEL"
  --precond.method "none" --precond.n_sims "$N_SIMS"
  --posterior.method "npe" --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"
  --flow.flow_layers "$FLOW_LAYERS" --flow.nn_width "$NN_WIDTH" --flow.knots "$KNOTS" --flow.interval "$INTERVAL"
  --flow.learning_rate "$LEARNING_RATE" --flow.max_epochs "$MAX_EPOCHS" --flow.max_patience "$MAX_PATIENCE" --flow.batch_size "$BATCH_SIZE"
)
cmd+=( --dataset "$DATASET" --patient_idx "$PATIENT_IDX" )

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"; echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

SUMMARIES_PATH=$([ "$SUMMARY" = "log" ] && echo "summary_log" || echo "summary_identity")

# Parameter metrics only when synthetic ground truth is known
if [[ "$OBS_MODEL" = "synthetic" ]]; then
  THETA_TARGET_DEFAULT="$THETA_DEFAULT"
  THETA_TARGET="${THETA_TARGET:-$THETA_TARGET_DEFAULT}"; THETA_TARGET="${THETA_TARGET//,/}"; read -r -a THETA_TARGET_ARR <<< "$THETA_TARGET"

  cat > "$OUTDIR/entrypoints.json" <<EOF
{
  "simulate": "precond_npe_misspec.examples.bvcbm:simulate_biphasic",
  "summaries": "precond_npe_misspec.examples.bvcbm:${SUMMARIES_PATH}",
  "sim_kwargs": {"T": $T, "start_volume": $START_VOLUME, "page": $PAGE},
  "summaries_kwargs": {}
}
EOF

  uv run python -m precond_npe_misspec.scripts.metrics_from_samples \
    --outdir "$OUTDIR" \
    --theta-target "${THETA_TARGET_ARR[@]}" \
    --level 0.95 --want-hpdi --want-central \
    --method NPE \
    --compute-ppd --ppd-entrypoints "$OUTDIR/entrypoints.json" \
    --ppd-n 1000 --ppd-metric l2
fi

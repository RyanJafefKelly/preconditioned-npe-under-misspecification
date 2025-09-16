#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DATE=$(date +"%Y%m%d-%H%M%S")

: "${SEED:=0}"
: "${K:=6}"
: "${T:=1000}"
: "${OBS_MODEL:=assumed}"   # assumed|true

# θ is 7D for k=6
THETA_DEFAULT="0.579 -0.143 0.836 0.745 -0.660 -0.254 0.1"
THETA="${THETA:-$THETA_DEFAULT}"; read -r -a THETA_ARR <<< "$THETA"
(( ${#THETA_ARR[@]} == 7 )) || { echo "need 7 values for THETA"; exit 1; }

: "${N_SIMS:=20000}"
: "${N_POSTERIOR_DRAWS:=20000}"

: "${FLOW_LAYERS:=8}"; : "${NN_WIDTH:=128}"; : "${KNOTS:=10}"; : "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"; : "${MAX_EPOCHS:=500}"; : "${MAX_PATIENCE:=10}"; : "${BATCH_SIZE:=512}"

th_parts=(); for i in "${!THETA_ARR[@]}"; do th_parts+=("p$((i+1))${THETA_ARR[$i]}"); done
THETA_TAG=$(IFS=_; echo "${th_parts[*]}")
GROUP="th_${THETA_TAG}-K_${K}-T_${T}-obs_${OBS_MODEL}-n_sims_${N_SIMS}"
OUTDIR="results/svar/npe/${GROUP}/seed-${SEED}/${DATE}"; mkdir -p "$OUTDIR"

cmd=(uv run python -m precond_npe_misspec.pipelines.svar
  --seed "$SEED" --obs_seed "$((10#$SEED + 1234))" --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}" --k "$K" --T "$T" --obs_model "$OBS_MODEL"
  --precond.method "none" --precond.n_sims "$N_SIMS"
  --posterior.method "npe" --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"
  --flow.flow_layers "$FLOW_LAYERS" --flow.nn_width "$NN_WIDTH" --flow.knots "$KNOTS" --flow.interval "$INTERVAL"
  --flow.learning_rate "$LEARNING_RATE" --flow.max_epochs "$MAX_EPOCHS" --flow.max_patience "$MAX_PATIENCE" --flow.batch_size "$BATCH_SIZE"
)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"; echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

THETA_TARGET_DEFAULT="0.835 0.382 0.899 0.824 0.172 0.283 0.1286"
THETA_TARGET="${THETA_TARGET:-$THETA_TARGET_DEFAULT}"; THETA_TARGET="${THETA_TARGET//,/}"; read -r -a THETA_TARGET_ARR <<< "$THETA_TARGET"

cat > "$OUTDIR/entrypoints.json" <<EOF
{
  "simulate": "precond_npe_misspec.examples.svar:simulate",
  "summaries": "precond_npe_misspec.examples.svar:summaries_for_metrics",
  "sim_kwargs": {"k": $K, "T": $T, "obs_model": "$OBS_MODEL"},
  "summaries_kwargs": {"k": $K}
}
EOF


uv run python -m precond_npe_misspec.scripts.metrics_from_samples \
  --outdir "$OUTDIR" \
  --theta-target "${THETA_TARGET_ARR[@]}" \
  --level 0.95 \
  --want-hpdi \
  --want-central \
  --method NPE \
  --compute-ppd \
  --ppd-entrypoints "$OUTDIR/entrypoints.json" \
  --ppd-n 1000 \
  --ppd-metric l2

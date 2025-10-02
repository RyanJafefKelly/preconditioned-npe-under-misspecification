#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DATE=$(date +"%Y%m%d-%H%M%S")

# Ensure local packages (e.g. data/) stay on the import path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${SEED:=0}"
: "${T:=1000}"
: "${THETA1:=0.0}"
: "${OBS_MODEL:=true}"      # assumed|true

# θ = (θ2, θ3, α)
THETA_DEFAULT="0.95 0.25 1.99"
THETA="${THETA:-$THETA_DEFAULT}"; read -r -a THETA_ARR <<< "$THETA"
(( ${#THETA_ARR[@]} == 3 )) || { echo "need 3 values for THETA"; exit 1; }

# Simulation budget for training pairs (prior predictive)
: "${N_SIMS:=4000}"

# Posterior draws
: "${N_POSTERIOR_DRAWS:=20000}"

# Flow hyperparameters
: "${FLOW_LAYERS:=8}"; : "${NN_WIDTH:=128}"; : "${KNOTS:=10}"; : "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"; : "${MAX_EPOCHS:=100}"; : "${MAX_PATIENCE:=10}"; : "${BATCH_SIZE:=512}"

# NPE‑RS (embedding + MMD)
: "${EMBED_DIM:=4}"
: "${EMBED_WIDTH:=128}"
: "${EMBED_DEPTH:=1}"
: "${MMD_WEIGHT:=1.0}"
MMD_TAG_RAW=$(printf '%s' "$MMD_WEIGHT" | tr '[:upper:]' '[:lower:]')
MMD_TAG=$(printf '%s' "$MMD_TAG_RAW" | tr -c '0-9a-z' '_')
: "${MMD_SUBSAMPLE:=256}"
: "${BANDWIDTH:=median}"
: "${KERNEL:=rbf}"
: "${WARMUP_EPOCHS:=0}"

th_parts=(); for i in "${!THETA_ARR[@]}"; do th_parts+=("p$((i+1))${THETA_ARR[$i]}"); done
THETA_TAG=$(IFS=_; echo "${th_parts[*]}")
GROUP="th_${THETA_TAG}-T_${T}-th1_${THETA1}-obs_${OBS_MODEL}-n_sims_${N_SIMS}-mmd_${MMD_TAG}"
OUTDIR="results/alpha_sv/npe_rs/${GROUP}/seed-${SEED}/${DATE}"; mkdir -p "$OUTDIR"

cmd=(uv run python -m precond_npe_misspec.pipelines.alpha_sv
  --seed "$SEED" --obs_seed "$((10#$SEED + 1234))" --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}" --T "$T" --theta1 "$THETA1"
  --obs_model "$OBS_MODEL"
  --precond.method "none" --precond.n_sims "$N_SIMS"
  --posterior.method "npe_rs" --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"
  --npers.embed_dim "$EMBED_DIM" --npers.embed_width "$EMBED_WIDTH" --npers.embed_depth "$EMBED_DEPTH"
  --npers.mmd_weight "$MMD_WEIGHT" --npers.mmd_subsample "$MMD_SUBSAMPLE"
  --npers.bandwidth "$BANDWIDTH" --npers.kernel "$KERNEL" --npers.warmup_epochs "$WARMUP_EPOCHS"
  --flow.flow_layers "$FLOW_LAYERS" --flow.nn_width "$NN_WIDTH" --flow.knots "$KNOTS" --flow.interval "$INTERVAL"
  --flow.learning_rate "$LEARNING_RATE" --flow.max_epochs "$MAX_EPOCHS" --flow.max_patience "$MAX_PATIENCE" --flow.batch_size "$BATCH_SIZE"
)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"; echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

THETA_TARGET_DEFAULT="0.95 0.25 1.99"
THETA_TARGET="${THETA_TARGET:-$THETA_TARGET_DEFAULT}"; THETA_TARGET="${THETA_TARGET//,/}"; read -r -a THETA_TARGET_ARR <<< "$THETA_TARGET"

cat > "$OUTDIR/entrypoints.json" <<EOF
{
  "simulate": "precond_npe_misspec.examples.alpha_stable_sv:simulate",
  "summaries": "precond_npe_misspec.examples.alpha_stable_sv:summaries_for_metrics",
  "sim_kwargs": {"T": $T, "theta1": $THETA1},
  "summaries_kwargs": {}
}
EOF

uv run python -m precond_npe_misspec.scripts.metrics_from_samples \
  --outdir "$OUTDIR" \
  --theta-target "${THETA_TARGET_ARR[@]}" \
  --level 0.95 --want-hpdi --want-central \
  --method NPE_RS \
  --compute-ppd --ppd-entrypoints "$OUTDIR/entrypoints.json" \
  --ppd-n 1000 --ppd-metric l2

#!/usr/bin/env bash
set -euo pipefail
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")
: "${SEED:=0}"

# Data
: "${N_OBS:=200}"
THETA_DEFAULT="0.8 2.0"         # (k, lambda)
: "${OBS_MODEL:=true}"          # assumed|true
: "${EPS:=0.05}"                # only used when OBS_MODEL=true
: "${ALPHA:=40.0}"              # unused in current true_dgp; kept for parity

THETA="${THETA:-$THETA_DEFAULT}"; read -r -a THETA_ARR <<< "$THETA"
(( ${#THETA_ARR[@]} == 2 )) || { echo "need 2 values for THETA"; exit 1; }

# Posterior draws
: "${N_POSTERIOR_DRAWS:=20000}"

# Flow (used inside flowjax; NPE‑RS still uses these for the conditional flow body)
: "${FLOW_LAYERS:=8}"; : "${NN_WIDTH:=128}"; : "${KNOTS:=10}"; : "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"; : "${MAX_EPOCHS:=50}"; : "${MAX_PATIENCE:=10}"; : "${BATCH_SIZE:=512}"

# NPE‑RS hyperparameters
: "${EMBED_DIM:=4}"
: "${EMBED_WIDTH:=128}"
: "${EMBED_DEPTH:=1}"
: "${MMD_WEIGHT:=10.0}"
: "${MMD_SUBSAMPLE:=256}"
: "${BANDWIDTH:=median}"        # or a number, e.g. 0.5
: "${KERNEL:=rbf}"
: "${WARMUP_EPOCHS:=0}"

GROUP="th_$(printf 'k%s_lam%s' "${THETA_ARR[@]}")-n_obs_${N_OBS}-obs_${OBS_MODEL}"
OUTDIR="results/contaminated_weibull/npe_rs/${GROUP}/seed-${SEED}/${DATE}"
mkdir -p "$OUTDIR"

cmd=(uv run python -m precond_npe_misspec.pipelines.contaminated_weibull
  --seed "$SEED" --obs_seed "$((10#$SEED + 1234))" --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}" --n_obs "$N_OBS" --obs_model "$OBS_MODEL" --eps "$EPS" --alpha "$ALPHA"
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

THETA_TARGET_DEFAULT="$THETA_DEFAULT"
THETA_TARGET="${THETA_TARGET:-$THETA_TARGET_DEFAULT}"; THETA_TARGET="${THETA_TARGET//,/}"; read -r -a THETA_TARGET_ARR <<< "$THETA_TARGET"

uv run python -m precond_npe_misspec.scripts.metrics_from_samples \
  --outdir "$OUTDIR" \
  --theta-target "${THETA_TARGET_ARR[@]}" \
  --level 0.95 --want-hpdi --want-central \
  --method NPE_RS \
  --compute-ppd --ppd-entrypoints "$OUTDIR/entrypoints.json" \
  --ppd-n 1000 --ppd-metric l2

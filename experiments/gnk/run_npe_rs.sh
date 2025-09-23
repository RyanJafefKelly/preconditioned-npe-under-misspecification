#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DATE=$(date +"%Y%m%d-%H%M%S")

: "${SEED:=0}"

# GNK specifics
: "${N_OBS:=5000}"
THETA_DEFAULT="2.3663 4.1757 1.7850 0.1001"
THETA="${THETA:-$THETA_DEFAULT}"
read -r -a THETA_ARR <<< "$THETA"
if (( ${#THETA_ARR[@]} != 4 )); then
  echo "Error: need 4 values for THETA (A B g k). Got: ${#THETA_ARR[@]}" >&2
  exit 1
fi

# Training set size (no preconditioning)
: "${N_SIMS:=20000}"

OUTDIR="results/gnk/npe_rs/th_$(printf 'A%s_B%s_g%s_k%s' "${THETA_ARR[@]}")-n_obs_${N_OBS}-n_sims_${N_SIMS}/seed-${SEED}/${DATE}"
mkdir -p "$OUTDIR"

# Posterior draws
: "${N_POSTERIOR_DRAWS:=20000}"

# Flow hyperparameters (posterior flow)
: "${FLOW_LAYERS:=8}"
: "${NN_WIDTH:=128}"
: "${KNOTS:=10}"
: "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"
: "${MAX_EPOCHS:=500}"
: "${MAX_PATIENCE:=10}"
: "${BATCH_SIZE:=512}"

# NPE‑RS (embedding + MMD)
: "${EMBED_WIDTH:=128}"
: "${EMBED_DEPTH:=2}"
: "${EMBED_DIM:=}"          # leave empty to default to s_dim in code
: "${MMD_WEIGHT:=1.0}"
: "${MMD_SUBSAMPLE:=256}"
: "${BANDWIDTH:=median}"    # median|fixed
: "${KERNEL:=rbf}"          # rbf only for now
: "${WARMUP_EPOCHS:=0}"

cmd=(uv run python -m precond_npe_misspec.pipelines.gnk
  --seed "$SEED"
  --obs_seed "$((10#$SEED + 1234))"
  --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}"
  --n_obs "$N_OBS"

  --precond.method "none"
  --precond.n_sims "$N_SIMS"

  --posterior.method "npe_rs"
  --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"

  --npers.embed_width "$EMBED_WIDTH"
  --npers.embed_depth "$EMBED_DEPTH"
  --npers.mmd_weight "$MMD_WEIGHT"
  --npers.mmd_subsample "$MMD_SUBSAMPLE"
  --npers.bandwidth "$BANDWIDTH"
  --npers.kernel "$KERNEL"
  --npers.warmup_epochs "$WARMUP_EPOCHS"

  --flow.flow_layers "$FLOW_LAYERS"
  --flow.nn_width "$NN_WIDTH"
  --flow.knots "$KNOTS"
  --flow.interval "$INTERVAL"
  --flow.learning_rate "$LEARNING_RATE"
  --flow.max_epochs "$MAX_EPOCHS"
  --flow.max_patience "$MAX_PATIENCE"
  --flow.batch_size "$BATCH_SIZE"
)

# Optional embed dim override if you don't want s_dim
if [[ -n "${EMBED_DIM:-}" ]]; then
  cmd+=(--npers.embed_dim "$EMBED_DIM")
fi

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"
echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

THETA_TARGET_DEFAULT="2.3663 4.1757 1.7850 0.1001"
THETA_TARGET="${THETA_TARGET:-$THETA_TARGET_DEFAULT}"
THETA_TARGET="${THETA_TARGET//,/}"
read -r -a THETA_TARGET_ARR <<< "$THETA_TARGET"

uv run python -m precond_npe_misspec.scripts.metrics_from_samples \
  --outdir "$OUTDIR" \
  --theta-target "${THETA_TARGET_ARR[@]}" \
  --level 0.95 \
  --want-hpdi \
  --want-central \
  --method NPE_RS \
  --compute-ppd \
  --ppd-entrypoints "$OUTDIR/entrypoints.json" \
  --ppd-n 1000 \
  --ppd-metric l2

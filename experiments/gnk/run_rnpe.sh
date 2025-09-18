#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DATE=$(date +"%Y%m%d-%H%M%S")

: "${SEED:=0}"

# GNK specifics
: "${N_OBS:=5000}"
# THETA_DEFAULT="3.0 1.0 2.0 0.5"
THETA_DEFAULT="2.3663 4.1757 1.7850 0.1001"

THETA="${THETA:-$THETA_DEFAULT}"
read -r -a THETA_ARR <<< "$THETA"
if (( ${#THETA_ARR[@]} != 4 )); then
  echo "Error: need 4 values for THETA (A B g k). Got: ${#THETA_ARR[@]}" >&2
  exit 1
fi

# Training set size (no preconditioning)
: "${N_SIMS:=20000}"

OUTDIR="results/gnk/rnpe/th_$(printf 'A%s_B%s_g%s_k%s' "${THETA_ARR[@]}")-n_obs_${N_OBS}-n_sims_${N_SIMS}/seed-${SEED}/${DATE}"
mkdir -p "$OUTDIR"

# Posterior draws
: "${N_POSTERIOR_DRAWS:=20000}"

# Flow hyperparameters
: "${FLOW_LAYERS:=8}"
: "${NN_WIDTH:=128}"
: "${KNOTS:=10}"
: "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"
: "${MAX_EPOCHS:=500}"
: "${MAX_PATIENCE:=10}"
: "${BATCH_SIZE:=512}"

# Denoising model
: "${DENOISE_MODEL:=spike_slab}"  # laplace|laplace_adaptive|student_t|cauchy|spike_slab
: "${LAPLACE_ALPHA:=0.3}"
: "${LAPLACE_MIN_SCALE:=0.01}"
: "${STUDENT_T_SCALE:=0.05}"
: "${STUDENT_T_DF:=1.0}"
: "${CAUCHY_SCALE:=0.05}"
: "${SPIKE_STD:=0.01}"
: "${SLAB_SCALE:=0.25}"
: "${MISSPECIFIED_PROB:=0.5}"
: "${LEARN_PROB:=0}"                   # 1 to infer rho
: "${MCMC_WARMUP:=1000}"
: "${MCMC_SAMPLES:=2000}"
: "${MCMC_THIN:=1}"

cmd=(uv run python -m precond_npe_misspec.pipelines.gnk
  --seed "$SEED"
  --obs_seed "$((10#$SEED + 1234))"
  --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}"
  --n_obs "$N_OBS"

  --precond.method "none"
  --precond.n_sims "$N_SIMS"

  --posterior.method "rnpe"
  --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"

  --robust.denoise_model "$DENOISE_MODEL"
  --robust.laplace_alpha "$LAPLACE_ALPHA"
  --robust.laplace_min_scale "$LAPLACE_MIN_SCALE"
  --robust.student_t_scale "$STUDENT_T_SCALE"
  --robust.student_t_df "$STUDENT_T_DF"
  --robust.cauchy_scale "$CAUCHY_SCALE"
  --robust.spike_std "$SPIKE_STD"
  --robust.slab_scale "$SLAB_SCALE"
  --robust.misspecified_prob "$MISSPECIFIED_PROB"

  --robust.mcmc_warmup "$MCMC_WARMUP"
  --robust.mcmc_samples "$MCMC_SAMPLES"
  --robust.mcmc_thin "$MCMC_THIN"

  --flow.flow_layers "$FLOW_LAYERS"
  --flow.nn_width "$NN_WIDTH"
  --flow.knots "$KNOTS"
  --flow.interval "$INTERVAL"
  --flow.learning_rate "$LEARNING_RATE"
  --flow.max_epochs "$MAX_EPOCHS"
  --flow.max_patience "$MAX_PATIENCE"
  --flow.batch_size "$BATCH_SIZE"
)

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
  --method RNPE \
  --compute-ppd \
  --ppd-entrypoints "$OUTDIR/entrypoints.json" \
  --ppd-n 1000 \
  --ppd-metric l2


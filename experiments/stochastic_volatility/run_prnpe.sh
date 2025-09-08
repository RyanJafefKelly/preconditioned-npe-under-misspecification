#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")
OUTDIR="results/stochastic_volatility/prnpe/${DATE}"
mkdir -p "$OUTDIR"

: "${SEED:=0}"
: "${T:=100}"

# θ is 2D: (sigma_rw, nu)
THETA_DEFAULT="0.01 10.0"
THETA="${THETA:-$THETA_DEFAULT}"
read -r -a THETA_ARR <<< "$THETA"
if (( ${#THETA_ARR[@]} != 2 )); then
  echo "Error: need 2 values for THETA. Got: ${#THETA_ARR[@]}" >&2
  exit 1
fi

# Misspecification for observed data
: "${SIGMA_MS:=2}"
: "${BLOCK_START:=50}"   # 1‑indexed inclusive
: "${BLOCK_END:=65}"     # 1‑indexed inclusive

# ABC preconditioning
: "${N_SIMS:=20000}"
: "${Q_PRECOND:=0.2}"

# Posterior sampling
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

# Distance
: "${DISTANCE:=euclidean}"   # euclidean|l1|mmd
: "${MMD_UNBIASED:=0}"
: "${MMD_BANDWIDTH:=}"       # empty -> median heuristic

# RNPE denoiser
: "${DENOISE_MODEL:=cauchy}"   # student_t|cauchy|spike_slab (TODO: test laplace)
: "${MCMC_NUM_WARMUP:=1000}"
: "${MCMC_NUM_SAMPLES:=2000}"
: "${MCMC_THINNING:=1}"

cmd=(uv run python -m precond_npe_misspec.pipelines.stochastic_volatility_prnpe
  --seed "$SEED"
  --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}"
  --T "$T"
  --sigma_ms "$SIGMA_MS"
  --block_start "$BLOCK_START"
  --block_end "$BLOCK_END"
  --n_sims "$N_SIMS"
  --q_precond "$Q_PRECOND"
  --n_posterior_draws "$N_POSTERIOR_DRAWS"
  --flow_layers "$FLOW_LAYERS"
  --nn_width "$NN_WIDTH"
  --knots "$KNOTS"
  --interval "$INTERVAL"
  --learning_rate "$LEARNING_RATE"
  --max_epochs "$MAX_EPOCHS"
  --max_patience "$MAX_PATIENCE"
  --batch_size "$BATCH_SIZE"
  --distance "$DISTANCE"
  --denoise_model "$DENOISE_MODEL"
  --mcmc_num_warmup "$MCMC_NUM_WARMUP"
  --mcmc_num_samples "$MCMC_NUM_SAMPLES"
  --mcmc_thinning "$MCMC_THINNING"
)

[[ -n "$MMD_BANDWIDTH" ]] && cmd+=(--mmd_bandwidth "$MMD_BANDWIDTH")
[[ "$MMD_UNBIASED" == "1" ]] && cmd+=(--mmd_unbiased)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"
echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

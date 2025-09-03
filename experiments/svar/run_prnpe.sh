# experiments/svar/run_prnpe.sh
#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")
OUTDIR="results/svar_prnpe/${DATE}"
mkdir -p "$OUTDIR"

: "${SEED:=0}"
: "${K:=6}"
: "${T:=1000}"

# θ is 7D for k=6. Space-separated ASCII numbers only. No quotes.
THETA_DEFAULT="0.579 -0.143 0.836 0.745 -0.660 -0.254 0.1"
THETA="${THETA:-$THETA_DEFAULT}"
read -r -a THETA_ARR <<< "$THETA"
if (( ${#THETA_ARR[@]} != 7 )); then
  echo "Error: need 7 values for THETA. Got: ${#THETA_ARR[@]}" >&2
  exit 1
fi

# Preconditioning (ABC)
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

# RNPE denoiser config
: "${DENOISE_MODEL:=spike_slab}"
: "${LAPLACE_ALPHA:=0.3}"
: "${LAPLACE_MIN_SCALE:=0.01}"
: "${STUDENT_T_SCALE:=0.05}"
: "${STUDENT_T_DF:=1.0}"
: "${CAUCHY_SCALE:=0.05}"
: "${SPIKE_STD:=0.01}"
: "${SLAB_SCALE:=0.25}"
: "${MISSPECIFIED_PROB:=0.5}"
: "${LEARN_PROB:=0}"             # 1 to enable, else 0

# MCMC config
: "${MCMC_WARMUP:=1000}"
: "${MCMC_SAMPLES:=2000}"
: "${MCMC_THIN:=1}"

cmd=(uv run python -m precond_npe_misspec.pipelines.svar_prnpe
  --seed "$SEED"
  --theta_true "${THETA_ARR[@]}"
  --k "$K" --T "$T"
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
  --denoise_model "$DENOISE_MODEL"
  --student_t_scale "$STUDENT_T_SCALE"
  --student_t_df "$STUDENT_T_DF"
  --cauchy_scale "$CAUCHY_SCALE"
  --spike_std "$SPIKE_STD"
  --slab_scale "$SLAB_SCALE"
  --misspecified_prob "$MISSPECIFIED_PROB"
  --mcmc_warmup "$MCMC_WARMUP"
  --mcmc_samples "$MCMC_SAMPLES"
  --mcmc_thin "$MCMC_THIN"
)

# Toggle boolean flag
[[ "$LEARN_PROB" == "1" ]] && cmd+=(--learn_prob)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"
echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

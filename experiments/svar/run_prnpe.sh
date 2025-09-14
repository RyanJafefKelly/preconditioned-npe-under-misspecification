# experiments/svar/run_prnpe.sh
#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")

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
# SMC‑ABC controls
: "${PRECOND_METHOD:=smc_abc}"        # smc_abc | rejection
: "${SMC_N_PARTICLES:=4000}"
: "${SMC_ALPHA:=0.5}"
: "${SMC_EPSILON0:=1e6}"
: "${SMC_EPS_MIN:=1e-3}"
: "${SMC_ACC_MIN:=0.10}"
: "${SMC_MAX_ITERS:=5}"
: "${SMC_INITIAL_R:=1}"
: "${SMC_C_TUNING:=0.01}"
: "${SMC_B_SIM:=1}"


th_parts=()
for i in "${!THETA_ARR[@]}"; do
  th_parts+=("p$((i+1))${THETA_ARR[$i]}")
done
THETA_TAG=$(IFS=_; echo "${th_parts[*]}")

GROUP="th_${THETA_TAG}-K_${K}-T_${T}-n_sims_${N_SIMS}-q_${Q_PRECOND}"

OUTDIR="results/svar/prnpe/${GROUP}/${DATE}"
mkdir -p "$OUTDIR"

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
  --obs_seed "$((10#$SEED + 1234))"
  --outdir "$OUTDIR"
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
  --precond_method "$PRECOND_METHOD"
  --smc_n_particles "$SMC_N_PARTICLES"
  --smc_alpha "$SMC_ALPHA"
  --smc_epsilon0 "$SMC_EPSILON0"
  --smc_eps_min "$SMC_EPS_MIN"
  --smc_acc_min "$SMC_ACC_MIN"
  --smc_max_iters "$SMC_MAX_ITERS"
  --smc_initial_R "$SMC_INITIAL_R"
  --smc_c_tuning "$SMC_C_TUNING"
  --smc_B_sim "$SMC_B_SIM"
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

THETA_TARGET_DEFAULT="0.835 0.382 0.899 0.824 0.172 0.283 0.1286"
THETA_TARGET="${THETA_TARGET:-$THETA_TARGET_DEFAULT}"
THETA_TARGET="${THETA_TARGET//,/}"
read -r -a THETA_TARGET_ARR <<< "$THETA_TARGET"

uv run python -m precond_npe_misspec.scripts.metrics_from_samples \
  --outdir "$OUTDIR" \
  --theta-target "${THETA_TARGET_ARR[@]}" \
  --level 0.95 \
  --want-hpdi \
  --want-central \
  --method PRNPE

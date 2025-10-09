#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DATE=$(date +"%Y%m%d-%H%M%S")

: "${METHOD:=prnpe}"

: "${SEED:=0}"
: "${T:=32}"
: "${START_VOLUME:=50.0}"
: "${PAGE:=5}"
: "${SUMMARY:=identity}"           # log|identity
: "${OBS_MODEL:=real}"   # synthetic|real
: "${EMBEDDER:=asv_tcn}"
: "${DATASET:=pancreatic}"
: "${PATIENT_IDX:=0}"

THETA_DEFAULT="200.0 12.0 50.0"
THETA="${THETA:-$THETA_DEFAULT}"; read -r -a THETA_ARR <<< "$THETA"
(( ${#THETA_ARR[@]} == 3 )) || { echo "need 3 values for THETA"; exit 1; }

# ABC preconditioning (SMC-ABC)
: "${N_SIMS:=20000}"; : "${Q_PRECOND:=0.2}"; : "${PRECOND_METHOD:=smc_abc}"
: "${SMC_N_PARTICLES:=4000}"; : "${SMC_ALPHA:=0.5}"; : "${SMC_EPSILON0:=1e6}"
: "${SMC_EPS_MIN:=1e-3}"; : "${SMC_ACC_MIN:=0.10}"; : "${SMC_MAX_ITERS:=4}"
: "${SMC_INITIAL_R:=1}"; : "${SMC_C_TUNING:=0.01}"; : "${SMC_B_SIM:=1}"

: "${N_POSTERIOR_DRAWS:=20000}"
: "${FLOW_LAYERS:=8}"; : "${NN_WIDTH:=128}"; : "${KNOTS:=10}"; : "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"; : "${MAX_EPOCHS:=500}"; : "${MAX_PATIENCE:=10}"; : "${BATCH_SIZE:=512}"

# RNPE denoiser hyperparameters
: "${DENOISE_MODEL:=spike_slab}"
: "${LAPLACE_ALPHA:=0.3}"; : "${LAPLACE_MIN_SCALE:=0.01}"
: "${STUDENT_T_SCALE:=0.05}"; : "${STUDENT_T_DF:=1.0}"
: "${CAUCHY_SCALE:=0.05}"; : "${SPIKE_STD:=0.01}"; : "${SLAB_SCALE:=0.25}"
: "${MISSPECIFIED_PROB:=0.5}"; : "${LEARN_PROB:=0}"

# MCMC inside RNPE
: "${MCMC_WARMUP:=1000}"; : "${MCMC_SAMPLES:=2000}"; : "${MCMC_THIN:=1}"

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
  --precond.method "$PRECOND_METHOD" --precond.n_sims "$N_SIMS" --precond.q_precond "$Q_PRECOND"
  --precond.smc_n_particles "$SMC_N_PARTICLES" --precond.smc_alpha "$SMC_ALPHA" --precond.smc_epsilon0 "$SMC_EPSILON0"
  --precond.smc_eps_min "$SMC_EPS_MIN" --precond.smc_acc_min "$SMC_ACC_MIN" --precond.smc_max_iters "$SMC_MAX_ITERS"
  --precond.smc_initial_R "$SMC_INITIAL_R" --precond.smc_c_tuning "$SMC_C_TUNING" --precond.smc_B_sim "$SMC_B_SIM"
  --posterior.method "rnpe" --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"
  --robust.denoise_model "$DENOISE_MODEL"
  --robust.laplace_alpha "$LAPLACE_ALPHA" --robust.laplace_min_scale "$LAPLACE_MIN_SCALE"
  --robust.student_t_scale "$STUDENT_T_SCALE" --robust.student_t_df "$STUDENT_T_DF"
  --robust.cauchy_scale "$CAUCHY_SCALE" --robust.spike_std "$SPIKE_STD" --robust.slab_scale "$SLAB_SCALE"
  --robust.misspecified_prob "$MISSPECIFIED_PROB"
  --robust.mcmc_warmup "$MCMC_WARMUP" --robust.mcmc_samples "$MCMC_SAMPLES" --robust.mcmc_thin "$MCMC_THIN"
  --flow.flow_layers "$FLOW_LAYERS" --flow.nn_width "$NN_WIDTH" --flow.knots "$KNOTS" --flow.interval "$INTERVAL"
  --flow.learning_rate "$LEARNING_RATE" --flow.max_epochs "$MAX_EPOCHS" --flow.max_patience "$MAX_PATIENCE" --flow.batch_size "$BATCH_SIZE"
)
cmd+=( --dataset "$DATASET" --patient_idx "$PATIENT_IDX" )
if [[ "${LEARN_PROB}" == "1" ]]; then cmd+=(--robust.learn_prob); fi

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"; echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

SUMMARIES_PATH=$([ "$SUMMARY" = "log" ] && echo "summary_log" || echo "summary_identity")

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
    --method PRNPE \
    --compute-ppd --ppd-entrypoints "$OUTDIR/entrypoints.json" \
    --ppd-n 1000 --ppd-metric l2
fi

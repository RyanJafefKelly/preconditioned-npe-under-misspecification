#!/usr/bin/env bash
# configs/gnk/method_matrix.sh
# Central defaults + method/scenario mapping for GNK experiments.
set -euo pipefail

# ---- Global defaults (override at submit time if needed) ----
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export RESULTS_ROOT="${RESULTS_ROOT:-results/gnk}"

# Data/model
export N_OBS="${N_OBS:-5000}"
export THETA_TRUE="${THETA_TRUE:-2.3663 4.1757 1.7850 0.1001}"

# Target used by metrics (override if you have a pseudo‑true)
export THETA_TARGET_OVERRIDE="${THETA_TARGET_OVERRIDE:-}"

# Simulation budget + flow
export N_SIMS="${N_SIMS:-20000}"
export N_POSTERIOR_DRAWS="${N_POSTERIOR_DRAWS:-20000}"
export FLOW_LAYERS="${FLOW_LAYERS:-8}"
export NN_WIDTH="${NN_WIDTH:-128}"
export KNOTS="${KNOTS:-10}"
export INTERVAL="${INTERVAL:-8.0}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
export MAX_EPOCHS="${MAX_EPOCHS:-500}"
export MAX_PATIENCE="${MAX_PATIENCE:-10}"
export BATCH_SIZE="${BATCH_SIZE:-512}"

# Preconditioning
export PRECOND_METHOD_DEFAULT="${PRECOND_METHOD_DEFAULT:-smc_abc}"   # smc_abc | rejection
export Q_PRECOND="${Q_PRECOND:-0.2}"
export SMC_N_PARTICLES="${SMC_N_PARTICLES:-4000}"
export SMC_ALPHA="${SMC_ALPHA:-0.5}"
export SMC_EPSILON0="${SMC_EPSILON0:-1e6}"
export SMC_EPS_MIN="${SMC_EPS_MIN:-1e-3}"
export SMC_ACC_MIN="${SMC_ACC_MIN:-0.10}"
export SMC_MAX_ITERS="${SMC_MAX_ITERS:-4}"
export SMC_INITIAL_R="${SMC_INITIAL_R:-1}"
export SMC_C_TUNING="${SMC_C_TUNING:-0.01}"
export SMC_B_SIM="${SMC_B_SIM:-1}"

# Robust RNPE (denoiser)
export DENOISE_MODEL="${DENOISE_MODEL:-spike_slab}"  # laplace|laplace_adaptive|student_t|cauchy|spike_slab
export LAPLACE_ALPHA="${LAPLACE_ALPHA:-0.3}"
export LAPLACE_MIN_SCALE="${LAPLACE_MIN_SCALE:-0.01}"
export STUDENT_T_SCALE="${STUDENT_T_SCALE:-0.05}"
export STUDENT_T_DF="${STUDENT_T_DF:-1.0}"
export CAUCHY_SCALE="${CAUCHY_SCALE:-0.05}"
export SPIKE_STD="${SPIKE_STD:-0.01}"
export SLAB_SCALE="${SLAB_SCALE:-0.25}"
export MISSPECIFIED_PROB="${MISSPECIFIED_PROB:-0.5}"
export LEARN_PROB="${LEARN_PROB:-1}"
export MCMC_WARMUP="${MCMC_WARMUP:-1000}"
export MCMC_SAMPLES="${MCMC_SAMPLES:-2000}"
export MCMC_THIN="${MCMC_THIN:-1}"

gnk_set_method_env() {
  local method="$1"

  # Base across all methods
  export THETA="${THETA_TRUE}"

  case "$method" in
    npe)
      export PRECOND_METHOD="none"
      export THETA_TARGET="${THETA_TARGET_OVERRIDE:-$THETA_TRUE}"
      ;;
    pnpe)
      export PRECOND_METHOD="${PRECOND_METHOD_DEFAULT}"
      export THETA_TARGET="${THETA_TARGET_OVERRIDE:-$THETA_TRUE}"
      ;;
    rnpe)
      export PRECOND_METHOD="none"
      export THETA_TARGET="${THETA_TARGET_OVERRIDE:-$THETA_TRUE}"
      ;;
    prnpe)
      export PRECOND_METHOD="${PRECOND_METHOD_DEFAULT}"
      export THETA_TARGET="${THETA_TARGET_OVERRIDE:-$THETA_TRUE}"
      ;;
    *)
      echo "Unknown method: $method" >&2; return 3;;
  esac
}

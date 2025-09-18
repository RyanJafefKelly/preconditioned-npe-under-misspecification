#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")

: "${SEED:=0}"

# GNK specifics
: "${N_OBS:=5000}"                      # samples in the observed dataset
# THETA_DEFAULT="3.0 1.0 2.0 0.5"         # (A, B, g, k)
THETA_DEFAULT="2.3663 4.1757 1.7850 0.1001"

THETA="${THETA:-$THETA_DEFAULT}"
read -r -a THETA_ARR <<< "$THETA"
if (( ${#THETA_ARR[@]} != 4 )); then
  echo "Error: need 4 values for THETA (A B g k). Got: ${#THETA_ARR[@]}" >&2
  exit 1
fi

# Preconditioning
: "${N_SIMS:=20000}"
: "${PRECOND_METHOD:=none}"              # none | rejection | smc_abc
: "${Q_PRECOND:=1.0}"

GROUP="th_$(printf 'A%s_B%s_g%s_k%s' "${THETA_ARR[@]}")-n_obs_${N_OBS}-n_sims_${N_SIMS}"
OUTDIR="results/gnk/npe/${GROUP}/seed-${SEED}/${DATE}"
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

cmd=(uv run python -m precond_npe_misspec.pipelines.gnk
  --seed "$SEED"
  --obs_seed "$((10#$SEED + 1234))"
  --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}"
  --n_obs "$N_OBS"

  --precond.method "$PRECOND_METHOD"
  --precond.n_sims "$N_SIMS"
  --precond.q_precond "$Q_PRECOND"

  --posterior.method "npe"
  --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"

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

# cat > "$OUTDIR/entrypoints.json" <<EOF
# {
#   "simulate": "precond_npe_misspec.examples.svar:simulate",
#   "summaries": "precond_npe_misspec.examples.svar:summaries_for_metrics",
#   "sim_kwargs": {"k": $K, "T": $T, "obs_model": "$OBS_MODEL"},
#   "summaries_kwargs": {"k": $K}
# }
# EOF

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

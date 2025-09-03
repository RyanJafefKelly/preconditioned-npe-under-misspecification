# experiments/gnk/run_pnpe.sh
#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")
OUTDIR="results/gnk_pnpe/${DATE}"
mkdir -p "$OUTDIR"

: "${SEED:=0}"

# GNK specifics
: "${N_OBS:=1000}"                     # samples per dataset
THETA_DEFAULT="3.0 1.0 2.0 0.5"        # (A, B, g, k)
THETA="${THETA:-$THETA_DEFAULT}"
read -r -a THETA_ARR <<< "$THETA"
if (( ${#THETA_ARR[@]} != 4 )); then
  echo "Error: need 4 values for THETA (A B g k). Got: ${#THETA_ARR[@]}" >&2
  exit 1
fi

# Preconditioning ABC
: "${N_SIMS:=20000}"
: "${Q_PRECOND:=0.2}"

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

# Summaries + distance
: "${SUMMARIES:=hexadeciles}"               # octile|duodecile|hexadeciles
: "${DISTANCE:=euclidean}"             # euclidean|l1|mmd
: "${MMD_UNBIASED:=0}"
: "${MMD_BANDWIDTH:=}"                 # empty -> median heuristic

cmd=(uv run python -m precond_npe_misspec.pipelines.gnk_pnpe
  --seed "$SEED"
  --theta_true "${THETA_ARR[@]}"
  --n_obs "$N_OBS"
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
  --summaries "$SUMMARIES"
  --distance "$DISTANCE"
  --outdir "$OUTDIR"
)

[[ -n "$MMD_BANDWIDTH" ]] && cmd+=(--mmd_bandwidth "$MMD_BANDWIDTH")
[[ "$MMD_UNBIASED" == "1" ]] && cmd+=(--mmd_unbiased)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"
echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

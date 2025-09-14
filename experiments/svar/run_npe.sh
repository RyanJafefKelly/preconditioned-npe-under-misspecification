# experiments/svar/run_pnpe.sh
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

: "${N_SIMS:=20000}"
: "${Q_PRECOND:=1.0}"

th_parts=()
for i in "${!THETA_ARR[@]}"; do
  th_parts+=("p$((i+1))${THETA_ARR[$i]}")
done
THETA_TAG=$(IFS=_; echo "${th_parts[*]}")

GROUP="th_${THETA_TAG}-K_${K}-T_${T}-n_sims_${N_SIMS}-q_${Q_PRECOND}"

OUTDIR="results/svar/npe/${GROUP}/${DATE}"
mkdir -p "$OUTDIR"


: "${N_POSTERIOR_DRAWS:=20000}"

: "${FLOW_LAYERS:=8}"
: "${NN_WIDTH:=128}"
: "${KNOTS:=10}"
: "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"
: "${MAX_EPOCHS:=500}"
: "${MAX_PATIENCE:=10}"
: "${BATCH_SIZE:=512}"

: "${DISTANCE:=euclidean}"   # euclidean|l1|mahalanobis|mmd
: "${MMD_UNBIASED:=0}"
: "${MMD_BANDWIDTH:=}"       # empty -> median heuristic

cmd=(uv run python -m precond_npe_misspec.pipelines.svar_pnpe
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
  --distance "$DISTANCE"
)

[[ -n "$MMD_BANDWIDTH" ]] && cmd+=(--mmd_bandwidth "$MMD_BANDWIDTH")
[[ "$MMD_UNBIASED" == "1" ]] && cmd+=(--mmd_unbiased)

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
  --method NPE


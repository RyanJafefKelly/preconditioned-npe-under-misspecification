#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")
OUTDIR="results/contaminated_slcp_nle_abc/${DATE}"
mkdir -p "$OUTDIR"

: "${SEED:=1}"

# θ is 5D. Space-separated ASCII numbers only.
THETA_DEFAULT="0.7 -2.9 -1.0 -0.9 0.6"
THETA="${THETA:-$THETA_DEFAULT}"
read -r -a THETA_ARR <<< "$THETA"
if (( ${#THETA_ARR[@]} != 5 )); then
  echo "Error: need 5 values for THETA. Got: ${#THETA_ARR[@]}" >&2
  exit 1
fi

: "${NUM_DRAWS:=5}"
: "${MISSPEC_LEVEL:=1.0}"
: "${N_TRAIN:=5000}"
: "${N_PROPS:=500000}"
: "${Q_ACCEPT:=0.05}"
: "${DISTANCE:=euclidean}"   # euclidean|l1|mahalanobis|mmd
: "${MAHALANOBIS_RIDGE:=1e-3}"
: "${N_REP_SUMMARIES:=1}"    # set >1 for mmd
: "${MMD_BANDWIDTH:=}"       # empty -> median heuristic
: "${MMD_UNBIASED:=0}"

cmd=(uv run python -m precond_npe_misspec.pipelines.contaminated_slcp_nle_abc \
  --seed "$SEED"
  --theta_true "${THETA_ARR[@]}"
  --num_draws "$NUM_DRAWS"
  --misspec_level "$MISSPEC_LEVEL"
  --n_train "$N_TRAIN"
  --n_props "$N_PROPS"
  --q_accept "$Q_ACCEPT"
  --distance "$DISTANCE"
  --mahalanobis_ridge "$MAHALANOBIS_RIDGE"
  --n_rep_summaries "$N_REP_SUMMARIES"
  --outdir "$OUTDIR"
)

[[ -n "$MMD_BANDWIDTH" ]] && cmd+=(--mmd_bandwidth "$MMD_BANDWIDTH")
[[ "$MMD_UNBIASED" == "1" ]] && cmd+=(--mmd_unbiased)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"
echo
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

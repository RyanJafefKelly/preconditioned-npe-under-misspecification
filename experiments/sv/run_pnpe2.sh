# experiments/sv/run_pnpe.sh
#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")
OUTDIR="results/sv_pnpe/${DATE}"
mkdir -p "$OUTDIR"

: "${SEED:=0}"
: "${T:=512}"
: "${LAM:=30.0}"
: "${I0:=200}"
: "${I1:=240}"
: "${K_BLOCKS:=8}"

: "${SIGMA_TRUE:=0.2}"
: "${NU_TRUE:=8.0}"
: "${MU_TRUE:=0.0}"
: "${PHI_TRUE:=0.98}"

: "${N_SIMS:=200000}"
: "${Q_PRECOND:=0.10}"
: "${N_POSTERIOR_DRAWS:=5000}"

: "${FLOW_LAYERS:=8}"
: "${NN_WIDTH:=128}"
: "${KNOTS:=10}"
: "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"
: "${MAX_EPOCHS:=500}"
: "${MAX_PATIENCE:=10}"
: "${BATCH_SIZE:=512}"

: "${SIGMA_MIN:=0.02}"
: "${SIGMA_MAX:=0.5}"
: "${NU_MIN:=3.0}"
: "${NU_MAX:=40.0}"
: "${MU_MIN:=-2.0}"
: "${MU_MAX:=2.0}"
: "${PHI_MIN:=0.90}"
: "${PHI_MAX:=0.999}"

# Basic validations
if (( I0 < 0 || I1 <= I0 || I1 > T )); then
  echo "Error: invalid burst window [I0,I1)=[$I0,$I1) for T=$T" >&2
  exit 1
fi
if (( T % K_BLOCKS != 0 )); then
  echo "Error: require T divisible by K_BLOCKS (got T=$T, K_BLOCKS=$K_BLOCKS)" >&2
  exit 1
fi

cmd=(uv run python -m precond_npe_misspec.pipelines.sv2_pnpe
  --seed "$SEED"
  --T "$T"
  --lam "$LAM"
  --i0 "$I0"
  --i1 "$I1"
  --k_blocks "$K_BLOCKS"
  --sigma_true "$SIGMA_TRUE"
  --nu_true "$NU_TRUE"
  --mu_true "$MU_TRUE"
  --phi_true "$PHI_TRUE"
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
  --sigma_min "$SIGMA_MIN"
  --sigma_max "$SIGMA_MAX"
  --nu_min "$NU_MIN"
  --nu_max "$NU_MAX"
  --mu_min "$MU_MIN"
  --mu_max "$MU_MAX"
  --phi_min "$PHI_MIN"
  --phi_max "$PHI_MAX"
)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"
echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

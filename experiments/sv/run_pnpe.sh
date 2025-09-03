# experiments/sv_burst/run_pnpe.sh
#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

DATE=$(date +"%Y%m%d-%H%M%S")
OUTDIR="results/sv_burst_pnpe/${DATE}"
mkdir -p "$OUTDIR"

: "${SEED:=0}"
: "${T:=512}"
: "${TAU_TRUE:=2.0}"
: "${NU_TRUE:=5.0}"
: "${LAM:=30.0}"
: "${I0:=200}"
: "${I1:=240}"

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

: "${TAU_MIN:=0.5}"
: "${TAU_MAX:=5.0}"
: "${NU_MIN:=3.0}"
: "${NU_MAX:=20.0}"

# Basic window validation
if (( I0 < 0 || I1 <= I0 || I1 > T )); then
  echo "Error: invalid burst window [I0,I1)=[$I0,$I1) for T=$T" >&2
  exit 1
fi

cmd=(uv run python -m precond_npe_misspec.pipelines.sv_pnpe
  --seed "$SEED"
  --tau_true "$TAU_TRUE"
  --nu_true "$NU_TRUE"
  --T "$T"
  --lam "$LAM"
  --i0 "$I0"
  --i1 "$I1"
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
  --tau_min "$TAU_MIN"
  --tau_max "$TAU_MAX"
  --nu_min "$NU_MIN"
  --nu_max "$NU_MAX"
)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"
echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

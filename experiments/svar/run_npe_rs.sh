#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DATE=$(date +"%Y%m%d-%H%M%S")

: "${SEED:=0}"
: "${K:=6}"
: "${T:=1000}"
: "${OBS_MODEL:=true}"     # assumed|true

# θ = (2m offsets) + σ, with m=3 when K=6
THETA_DEFAULT="0.579 -0.143 0.836 0.745 -0.660 -0.254 0.1"
THETA="${THETA:-$THETA_DEFAULT}"; read -r -a THETA_ARR <<< "$THETA"
(( ${#THETA_ARR[@]} == 7 )) || { echo "need 7 values for THETA"; exit 1; }

# Training set size (no preconditioning)
: "${N_SIMS:=8000}"

OUTDIR="results/svar/npe_rs/th_$(printf 'p1%s_p2%s_p3%s_p4%s_p5%s_p6%s_s%s' "${THETA_ARR[@]}")-K_${K}-T_${T}-obs_${OBS_MODEL}-n_sims_${N_SIMS}/seed-${SEED}/${DATE}"
mkdir -p "$OUTDIR"

# Posterior draws
: "${N_POSTERIOR_DRAWS:=20000}"

# Flow hyperparameters
: "${FLOW_LAYERS:=8}"; : "${NN_WIDTH:=128}"; : "${KNOTS:=10}"; : "${INTERVAL:=8.0}"
: "${LEARNING_RATE:=5e-4}"; : "${MAX_EPOCHS:=500}"; : "${MAX_PATIENCE:=10}"; : "${BATCH_SIZE:=64}"

# NPE‑RS (embedding + MMD)
: "${EMBED_DIM:=8}"         # match manual summary dim
: "${EMBED_WIDTH:=128}"     # width for MLP head (or fallback)
: "${EMBED_DEPTH:=1}"
: "${MMD_WEIGHT:=1.0}"
: "${MMD_SUBSAMPLE:=256}"
: "${BANDWIDTH:=median}"    # median|fixed
: "${KERNEL:=rbf}"
: "${WARMUP_EPOCHS:=0}"

cmd=(uv run python -m precond_npe_misspec.pipelines.svar
  --seed "$SEED" --obs_seed "$((10#$SEED + 1234))" --outdir "$OUTDIR"
  --theta_true "${THETA_ARR[@]}" --k "$K" --T "$T" --obs_model "$OBS_MODEL"
  --precond.method "none" --precond.n_sims "$N_SIMS"
  --posterior.method "npe_rs" --posterior.n_posterior_draws "$N_POSTERIOR_DRAWS"
  --npers.embed_dim "$EMBED_DIM" --npers.embed_width "$EMBED_WIDTH" --npers.embed_depth "$EMBED_DEPTH"
  --npers.mmd_weight "$MMD_WEIGHT" --npers.mmd_subsample "$MMD_SUBSAMPLE"
  --npers.bandwidth "$BANDWIDTH" --npers.kernel "$KERNEL" --npers.warmup_epochs "$WARMUP_EPOCHS"
  --flow.flow_layers "$FLOW_LAYERS" --flow.nn_width "$NN_WIDTH" --flow.knots "$KNOTS" --flow.interval "$INTERVAL"
  --flow.learning_rate "$LEARNING_RATE" --flow.max_epochs "$MAX_EPOCHS" --flow.max_patience "$MAX_PATIENCE" --flow.batch_size "$BATCH_SIZE"
)

printf '%q ' "${cmd[@]}" | tee "${OUTDIR}/cmd.txt"; echo
env | sort > "${OUTDIR}/env.txt"
"${cmd[@]}" 2>&1 | tee "${OUTDIR}/stdout.log"

THETA_TARGET_DEFAULT="0.579 -0.143 0.836 0.745 -0.660 -0.254 0.1"
THETA_TARGET="${THETA_TARGET:-$THETA_TARGET_DEFAULT}"; THETA_TARGET="${THETA_TARGET//,/}"; read -r -a THETA_TARGET_ARR <<< "$THETA_TARGET"

# PPD subset/standardise controls (leave PPD_IDX empty to use all)
: "${PPD_IDX:=0 1 2 3 4 5 6}"
: "${PPD_STANDARDISE:=0}"

# Build metrics command
metrics_cmd=(uv run python -m precond_npe_misspec.scripts.metrics_from_samples
  --outdir "$OUTDIR"
  --theta-target "${THETA_TARGET_ARR[@]}"
  --level 0.95 --want-hpdi --want-central
  --method NPE_RS
  --compute-ppd --ppd-entrypoints "$OUTDIR/entrypoints.json"
  --ppd-n 1000 --ppd-metric l2
)

# Append subset indices if provided
if [[ -n "$PPD_IDX" ]]; then
  read -r -a PPD_IDX_ARR <<< "$PPD_IDX"
  metrics_cmd+=( --ppd-idx "${PPD_IDX_ARR[@]}" )
fi

# Optional standardisation
if [[ "$PPD_STANDARDISE" == "1" ]]; then
  metrics_cmd+=( --ppd-standardise )
fi

# Run + record
printf '%q ' "${metrics_cmd[@]}" | tee "${OUTDIR}/cmd_metrics.txt"; echo
"${metrics_cmd[@]}" 2>&1 | tee -a "${OUTDIR}/stdout.log"

# Log the subset used for reproducibility (does not affect aggregation)
[[ -n "${PPD_IDX:-}" ]] && printf '%s\n' "${PPD_IDX_ARR[@]}" > "${OUTDIR}/ppd_idx.txt"

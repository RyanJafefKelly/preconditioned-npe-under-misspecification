#!/usr/bin/env bash
set -euo pipefail

R=${R:-100}                                  # replicates
METHODS_STR=${METHODS_STR:-"npe rnpe pnpe prnpe"}
THETA_TRUE="${THETA_TRUE:-3.0 1.0 2.0 0.5}"
THETA_DAGGER="${THETA_DAGGER:-2.3663 4.1757 1.7850 0.1001}"
N_OBS=${N_OBS:-100}
N_SIMS=${N_SIMS:-20000}
Q_PRECOND=${Q_PRECOND:-0.2}
N_POST=${N_POST:-20000}
LEVEL=${LEVEL:-0.95}
OBS_SEED_OFFSET=${OBS_SEED_OFFSET:-1234}
RESULTS_ROOT=${RESULTS_ROOT:-"results/gnk"}
GROUP_HINT=${GROUP_HINT:-""}                 # optional substring to filter that GROUP

# PBS array flag: -J (PBS Pro) or -t (Torque)
PBS_ARRAY_FLAG=${PBS_ARRAY_FLAG:--J}

mkdir -p logs

# ---- submit worker array ----
read -r -a METHODS <<< "$METHODS_STR"
M=${#METHODS[@]}
TOT=$((R * M))
echo "[launcher] R=$R methods=(${METHODS[@]}) total-tasks=$TOT"

JOBID=$(qsub ${PBS_ARRAY_FLAG} 0-$((TOT-1)) \
  -N gnk_cov_worker \
  -l walltime=00:20:00,mem=8GB,ncpus=1 \
  -o logs/gnk_cov_worker.o\$PBS_JOBID.\$PBS_ARRAY_INDEX \
  -e logs/gnk_cov_worker.e\$PBS_JOBID.\$PBS_ARRAY_INDEX \
  -v R="$R",METHODS_STR="$METHODS_STR",OBS_SEED_OFFSET="$OBS_SEED_OFFSET",\
THETA_TRUE="$THETA_TRUE",THETA_DAGGER="$THETA_DAGGER",N_OBS="$N_OBS",\
N_SIMS="$N_SIMS",Q_PRECOND="$Q_PRECOND",N_POST="$N_POST",LEVEL="$LEVEL" \
  jobs/gnk_coverage_worker.pbs | tr -d '[:space:]')
echo "[launcher] array job id: $JOBID"

JOBBASE="${JOBID%%.*}"       # e.g. 12028067[]    from 12028067[].aqua
DEP_ID="${JOBBASE%%[*}"      # e.g. 12028067

qsub -W depend=afterok:$DEP_ID -N gnk_cov_collate \
  -l walltime=00:10:00,mem=2GB,ncpus=1 \
  -o logs/gnk_cov_collate.o\$PBS_JOBID \
  -e logs/gnk_cov_collate.e\$PBS_JOBID \
  -- /bin/bash -lc "export PATH=\$HOME/.local/bin:\$PATH; \
                    export UV_CACHE_DIR=\${SCRATCH:-\$HOME}/.cache/uv; \
                    export UV_PROJECT_ENVIRONMENT=\$PBS_O_WORKDIR/.venv; \
                    uv run python -m precond_npe_misspec.scripts.collate_gnk_coverage \
                      --results-root \"$RESULTS_ROOT\" \
                      --group-contains \"$GROUP_HINT\" \
                      --methods ${METHODS_STR} \
                      --level $LEVEL"

echo "[launcher] submitted collator with dependency on $JOBID"

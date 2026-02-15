Neural posterior estimation (NPE) with preconditioning and robust inference under model misspecification. Implements NPE, preconditioned NPE (PNPE), robust NPE (RNPE), preconditioned robust NPE (PRNPE), and ABC-based preconditioning (SMC-ABC, ABC-RF). Built on JAX, FlowJAX, NumPyro, and BlackJAX.

## Installation

Requires Python >= 3.13. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

Two dependencies are installed from Git sources (a BlackJAX fork with SMC-ABC support, and a tumour model package); `uv sync` resolves these automatically.

## Experiments

Three simulation studies are included:
The three in paper:
- **Contaminated Weibull** — survival model with contaminated observations
- **BVCBM** — bivariate competing-risks tumour model (uses real cancer data)
- **SVAR** — structural vector autoregression (uses S&P 500 returns)

## Reproducing results

Each experiment is run via shell scripts in `experiments/`, one per method--experiment pair. For example, to run preconditioned robust NPE on the contaminated Weibull experiment:

```bash
bash experiments/contaminated_weibull/run_prnpe.sh
```

Scripts use environment variables for configuration (seed, simulation budget, hyperparameters, etc.) with sensible defaults. Shared defaults live in `configs/<experiment>/method_matrix.sh`. To run with a different seed:

```bash
SEED=42 bash experiments/gnk/run_npe.sh
```

Each script invokes `uv run python -m precond_npe_misspec.pipelines.<experiment>` with the appropriate flags, then computes posterior metrics. Results are written to `results/`.

**Figures and tables** are generated from the saved results using the plotting notebooks in `notebooks/`:

| Experiment | Notebook |
|---|---|
| Contaminated Weibull | `contaminated_weibull_plots.ipynb` |
| BVCBM | `bvcbm_ppc_plots_p{0,1,2,3}.ipynb` |
| SVAR | `svar_plots.ipynb` |

Aggregate metric tables are produced by `scripts/aggregate_metrics.py`.

## Data

All datasets are bundled in `src/precond_npe_misspec/data/` (cancer survival data for BVCBM, S&P 500 log returns for SVAR, precomputed covariance matrices). No external downloads are required. The remaining experiments (contaminated Weibull, g-and-k, alpha-stable SV) use synthetic data generated at runtime by the simulator.

## Directory structure

- `src/precond_npe_misspec/` — importable library code
  - `engine/` — core experiment runner: preconditioning, posterior flow fitting, robust denoising, sampling
  - `pipelines/` — CLI entry points for each experiment (invoked via `python -m precond_npe_misspec.pipelines.<name>`)
  - `examples/` — model definitions (prior, simulator, summary statistics, flow builder) for each experiment
  - `algorithms/` — ABC algorithms (SMC-ABC, ABC-RF)
  - `robust/` — robust inference components (spike-and-slab / heavy-tailed denoising, NumPyro bridge)
  - `utils/` — metrics, distances, MMD, artefact I/O
  - `data/` — bundled datasets (cancer survival data, S&P 500 log returns, covariance matrices)
  - `scripts/` — post-hoc analysis scripts (compute metrics from posterior samples, posterior predictive draws)

- `configs/` — shared experiment defaults and method-matrix shell scripts (per experiment)
- `experiments/` — shell scripts to run each method on each experiment (one script per method-experiment pair)
- `scripts/` — top-level utility scripts (metric aggregation, collation, HPC array launchers)
- `notebooks/` — Jupyter notebooks for plotting, exploration, and figure generation
- `jobs/` — PBS/HPC job scripts
- `tests/` — smoke and integration tests
- `results/` — run artefacts (gitignored)

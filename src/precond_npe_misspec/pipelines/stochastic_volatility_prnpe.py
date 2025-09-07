"""Stochastic volatility model with preconditioned robust NPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from precond_npe_misspec.examples.stochastic_volatility import (
    assumed_dgp as sv_assumed_dgp,
)
from precond_npe_misspec.examples.stochastic_volatility import (
    prior_sample as sv_prior_sample,
)
from precond_npe_misspec.examples.stochastic_volatility import summaries as sv_summaries
from precond_npe_misspec.examples.stochastic_volatility import true_dgp as sv_true_dgp
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec,
    FlowConfig,
    RunConfig,
    default_posterior_flow_builder,
    default_theta_flow_builder,
)
from precond_npe_misspec.pipelines.base_prnpe import run_experiment_prnpe

type Array = jax.Array


@dataclass
class Config:
    # Data‑generating setup
    seed: int = 0
    theta_true: tuple[float, float] = (0.02, 10.0)  # (sigma_rw, nu)
    T: int = 1000
    # Misspecification for x_obs only
    sigma_ms: int = 0
    block_start: int = 50  # 1‑indexed inclusive
    block_end: int = 65  # 1‑indexed inclusive
    # Preconditioning ABC
    n_sims: int = 200_000
    q_precond: float = 0.1
    # Posterior draws
    n_posterior_draws: int = 5000
    # Flow hyperparameters
    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 8.0
    learning_rate: float = 5e-4
    max_epochs: int = 500
    max_patience: int = 10
    batch_size: int = 512
    # Distance options
    distance: Literal["euclidean", "l1", "mmd"] = "euclidean"
    # mmd_unbiased: bool = False
    # mmd_bandwidth: float | None = None
    # Summaries
    # RNPE denoiser
    denoise_model: Literal["student_t", "cauchy", "spike_slab"] = "student_t"
    mcmc_num_warmup: int = 1000
    mcmc_num_samples: int = 2000
    mcmc_thinning: int = 1

    # Summary options
    lags: tuple[int, ...] = (1, 5, 10, 20, 50)  # for sv_summaries  # TODO: HOW SET?


def main(cfg: Config) -> None:
    theta_dim = 2
    s_dim = 5 + len(cfg.lags)

    flow_cfg = FlowConfig(
        flow_layers=cfg.flow_layers,
        nn_width=cfg.nn_width,
        knots=cfg.knots,
        interval=cfg.interval,
        learning_rate=cfg.learning_rate,
        max_epochs=cfg.max_epochs,
        max_patience=cfg.max_patience,
        batch_size=cfg.batch_size,
    )

    run_cfg = RunConfig(
        seed=cfg.seed,
        theta_true=jnp.asarray(cfg.theta_true),
        n_sims=cfg.n_sims,
        q_precond=cfg.q_precond,
        n_posterior_draws=cfg.n_posterior_draws,
        sim_kwargs={},  # not forwarded to simulate()
        batch_size=cfg.batch_size,
    )

    spec = ExperimentSpec(
        name="stochastic_volatility_prnpe",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=lambda key: sv_prior_sample(key),
        true_dgp=lambda key, theta, **kw: sv_true_dgp(
            key,
            theta,
            T=cfg.T,
            sigma_ms=cfg.sigma_ms,
            block_1idx_inclusive=(cfg.block_start, cfg.block_end),
        ),
        simulate=lambda key, theta, **kw: sv_assumed_dgp(key, theta, T=cfg.T),
        summaries=lambda x: sv_summaries(x, lags=cfg.lags),
        build_theta_flow=default_theta_flow_builder(theta_dim),
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
        # make_distance=_distance_factory_from_cfg(cfg),
    )

    result = run_experiment_prnpe(
        spec,
        run_cfg,
        flow_cfg,
        denoise_model=cfg.denoise_model,
        mcmc_num_warmup=cfg.mcmc_num_warmup,
        mcmc_num_samples=cfg.mcmc_num_samples,
        mcmc_thinning=cfg.mcmc_thinning,
    )

    def qtiles(arr: jnp.ndarray) -> tuple[float, float, float]:
        q = jnp.quantile(arr, jnp.array([0.025, 0.5, 0.975]), axis=0)
        return float(q[0]), float(q[1]), float(q[2])

    print(f"Accepted in preconditioning: {int(result.theta_acc_precond.shape[0])}")
    post = result.posterior_samples_at_obs_robust  # (n, 2)
    labels = ("sigma_rw", "nu")
    for i, lab in enumerate(labels):
        lo, med, hi = qtiles(post[:, i])
        print(f"{lab:>9}  median={med:.4f}  95% CI=({lo:.4f}, {hi:.4f})")
    print("Observed summaries:", result.s_obs)
    print("True parameters:", cfg.theta_true)
    print("Posterior mean error:", jnp.mean(post, axis=0) - jnp.asarray(cfg.theta_true))


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))

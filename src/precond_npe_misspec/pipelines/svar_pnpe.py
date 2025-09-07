"""Run preconditioned NPE on the SVAR model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from precond_npe_misspec.examples.svar import assumed_dgp as svar_assumed_dgp
from precond_npe_misspec.examples.svar import default_pairs
from precond_npe_misspec.examples.svar import prior_sample as svar_prior_sample
from precond_npe_misspec.examples.svar import summaries as svar_summaries
from precond_npe_misspec.examples.svar import true_dgp as svar_true_dgp
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec,
    FlowConfig,
    RunConfig,
    default_posterior_flow_builder,
    default_theta_flow_builder,
    run_experiment,
)

type Array = jax.Array


@dataclass
class Config:
    # Data-generating setup
    seed: int = 0
    theta_true: tuple[float, ...] = (
        0.579,
        -0.143,
        0.836,
        0.745,
        -0.660,
        -0.254,
        0.1,
    )
    k: int = 6
    T: int = 1000
    # Preconditioning (ABC) settings
    n_sims: int = 200_000
    q_precond: float = 0.1
    # NPE training
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
    distance: Literal["euclidean", "l1", "mahalanobis", "mmd"] = "euclidean"
    mmd_unbiased: bool = False
    mmd_bandwidth: float | None = None  # None -> median heuristic


def main(cfg: Config) -> None:
    pairs = default_pairs(cfg.k)  # shape (m,2), default for k=6
    m = int(pairs.shape[0])
    theta_dim = 2 * m + 1
    s_dim = 2 * m + 1 + 1

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
        sim_kwargs={},  # fixed below via lambdas
        batch_size=cfg.batch_size,
    )

    spec = ExperimentSpec(
        name="svar_pnpe",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=lambda key: svar_prior_sample(key, pairs=pairs),
        true_dgp=lambda key, theta, **kw: svar_true_dgp(key, theta, k=cfg.k, T=cfg.T, pairs=pairs),
        simulate=lambda key, theta, **kw: svar_assumed_dgp(key, theta, k=cfg.k, T=cfg.T, pairs=pairs),
        summaries=lambda x: svar_summaries(x, pairs=pairs),
        build_theta_flow=default_theta_flow_builder(theta_dim),
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
    )

    result = run_experiment(spec, run_cfg, flow_cfg)

    def qtiles(arr: jnp.ndarray) -> tuple[float, float, float]:
        q = jnp.quantile(arr, jnp.array([0.025, 0.5, 0.975]), axis=0)
        return float(q[0]), float(q[1]), float(q[2])

    print(f"Accepted in preconditioning: {int(result.theta_acc_precond.shape[0])}")
    post = result.posterior_samples_at_obs  # (n, theta_dim)
    for i in range(theta_dim):
        lo, med, hi = qtiles(post[:, i])
        print(f"θ[{i}]  median={med:.4f}  95% CI=({lo:.4f}, {hi:.4f})")
    print("Observed summaries:", result.s_obs)
    print("True parameters:", cfg.theta_true)
    print(
        "Error: ",
        jnp.mean(result.posterior_samples_at_obs, axis=0) - jnp.array(cfg.theta_true),
    )


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))

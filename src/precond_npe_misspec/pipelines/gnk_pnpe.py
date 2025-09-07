from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from precond_npe_misspec.examples.gnk import gnk, ss_octile, true_dgp
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
    # Data‑generating setup
    seed: int = 0
    # θ = (A, B, g, k)
    theta_true: tuple[float, float, float, float] = (3.0, 1.0, 2.0, 0.5)
    n_obs: int = 100

    # Preconditioning (ABC)
    n_sims: int = 25_000
    q_precond: float = 0.2

    # Posterior draws
    n_posterior_draws: int = 20_000

    # Flow hyperparameters
    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 8.0
    learning_rate: float = 5e-4
    max_epochs: int = 500
    max_patience: int = 10
    batch_size: int = 512
    distance: Literal["euclidean", "l1", "mmd"] = "euclidean"

    # Summaries: choose quantiles set
    summaries: Literal["octile", "duodecile", "hexadeciles"] = "octile"

    # Prior ranges (uniform on each component)
    A_min: float = 0.0
    A_max: float = 10.0
    B_min: float = 0.0
    B_max: float = 10.0
    g_min: float = 0.0
    g_max: float = 10.0
    k_min: float = 0.0
    k_max: float = 10.0

    # To save plots to
    outdir: str = ""


# ---------------- GNK helpers ----------------


def _prior_sample_factory(cfg: Config) -> Callable[[Array], jnp.ndarray]:
    lo = jnp.array([cfg.A_min, cfg.B_min, cfg.g_min, cfg.k_min])
    hi = jnp.array([cfg.A_max, cfg.B_max, cfg.g_max, cfg.k_max])

    def prior_sample(key: Array) -> jnp.ndarray:
        u = jax.random.uniform(key, shape=(4,), minval=0.0, maxval=1.0)
        return lo + u * (hi - lo)

    return prior_sample


def simulate_gnk(key: Array, theta: jnp.ndarray, *, n_obs: int) -> jnp.ndarray:
    """Draw n_obs i.i.d. samples from g‑and‑k via its quantile function."""
    z = jax.random.normal(key, (n_obs,), dtype=theta.dtype)
    A, B, g, k = theta
    return gnk(z, A, B, g, k)


# ---------------- Main ----------------


def main(cfg: Config) -> None:
    theta_dim = 4
    s_dim = 7  # TODO: manual specified ... octiles (so 7 dims)

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
        sim_kwargs={"n_obs": cfg.n_obs},
        batch_size=cfg.batch_size,
    )

    prior_sample = _prior_sample_factory(cfg)

    spec = ExperimentSpec(
        name="gnk_pnpe",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=prior_sample,
        true_dgp=lambda key, _, **kw: true_dgp(key, n_obs=cfg.n_obs),  # well‑specified
        simulate=lambda key, theta, **kw: simulate_gnk(key, theta, n_obs=cfg.n_obs),
        summaries=ss_octile,
        build_theta_flow=default_theta_flow_builder(theta_dim),
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
        # keep Euclidean distance (default) for quantile summaries
    )

    res = run_experiment(spec, run_cfg, flow_cfg)
    print("mean: ", res.S_mean)
    print("std: ", res.S_std)
    pseudo_true = (1.17, 1.50, 0.41, 0.23)

    def qtiles(arr: jnp.ndarray) -> tuple[float, float, float]:
        q = jnp.quantile(arr, jnp.array([0.025, 0.5, 0.975]), axis=0)
        return float(q[0]), float(q[1]), float(q[2])

    print(f"Accepted in preconditioning: {int(res.theta_acc_precond.shape[0])}")
    post = res.posterior_samples_at_obs
    for i, name in enumerate(["A", "B", "g", "k"]):
        lo, med, hi = qtiles(post[:, i])
        print(f"{name:>1}  median={med:.4f}  95% CI=({lo:.4f}, {hi:.4f})")
    print("Observed summaries:", res.s_obs)
    print("True parameters:", cfg.theta_true)
    print("Posterior mean error:", jnp.mean(post, axis=0) - jnp.array(pseudo_true))
    # plot marginals
    import matplotlib.pyplot as plt

    for i in range(theta_dim):
        plt.hist(post[:, i], bins=30, alpha=0.5, label=f"Posterior {i}")
        plt.axvline(x=pseudo_true[i], color="r", linestyle="--", label=f"True {i}")
        plt.legend()
        plt.savefig(f"{cfg.outdir}/posterior_{i}.png")
        plt.clf()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))

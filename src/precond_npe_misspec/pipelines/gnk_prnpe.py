"""Run preconditioned robust NPE on g-and-k with misspecified data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from precond_npe_misspec.examples.gnk import gnk as gnk_quantile
from precond_npe_misspec.examples.gnk import (
    ss_robust,  # ss_octile,
    true_dgp,  # added earlier
)
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec,
    FlowConfig,
    RunConfig,
    default_posterior_flow_builder,
)
from precond_npe_misspec.pipelines.base_prnpe import run_experiment_prnpe
from precond_npe_misspec.utils import distances as dist

type Array = jax.Array


def _prior_sample_factory(cfg: Config) -> Callable[[Array], jnp.ndarray]:
    lo = jnp.array([cfg.A_min, cfg.B_min, cfg.g_min, cfg.k_min])
    hi = jnp.array([cfg.A_max, cfg.B_max, cfg.g_max, cfg.k_max])

    def prior_sample(key: Array) -> jnp.ndarray:
        u = jax.random.uniform(key, shape=(4,), minval=0.0, maxval=1.0)
        return lo + u * (hi - lo)

    return prior_sample


def _distance_factory_from_cfg(cfg):  # type: ignore
    if cfg.distance == "euclidean":
        return lambda S_tr_w: dist.euclidean
    if cfg.distance == "l1":
        return lambda S_tr_w: dist.l1
    if cfg.distance == "mmd":
        if cfg.mmd_bandwidth is None:
            return lambda S_tr_w: dist.mmd_rbf_with_median(S_tr_w, sqrt=True, unbiased=cfg.mmd_unbiased)
        else:
            return lambda _S_tr_w: dist.mmd_rbf_factory(cfg.mmd_bandwidth, sqrt=True, unbiased=cfg.mmd_unbiased)
    raise ValueError(f"Unknown distance {cfg.distance}")


def _simulate_gnk(key: Array, theta: jnp.ndarray, n_obs: int) -> jnp.ndarray:
    """Simulate n_obs from g-and-k via its quantile function."""
    A, B, g, k = theta
    z = jax.random.normal(key, (n_obs,), dtype=theta.dtype)
    return gnk_quantile(z, A, B, g, k)


@dataclass
class Config:
    # Data and misspecification
    seed: int = 0
    obs_seed: int = 1234
    outdir: str | None = None
    theta_true: tuple[float, float, float, float] = (3.0, 1.0, 2.0, 0.5)
    n_obs: int = 100
    mix_w: float = 0.9
    mix_mu1: float = 1.0
    mix_var1: float = 2.0
    mix_mu2: float = 7.0
    mix_var2: float = 2.0

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

    # Summaries + distance
    summaries: Literal["octile", "duodecile", "hexadeciles"] = "octile"
    distance: Literal["euclidean", "l1", "mmd"] = "euclidean"
    mmd_unbiased: bool = False
    mmd_bandwidth: float | None = None

    # Denoising model
    denoise_model: Literal["laplace", "laplace_adaptive", "student_t", "cauchy", "spike_slab"] = "laplace_adaptive"
    laplace_alpha: float = 0.3
    laplace_min_scale: float = 0.01
    student_t_scale: float = 0.05
    student_t_df: float = 1.0
    cauchy_scale: float = 0.05
    spike_std: float = 0.01
    slab_scale: float = 0.25
    misspecified_prob: float = 0.5
    learn_prob: bool = False

    # MCMC for denoising
    mcmc_warmup: int = 1000
    mcmc_samples: int = 2000
    mcmc_thin: int = 1

    # Prior ranges (uniform on each component)
    A_min: float = 0.0
    A_max: float = 10.0
    B_min: float = 0.0
    B_max: float = 10.0
    g_min: float = 0.0
    g_max: float = 10.0
    k_min: float = 0.0
    k_max: float = 10.0


def main(cfg: Config) -> None:
    s_dim = 4  # NOTE: Manually set for octile summaries
    theta_dim = 4

    # Flow training hyperparams
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
        obs_seed=cfg.obs_seed,
        outdir=cfg.outdir,
        theta_true=jnp.asarray(cfg.theta_true),
        n_sims=cfg.n_sims,
        q_precond=cfg.q_precond,
        n_posterior_draws=cfg.n_posterior_draws,
        sim_kwargs={"n_obs": cfg.n_obs},  # used by simulate(true does not depend on θ)
        batch_size=cfg.batch_size,
    )

    # Builders
    build_post_flow = default_posterior_flow_builder(theta_dim, s_dim)
    prior_sample = _prior_sample_factory(cfg)

    # Experiment spec
    spec = ExperimentSpec(
        name="gnk_prnpe",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=prior_sample,
        true_dgp=lambda key, _, **kw: true_dgp(key, n_obs=cfg.n_obs),  # well‑specified
        simulate=lambda key, theta, **kw: _simulate_gnk(key, theta, n_obs=cfg.n_obs),
        summaries=lambda y: ss_robust(y),  # y is (n_obs,)
        # build_theta_flow=lambda key, fc: coupling_flow(  #
        #     key=key,
        #     base_dist=Normal(jnp.zeros(theta_dim)),
        #     transformer=bij.RationalQuadraticSpline(knots=fc.knots, interval=fc.interval),
        #     cond_dim=0,
        #     flow_layers=fc.flow_layers,
        #     nn_width=fc.nn_width,
        # ),
        build_posterior_flow=build_post_flow,
        # make_distance=_distance_factory_from_cfg(cfg),
    )

    # Denoiser kwargs
    denoise_kwargs = dict(
        laplace_alpha=cfg.laplace_alpha,
        laplace_min_scale=cfg.laplace_min_scale,
        student_t_scale=cfg.student_t_scale,
        student_t_df=cfg.student_t_df,
        cauchy_scale=cfg.cauchy_scale,
        spike_std=cfg.spike_std,
        slab_scale=cfg.slab_scale,
        misspecified_prob=cfg.misspecified_prob,
        learn_prob=cfg.learn_prob,
    )

    res = run_experiment_prnpe(
        spec,
        run_cfg,
        flow_cfg,
        denoise_model=cfg.denoise_model,
        denoise_kwargs=denoise_kwargs,
        mcmc_num_warmup=cfg.mcmc_warmup,
        mcmc_num_samples=cfg.mcmc_samples,
        mcmc_thinning=cfg.mcmc_thin,
    )

    def qtiles(arr: jnp.ndarray) -> tuple[float, float, float]:
        q = jnp.quantile(arr, jnp.array([0.025, 0.5, 0.975]), axis=0)
        return float(q[0]), float(q[1]), float(q[2])

    print(f"Accepted in preconditioning: {int(res.theta_acc_precond.shape[0])}")

    post = res.posterior_samples_at_obs_robust  # robust posterior samples
    for i in range(theta_dim):
        lo, med, hi = qtiles(post[:, i])
        print(f"θ[{i}]  median={med:.4f}  95% CI=({lo:.4f}, {hi:.4f})")

    if getattr(res, "misspec_probs", None) is not None:
        print("Posterior misspecification probabilities:", res.misspec_probs)

    # Optional: denoising diagnostics if available
    x_samps = getattr(res, "x_denoised_samples", None)
    if x_samps is not None:
        x_mean = jnp.mean(x_samps, axis=0)
        x_lo, x_hi = jnp.quantile(x_samps, jnp.array([0.025, 0.975]), axis=0)
        deltas = res.s_obs - x_mean
        names = [f"s[{i}]" for i in range(spec.s_dim)]
        print("Denoising adjustments (obs − E[x|y]) and denoised 95% intervals:")
        for i, nm in enumerate(names):
            print(
                f"{nm}: obs={res.s_obs[i]:+0.4f}  E[x|y]={x_mean[i]:+0.4f}  "
                f"Δ={deltas[i]:+0.4f}  denoised95%=({x_lo[i]:+0.4f}, {x_hi[i]:+0.4f})"
            )

    print("Observed summaries:", res.s_obs)
    print("True parameters:", cfg.theta_true)
    pseudo_true = (2.3663, 4.1757, 1.7850, 0.1001)
    print(
        "Posterior mean error:",
        jnp.mean(post, axis=0) - jnp.array(pseudo_true),
    )
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

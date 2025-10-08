# src/precond_npe_misspec/pipelines/contaminated_weibull.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import tyro

from precond_npe_misspec.engine.run import (
    NpeRsConfig,
    PosteriorConfig,
    PrecondConfig,
    RobustConfig,
    RunConfig,
    run_experiment,
)
from precond_npe_misspec.engine.spec import (
    ExperimentSpec,
    FlowConfig,
    default_posterior_flow_builder,
)
from precond_npe_misspec.examples import contaminated_weibull as cw
from precond_npe_misspec.examples.embeddings import build as get_embedder

type Array = jax.Array


# -----------------------------
# Prior log-density (for SMC-ABC if enabled)
# k ~ scaled Beta(2,2) on [k_min, k_max]; λ ~ Exp(1)
# -----------------------------
# def _prior_logpdf(
#     theta: jnp.ndarray, k_min: float = 0.05, k_max: float = 1.0
# ) -> jnp.ndarray:
#     th = jnp.asarray(theta)
#     if th.shape[-1] != 2:
#         raise ValueError("theta must have last dimension 2: (k, lambda).")
#     k, lam = th[..., 0], th[..., 1]
#     span = k_max - k_min
#     u = (k - k_min) / span
#     in_k = (u > 0.0) & (u < 1.0)
#     in_lam = lam > 0.0
#     valid = in_k & in_lam

#     # Beta(2,2) pdf on u is 6*u*(1-u); scaled by 1/span
#     log_fk = jnp.log(6.0) + jnp.log(u) + jnp.log1p(-u) - jnp.log(span)
#     log_fl = -lam  # Exp(1)

#     logp = log_fk + log_fl
#     neg_inf = jnp.full_like(k, -jnp.inf)
#     return jnp.where(valid, logp, neg_inf)


def _lognorm_logpdf(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    x = jnp.asarray(x)
    valid = x > 0
    z = (jnp.log(x) - mu) / sigma
    lp = -0.5 * z * z - jnp.log(x) - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.where(valid, lp, -jnp.inf)


def _prior_logpdf_lognorm(
    theta: jnp.ndarray,
    logk_mu: float,
    logk_sigma: float,
    loglam_mu: float,
    loglam_sigma: float,
) -> jnp.ndarray:
    th = jnp.asarray(theta)
    if th.shape[-1] != 2:
        raise ValueError("theta must have last dimension 2: (k, lambda).")
    k, lam = th[..., 0], th[..., 1]
    return _lognorm_logpdf(k, logk_mu, logk_sigma) + _lognorm_logpdf(
        lam, loglam_mu, loglam_sigma
    )


@dataclass
class Config:
    # Data + observation model
    seed: int = 0
    obs_seed: int = 1234
    outdir: str | None = None
    n_obs: int = 2000
    theta_true: tuple[float] = (0.8,)
    obs_model: Literal["assumed", "true"] = "true"
    eps: float = 0.05
    alpha: float = 40.0

    # Engines
    precond: PrecondConfig = PrecondConfig()
    # precond: PrecondConfig = PrecondConfig(
    #     method="rejection", q_precond=0.10, n_sims=200_000
    # )
    posterior: PosteriorConfig = PosteriorConfig()  # default "rnpe"
    robust: RobustConfig = RobustConfig()
    flow: FlowConfig = FlowConfig()
    npers: NpeRsConfig = NpeRsConfig()
    # Prior hyperparameters (log‑normal on k and λ)
    logk_mu: float = -0.4
    logk_sigma: float = 1.2


def _prior_logpdf_factory(cfg: Config) -> Callable[[jax.Array], jnp.ndarray]:
    def _logpdf(theta: jnp.ndarray) -> jnp.ndarray:
        return cw.prior_logpdf(theta, mu=cfg.logk_mu, sigma=cfg.logk_sigma)

    return _logpdf


def _prior_sample_factory(cfg: Config) -> Callable[[jax.Array], jnp.ndarray]:
    def _sampler(key: jax.Array) -> jnp.ndarray:
        return cw.prior_sample(
            key,
            mu=cfg.logk_mu,
            sigma=cfg.logk_sigma,
            # loglam_mu=cfg.loglam_mu,
            # loglam_sigma=cfg.loglam_sigma,
        )

    return _sampler


def _make_spec(cfg: Config) -> ExperimentSpec:
    # Probe summaries to set s_dim
    x_probe = cw.simulate(
        jax.random.key(0), jnp.asarray(cfg.theta_true), n_obs=cfg.n_obs
    )

    # Choose observation DGP
    if cfg.obs_model == "assumed":
        true_dgp = lambda key, theta, **kw: cw.simulate(  # noqa: E731
            key, theta, n_obs=cfg.n_obs
        )
    else:
        true_dgp = lambda key, theta, **kw: cw.true_dgp(  # noqa: E731
            key,
            theta,
            n_obs=cfg.n_obs,
            eps=cfg.eps,  #   alpha=cfg.alpha
        )

    summaries_fn = cw.summaries
    # if cfg.posterior.method == "npe_rs":

    #     def flatten_raw(x: Array) -> Array:
    #         return jnp.ravel(x)  # pass raw data to the embedder

    # summaries_fn = flatten_raw
    x_probe = cw.simulate(
        jax.random.key(0), jnp.asarray(cfg.theta_true), n_obs=cfg.n_obs
    )
    s_dim = int(summaries_fn(x_probe).shape[-1])
    return ExperimentSpec(
        name="contaminated_weibull",
        theta_dim=1,
        s_dim=s_dim,
        prior_sample=_prior_sample_factory(cfg),
        prior_logpdf=_prior_logpdf_factory(cfg),
        true_dgp=true_dgp,
        simulate=lambda key, theta, **kw: cw.simulate(key, theta, n_obs=cfg.n_obs),
        summaries=lambda x: summaries_fn(x),
        build_posterior_flow=default_posterior_flow_builder(1, s_dim),
        build_embedder=get_embedder("iid_deepset"),
        # make_distance=cw.make_distance_ignore_skew,  # ignore skew in ABC distance
        theta_labels=("k",),
        summary_labels=("var", "mean", "min"),
        simulate_path="precond_npe_misspec.examples.contaminated_weibull:simulate",
        summaries_path="precond_npe_misspec.examples.contaminated_weibull:summaries",
        theta_lo=jnp.zeros(1),
        theta_hi=100 * jnp.ones(1),
        # leave theta_lo/hi=None → unconstrained θ training
    )


# -------- entrypoints for metrics/PPD (recorded in artifacts) --------
def simulate(
    key: Array,
    theta: jnp.ndarray,
    *,
    n_obs: int,
    obs_model: Literal["assumed", "true"] = "assumed",
    eps: float = 0.00,
    alpha: float = 40.0,
) -> Array:
    if obs_model == "assumed":
        return cw.simulate(key, theta, n_obs=n_obs)
    return cw.true_dgp(key, theta, n_obs=n_obs, eps=eps)


def summaries_for_metrics(x: Array) -> Array:
    return cw.summaries(x)


def main(cfg: Config) -> None:
    spec = _make_spec(cfg)
    run_experiment(
        spec,
        RunConfig(
            seed=cfg.seed,
            obs_seed=cfg.obs_seed,
            theta_true=cfg.theta_true,
            sim_kwargs={
                "n_obs": cfg.n_obs,
            },
            summaries_kwargs={},  # summaries take no kwargs here
            outdir=cfg.outdir,
            precond=cfg.precond,
            posterior=cfg.posterior,
            robust=cfg.robust,
            batch_size=cfg.flow.batch_size,
            npers=cfg.npers,
        ),
        cfg.flow,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))

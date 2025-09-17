# src/precond_npe_misspec/pipelines/svar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import tyro

from precond_npe_misspec.engine.run import PosteriorConfig, PrecondConfig, RobustConfig, RunConfig, run_experiment
from precond_npe_misspec.examples.svar import assumed_dgp as svar_assumed_dgp
from precond_npe_misspec.examples.svar import default_pairs
from precond_npe_misspec.examples.svar import prior_logpdf as svar_prior_logpdf
from precond_npe_misspec.examples.svar import prior_sample as svar_prior_sample
from precond_npe_misspec.examples.svar import summaries as svar_summaries
from precond_npe_misspec.examples.svar import true_dgp as svar_true_dgp
from precond_npe_misspec.pipelines.base_pnpe import ExperimentSpec, FlowConfig, default_posterior_flow_builder

type Array = jax.Array


@dataclass
class Config:
    # Data + model size
    seed: int = 0
    obs_seed: int = 1234
    outdir: str | None = None
    k: int = 6  # number of channels
    T: int = 1000  # series length
    # θ layout is (2*|pairs|) + 1 (pooled σ). Default below assumes |pairs|=3.
    theta_true: tuple[float, ...] = (0.579, -0.143, 0.836, 0.745, -0.660, -0.254, 0.1)

    # Observation model: "assumed" → well-specified; "true" → misspecified heavy-tails
    obs_model: Literal["assumed", "true"] = "true"
    eps: float = 0.02
    kappa: float = 12.0
    df: float = 1.0
    per_channel: bool = True

    # Engines
    precond: PrecondConfig = PrecondConfig()
    posterior: PosteriorConfig = PosteriorConfig()
    robust: RobustConfig = RobustConfig()
    flow: FlowConfig = FlowConfig()


def _make_spec(cfg: Config) -> ExperimentSpec:
    pairs = default_pairs(cfg.k)  # shape (m, 2)
    m = int(pairs.shape[0])
    theta_dim = 2 * m + 1

    # probe summaries to get s_dim
    x_probe = svar_assumed_dgp(
        jax.random.key(0),
        jnp.asarray(cfg.theta_true),
        k=cfg.k,
        T=cfg.T,
        pairs=pairs,
    )
    s_dim = int(svar_summaries(x_probe, pairs=pairs).shape[-1])

    # choose observation DGP
    if cfg.obs_model == "assumed":
        true_dgp = lambda key, theta, **kw: svar_assumed_dgp(  # noqa: E731
            key, theta, k=cfg.k, T=cfg.T, pairs=pairs
        )
    else:
        true_dgp = lambda key, theta, **kw: svar_true_dgp(  # noqa: E731
            key,
            theta,
            k=cfg.k,
            T=cfg.T,
            pairs=pairs,
            eps=cfg.eps,
            kappa=cfg.kappa,
            df=cfg.df,
            per_channel=cfg.per_channel,
        )

    return ExperimentSpec(
        name="svar",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=lambda key: svar_prior_sample(key, pairs=pairs),
        prior_logpdf=lambda th: svar_prior_logpdf(th, pairs=pairs),
        true_dgp=true_dgp,
        simulate=lambda key, theta, **kw: svar_assumed_dgp(key, theta, k=cfg.k, T=cfg.T, pairs=pairs),
        summaries=lambda x: svar_summaries(x, pairs=pairs),
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
        theta_lo=jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0]),
        theta_hi=jnp.ones(7),
        simulate_path="precond_npe_misspec.examples.svar:simulate",
        summaries_path="precond_npe_misspec.examples.svar:summaries_for_metrics",
        # leave theta bounds None → train in unconstrained space
    )


def simulate(
    key: Array,
    theta: Array,
    *,
    k: int,
    T: int,
    obs_model: Literal["assumed", "true"] = "assumed",
    eps: float = 0.02,
    kappa: float = 12.0,
    df: float = 1.0,
    per_channel: bool = True,
) -> Array:
    pairs = default_pairs(k)
    if obs_model == "assumed":
        return svar_assumed_dgp(key, theta, k=k, T=T, pairs=pairs)
    return svar_true_dgp(
        key,
        theta,
        k=k,
        T=T,
        pairs=pairs,
        eps=eps,
        kappa=kappa,
        df=df,
        per_channel=per_channel,
    )


def summaries_for_metrics(x: jax.Array, *, k: int) -> Array:
    pairs = default_pairs(k)
    return svar_summaries(x, pairs=pairs)


def main(cfg: Config) -> None:
    spec = _make_spec(cfg)
    run_experiment(
        spec,
        RunConfig(
            seed=cfg.seed,
            obs_seed=cfg.obs_seed,
            theta_true=cfg.theta_true,
            sim_kwargs={"k": cfg.k, "T": cfg.T, "obs_model": cfg.obs_model},
            summaries_kwargs={"k": cfg.k},
            outdir=cfg.outdir,
            precond=cfg.precond,
            posterior=cfg.posterior,
            robust=cfg.robust,
            batch_size=cfg.flow.batch_size,
        ),
        cfg.flow,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))

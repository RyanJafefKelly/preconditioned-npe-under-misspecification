# src/precond_npe_misspec/pipelines/alpha_sv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import tyro

from precond_npe_misspec.data.markets import load_sp500_returns_yahoo
from precond_npe_misspec.engine.run import (NpeRsConfig, PosteriorConfig,
                                            PrecondConfig, RobustConfig,
                                            RunConfig, run_experiment)
from precond_npe_misspec.examples.alpha_stable_sv import \
    assumed_dgp as asv_assumed_dgp
from precond_npe_misspec.examples.alpha_stable_sv import \
    prior_sample as asv_prior_sample
from precond_npe_misspec.examples.alpha_stable_sv import \
    summaries_for_metrics as asv_summaries_for_metrics
from precond_npe_misspec.examples.alpha_stable_sv import theta_bounds_3d
from precond_npe_misspec.examples.embeddings import build as get_embedder
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec, FlowConfig, default_posterior_flow_builder)


def _uniform_logpdf_box(
    theta: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray
) -> jnp.ndarray:
    """Log-density of a uniform over the axis-aligned box [lo, hi]."""
    th = jnp.asarray(theta)
    inside = jnp.all((th >= lo) & (th <= hi))
    return jnp.where(inside, 0.0, -jnp.inf)


type Array = jax.Array


# ---------- Public config ----------
@dataclass
class Config:
    # Data + model
    seed: int = 0
    obs_seed: int = 1234
    outdir: str | None = None
    T: int = 1000
    theta_true: tuple[float, float, float] = (0.95, 0.20, 1.30)  # (θ2, θ3, α)
    theta1: float = 0.0  # intercept in log-variance

    obs_model: Literal["assumed", "true"] = "true"

    # Real data (used when obs_model=="true")
    yahoo_start: str = "2013-01-02"
    yahoo_end: str = "2017-02-07"
    yahoo_field: str = "Close"
    yahoo_log_returns: bool = True
    yahoo_standardise: bool = False

    # Engines
    precond: PrecondConfig = PrecondConfig()
    posterior: PosteriorConfig = PosteriorConfig()
    robust: RobustConfig = RobustConfig()
    flow: FlowConfig = FlowConfig()
    npers: NpeRsConfig = NpeRsConfig()


# ---------- Spec builder ----------
def _make_spec(cfg: Config, y_obs: jnp.ndarray | None, T_sim: int) -> ExperimentSpec:
    theta_dim = 3

    x_probe = asv_assumed_dgp(
        jax.random.key(0),
        jnp.asarray(cfg.theta_true, jnp.float32),
        T=T_sim,
        theta1=cfg.theta1,
    )
    s_dim = int(asv_summaries_for_metrics(x_probe).shape[-1])

    if cfg.obs_model == "assumed":
        true_dgp = lambda key, theta, **kw: asv_assumed_dgp(  # noqa: E731
            key, theta, T=T_sim, theta1=cfg.theta1
        )
    else:
        assert y_obs is not None, "y_obs must be provided when obs_model='true'"
        true_dgp = lambda key, _theta, **kw: y_obs  # noqa: E731

    lo, hi = theta_bounds_3d()

    return ExperimentSpec(
        name="alpha_sv",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=lambda key: asv_prior_sample(key),
        prior_logpdf=lambda th: _uniform_logpdf_box(th, lo, hi),
        true_dgp=true_dgp,
        simulate=lambda key, theta, **kw: asv_assumed_dgp(
            key,
            theta,
            T=int(kw.get("T", T_sim)),
            theta1=float(kw.get("theta1", cfg.theta1)),
        ),
        summaries=lambda x: asv_summaries_for_metrics(x),
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
        build_embedder=get_embedder("asv_tcn"),
        theta_labels=(r"θ2 (AR)", r"θ3 (shock)", r"θ4 (α)"),
        summary_labels=(
            "logMAD",
            "acf1|z|",
            "acf5|z|",
            "q99/q95",
            "Pr(|z|>3.5)",
            "ME@2.0",
            "leverage",
        ),
        theta_lo=lo,
        theta_hi=hi,
        simulate_path="precond_npe_misspec.examples.alpha_stable_sv:simulate",
        summaries_path="precond_npe_misspec.examples.alpha_stable_sv:summaries_for_metrics",
    )


# ---------- CLI ----------
def main(cfg: Config) -> None:
    y_obs = None
    T_sim = cfg.T
    if cfg.obs_model == "true":
        y_obs = load_sp500_returns_yahoo(
            start=cfg.yahoo_start,
            end=cfg.yahoo_end,
            field=cfg.yahoo_field,
            log_returns=cfg.yahoo_log_returns,
            standardise=cfg.yahoo_standardise,
        )
        T_sim = int(y_obs.shape[0])

    spec = _make_spec(cfg, y_obs, T_sim)
    run_experiment(
        spec,
        RunConfig(
            seed=cfg.seed,
            obs_seed=cfg.obs_seed,
            theta_true=cfg.theta_true,
            sim_kwargs={
                "T": T_sim,
                "theta1": cfg.theta1,
            },
            summaries_kwargs={},
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

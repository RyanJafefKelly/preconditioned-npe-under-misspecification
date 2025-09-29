# pipelines/gnk.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro

from precond_npe_misspec.engine.run import (NpeRsConfig, PosteriorConfig,
                                            PrecondConfig, RobustConfig,
                                            RunConfig, run_experiment)
from precond_npe_misspec.examples.gnk import gnk as gnk_quantile
from precond_npe_misspec.examples.gnk import prior_logpdf as gnk_prior_logpdf
from precond_npe_misspec.examples.gnk import simulate as gnk_simulate
from precond_npe_misspec.examples.gnk import ss_robust, true_dgp
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec, FlowConfig, default_posterior_flow_builder)

type Array = jax.Array


def _prior_sample_factory(cfg: Config) -> Callable[[jax.Array], jnp.ndarray]:
    lo = jnp.array([cfg.A_min, cfg.B_min, cfg.g_min, cfg.k_min])
    hi = jnp.array([cfg.A_max, cfg.B_max, cfg.g_max, cfg.k_max])

    def prior_sample(key: jax.Array) -> jnp.ndarray:
        u = jax.random.uniform(key, shape=(4,), minval=0.0, maxval=1.0)
        return lo + u * (hi - lo)

    return prior_sample


def _simulate_gnk(key: jax.Array, theta: jnp.ndarray, n_obs: int) -> jnp.ndarray:
    z = jax.random.normal(key, (n_obs,), dtype=theta.dtype)
    A, B, g, k = theta
    return gnk_quantile(z, A, B, g, k)


@dataclass
class Config:
    seed: int = 0
    obs_seed: int = 1234
    outdir: str | None = None
    theta_true: tuple[float, ...] = (3.0, 1.0, 2.0, 0.5)
    n_obs: int = 5000
    precond: PrecondConfig = PrecondConfig()
    posterior: PosteriorConfig = PosteriorConfig()
    flow: FlowConfig = FlowConfig()
    robust: RobustConfig = RobustConfig()
    npers: NpeRsConfig = NpeRsConfig()
    # prior ranges...
    A_min: float = 0.0
    A_max: float = 10.0
    B_min: float = 0.0
    B_max: float = 10.0
    g_min: float = 0.0
    g_max: float = 10.0
    k_min: float = 0.0
    k_max: float = 10.0


def main(cfg: Config) -> None:
    # derive s_dim from a probe
    x_probe = true_dgp(jax.random.key(0), n_obs=cfg.n_obs)

    summaries_fn = ss_robust
    if cfg.posterior.method == "npe_rs":

        def flatten_raw(x: Array) -> Array:
            return jnp.ravel(x)  # pass raw data to the embedder

        summaries_fn = flatten_raw

    s_dim = int(summaries_fn(x_probe).shape[-1])
    spec = ExperimentSpec(
        name="gnk",
        theta_dim=4,
        s_dim=s_dim,
        prior_sample=_prior_sample_factory(cfg),
        prior_logpdf=lambda th: gnk_prior_logpdf(
            th,
            A_min=cfg.A_min,
            A_max=cfg.A_max,
            B_min=cfg.B_min,
            B_max=cfg.B_max,
            g_min=cfg.g_min,
            g_max=cfg.g_max,
            k_min=cfg.k_min,
            k_max=cfg.k_max,
        ),
        true_dgp=lambda key, _, **kw: true_dgp(key, n_obs=cfg.n_obs),
        simulate=lambda key, th, **kw: gnk_simulate(key, th, n_obs=cfg.n_obs),
        summaries=summaries_fn,
        build_posterior_flow=default_posterior_flow_builder(4, s_dim),
        theta_lo=jnp.array([cfg.A_min, cfg.B_min, cfg.g_min, cfg.k_min]),
        theta_hi=jnp.array([cfg.A_max, cfg.B_max, cfg.g_max, cfg.k_max]),
        simulate_path="precond_npe_misspec.examples.gnk:simulate",
        summaries_path="precond_npe_misspec.examples.gnk:summaries_for_metrics",
    )

    # Optional: per‑example embedder. Default: MLP over vector x (GNK is 1D).
    # def _gnk_embedder_builder(
    #     key: jax.Array,
    #     embed_dim: int,
    #     raw_cond_shape: tuple[int, ...],
    #     npers_cfg: NpeRsConfig,
    # ):
    #     in_size = int(raw_cond_shape[0])  # GNK: 1D vector of length n_obs
    #     act = jnn.relu if str(npers_cfg.activation) == "relu" else jnn.tanh
    #     return MLP(
    #         in_size=in_size,
    #         out_size=int(embed_dim),
    #         width_size=int(npers_cfg.embed_width),
    #         depth=int(npers_cfg.embed_depth),
    #         activation=act,
    #         key=key,
    #     )

    # # Attach for NPE‑RS to pick up. Other examples can supply their own builder.
    # setattr(spec, "build_embedder", _gnk_embedder_builder)
    # setattr(spec, "x_shape", x_shape)

    run_experiment(
        spec,
        RunConfig(
            seed=cfg.seed,
            obs_seed=cfg.obs_seed,
            theta_true=cfg.theta_true,
            sim_kwargs={"n_obs": cfg.n_obs},
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

# src/precond_npe_misspec/pipelines/bvcbm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from precond_npe_misspec.engine.run import (
    NpeRsConfig,
    PosteriorConfig,
    PrecondConfig,
    RobustConfig,
    RunConfig,
    run_experiment,
)
from precond_npe_misspec.examples import bvcbm as ex
from precond_npe_misspec.examples.embeddings import build as get_embedder
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec,
    FlowConfig,
    default_posterior_flow_builder,
)


def _uniform_logpdf_box(
    theta: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray
) -> jnp.ndarray:
    th = jnp.asarray(theta)
    inside = jnp.all((th >= lo) & (th <= hi))
    return jnp.where(inside, 0.0, -jnp.inf)


# ---------- Public config ----------
@dataclass
class Config:
    # Data + model
    seed: int = 0
    obs_seed: int = 1234
    outdir: str | None = None
    T: int = 19
    start_volume: float = 50.0
    page: int = 5

    # (p0_1, psc_1, dmax_1, gage_1[h], p0_2, psc_2, dmax_2, gage_2[h], tau[days])
    theta_true: tuple[float, float, float, float, float, float, float, float, float] = (
        0.05,
        0.01,
        30.0,
        48.0,
        0.04,
        0.008,
        30.0,
        48.0,
        7.0,
    )

    obs_model: Literal["synthetic", "real"] = "synthetic"
    summary: Literal["identity", "log"] = "log"
    embedder: str = "asv_tcn"

    # Engines
    precond: PrecondConfig = PrecondConfig()
    posterior: PosteriorConfig = PosteriorConfig()
    robust: RobustConfig = RobustConfig()
    flow: FlowConfig = FlowConfig()
    npers: NpeRsConfig = NpeRsConfig()


# ---------- Spec builder ----------
def _make_spec(cfg: Config, y_obs: jnp.ndarray | None, T_sim: int) -> ExperimentSpec:
    theta_dim = 9

    # --- host simulator via callback (JAX-safe) ---
    T_fixed = int(T_sim)
    sim_py: Callable[[np.ndarray, int], np.ndarray] = ex.simulator_biphasic(
        T=T_fixed, start_volume=cfg.start_volume, page=cfg.page
    )

    def _simulate_np(theta_np: np.ndarray, seed_np: np.ndarray) -> np.ndarray:
        th = np.asarray(theta_np, dtype=float)
        seed = int(np.asarray(seed_np))
        y = sim_py(th, seed)  # tumourmodel call; np.ndarray shape (T_fixed,)
        return np.asarray(y, dtype=np.float32)

    def simulate(key, theta, **kw):
        seed = jax.random.randint(key, (), 0, 2**31 - 1, dtype=jnp.uint32)
        out_shape = jax.ShapeDtypeStruct((T_fixed,), jnp.float32)
        return jax.pure_callback(
            _simulate_np, out_shape, theta, seed, vmap_method="sequential"
        )

    # Summaries
    if cfg.summary == "log":
        summaries = lambda x: jnp.asarray(ex.summary_log(jnp.asarray(x)))
    else:
        summaries = lambda x: jnp.asarray(ex.summary_identity(jnp.asarray(x)))

    # Probe to infer summary dimension
    x_probe = simulate(
        jax.random.key(0), jnp.asarray(cfg.theta_true, jnp.float32), T=T_sim
    )
    s_dim = int(summaries(x_probe).shape[-1])

    # Priors (Uniforms; g_age in hours, τ in days)
    lo, hi = ex.theta_bounds_biphasic(T_sim)
    prior = ex.prior_biphasic(T_sim)
    prior_sample = lambda key: prior.sample(key)
    prior_logpdf = lambda th: prior.log_prob(th)  # or _uniform_logpdf_box(th, lo, hi)

    # Observed DGP
    if cfg.obs_model == "synthetic":
        true_dgp = lambda key, theta, **kw: simulate(key, theta, T=T_sim)
    else:
        assert y_obs is not None, "y_obs must be provided when obs_model='real'"
        true_dgp = lambda key, _theta, **kw: y_obs

    return ExperimentSpec(
        name="bvcbm_biphasic",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=prior_sample,
        prior_logpdf=prior_logpdf,
        true_dgp=true_dgp,
        simulate=simulate,
        summaries=summaries,
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
        build_embedder=get_embedder(cfg.embedder),
        theta_labels=(
            r"$p_0^{(1)}$",
            r"$p_{\mathrm{psc}}^{(1)}$",
            r"$d_{\max}^{(1)}$",
            r"$g_{\mathrm{age}}^{(1)}\,[\mathrm{h}]$",
            r"$p_0^{(2)}$",
            r"$p_{\mathrm{psc}}^{(2)}$",
            r"$d_{\max}^{(2)}$",
            r"$g_{\mathrm{age}}^{(2)}\,[\mathrm{h}]$",
            r"$\tau\,[\mathrm{days}]$",
        ),
        summary_labels=tuple(
            f"{'logV' if cfg.summary=='log' else 'V'}[t={t}]" for t in range(s_dim)
        ),
        theta_lo=lo,
        theta_hi=hi,
        simulate_path="precond_npe_misspec.examples.bvcbm:simulate_biphasic",  # adaptor to add in examples
        summaries_path="precond_npe_misspec.examples.bvcbm:summary_log",
    )


# ---------- CLI ----------
def main(cfg: Config) -> None:
    y_obs = None
    T_sim = cfg.T

    spec = _make_spec(cfg, y_obs, T_sim)
    run_experiment(
        spec,
        RunConfig(
            seed=cfg.seed,
            obs_seed=cfg.obs_seed,
            theta_true=cfg.theta_true,
            sim_kwargs={"T": T_sim},
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

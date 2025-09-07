# src/precond_npe_misspec/pipelines/base_prnpe.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.numpy as jnp
from flowjax.distributions import Normal, Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec,
    FlowConfig,
    RunConfig,
    preconditioning_step,  # reuse your ABC step
)
from precond_npe_misspec.robust.denoise import run_denoising_mcmc

type Array = jax.Array
DistanceFn = Callable[[Array, Array], Array]
DistanceFactory = Callable[[Array], DistanceFn]

# ---------------- Extra builders ----------------


def default_summaries_flow_builder(
    s_dim: int,
) -> Callable[[Array, FlowConfig], eqx.Module]:
    def _builder(key: Array, cfg: FlowConfig) -> eqx.Module:
        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(s_dim)),  # random variable is s
            transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
            cond_dim=None,  # unconditional q(s)
            flow_layers=cfg.flow_layers,
            nn_width=cfg.nn_width,
        )

    return _builder


# ---------------- Helpers ----------------


def _standardise(x: jnp.ndarray, m: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    return (x - m) / (s + 1e-6)


@dataclass
class RobustRunResult:
    theta_acc_precond: jnp.ndarray
    posterior_flow: eqx.Module  # q(theta | s) in original theta scale
    s_flow: eqx.Module  # q(s) in whitened s-space
    x_obs: jnp.ndarray
    s_obs: jnp.ndarray
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean_post: jnp.ndarray
    th_std_post: jnp.ndarray
    denoised_s_samples: jnp.ndarray  # samples ~ p(s | y_obs)
    posterior_samples_at_obs_robust: jnp.ndarray
    misspec_probs: jnp.ndarray | None


# ---------------- Core steps ----------------


def _make_dataset_summaries(
    simulate_summaries: Callable[[Array, int], tuple[jnp.ndarray, jnp.ndarray]],
    n: int,
    batch_size: int | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate (theta, S) safely in batches given a helper closure; mirrors your PNPE batching."""
    if batch_size is None:
        batch_size = max(1, min(n, 2048))
    th_parts, S_parts = [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        th_b, S_b = simulate_summaries(jnp.arange(start, end, dtype=jnp.uint32), end - start)
        th_parts.append(th_b)
        S_parts.append(S_b)
    return jnp.concatenate(th_parts, 0), jnp.concatenate(S_parts, 0)


def _fit_posterior_flow(
    key: Array,
    spec: ExperimentSpec,
    theta_acc: jnp.ndarray,
    S_acc: jnp.ndarray,
    flow_cfg: FlowConfig,
) -> tuple[eqx.Module, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    S_mean, S_std = jnp.mean(S_acc, 0), jnp.std(S_acc, 0) + 1e-8
    th_mean, th_std = jnp.mean(theta_acc, 0), jnp.std(theta_acc, 0) + 1e-8
    S_proc = _standardise(S_acc, S_mean, S_std)
    th_proc = _standardise(theta_acc, th_mean, th_std)

    k_build, k_fit = jax.random.split(key)
    flow0 = spec.build_posterior_flow(k_build, flow_cfg)
    flow_fit, _ = fit_to_data(
        key=k_fit,
        dist=flow0,
        data=(th_proc, S_proc),
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )
    # Create an affine bijection for standardisation
    affine_bijection = bij.Affine(-th_mean / th_std, 1.0 / th_std)
    # Invert the affine bijection, specifying shape and cond_shape=None
    invert_bijection = bij.Invert(bijection=affine_bijection, shape=th_proc.shape[1:], cond_shape=None)
    # Wrap the fitted flow with the invert bijection and a base distribution
    posterior_flow = Transformed(
        base_dist=flow_fit,
        bijection=invert_bijection,
        shape=th_proc.shape[1:],  # shape of theta
        cond_shape=S_proc.shape[1:],  # shape of summaries
    )
    return posterior_flow, S_mean, S_std, th_mean, th_std


def _fit_summaries_flow(
    key: Array,
    s_dim: int,
    S_train: jnp.ndarray,
    flow_cfg: FlowConfig,
) -> eqx.Module:
    S_mean, S_std = jnp.mean(S_train, 0), jnp.std(S_train, 0) + 1e-8
    S_proc = _standardise(S_train, S_mean, S_std)

    k_build, k_fit = jax.random.split(key)
    s_flow0 = default_summaries_flow_builder(s_dim)(k_build, flow_cfg)
    s_flow_fit, _ = fit_to_data(
        key=k_fit,
        dist=s_flow0,
        data=S_proc,  # unconditional MLE on summaries
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )

    return s_flow_fit


def _sample_robust_posterior(
    key: Array,
    posterior_flow: eqx.Module,  # q(theta | s_w)
    s_w_samples: jnp.ndarray,  # (M, s_dim), whitened
    n_draws: int,
    batch_size: int = 512,
) -> jnp.ndarray:
    """Monte Carlo mixture: for each draw pick a denoised s, then sample theta."""
    M = s_w_samples.shape[0]
    idx = jax.random.randint(key, (n_draws,), minval=0, maxval=M)

    # Chunked sampling to avoid large vmaps on condition
    out = []
    for start in range(0, n_draws, batch_size):
        end = min(start + batch_size, n_draws)
        k_chunk = jax.random.split(jax.random.fold_in(key, start), end - start)
        s_chunk = s_w_samples[idx[start:end]]

        # vmap single-sample draws
        def _draw(kk, s_w):  # type: ignore
            return posterior_flow.sample(kk, (1,), condition=s_w)[0]

        thetas = jax.vmap(_draw)(k_chunk, s_chunk)
        out.append(thetas)
    return jnp.concatenate(out, axis=0)


# ---------------- Orchestration ----------------


def run_experiment_prnpe(
    spec: ExperimentSpec,
    run: RunConfig,
    flow_cfg: FlowConfig,
    denoise_model: Literal["laplace", "laplace_adaptive", "student_t", "cauchy", "spike_slab"] = "spike_slab",
    denoise_kwargs: dict[str, Any] | None = None,
    mcmc_num_warmup: int = 1000,
    mcmc_num_samples: int = 2000,
    mcmc_thinning: int = 1,
) -> RobustRunResult:
    rng = jax.random.key(run.seed)
    sim_kwargs = {} if run.sim_kwargs is None else dict(run.sim_kwargs)

    # Observed data
    rng, k_obs = jax.random.split(rng)
    x_obs = spec.true_dgp(k_obs, jnp.asarray(run.theta_true), **sim_kwargs)
    print("x_obs.shape, ", x_obs.shape)
    s_obs = spec.summaries(x_obs)
    print("s_obs, ", s_obs)

    # Preconditioning ABC -> accepted (theta, S)
    rng, k_pre = jax.random.split(rng)
    theta_acc, S_acc = preconditioning_step(spec, k_pre, s_obs, run, flow_cfg)

    # Train q(theta|s) on accepted set
    rng, k_postfit = jax.random.split(rng)
    posterior_flow, S_mean, S_std, th_mean, th_std = _fit_posterior_flow(k_postfit, spec, theta_acc, S_acc, flow_cfg)

    # Train q(s) on prior predictive summaries (reuse S_acc for speed, or draw fresh if you prefer)
    # Using S_acc is fine because RNPE only needs a good density baseline near posterior support.
    rng, k_sfit = jax.random.split(rng)
    s_flow = _fit_summaries_flow(k_sfit, spec.s_dim, S_acc, flow_cfg)

    # Denoise observed summaries: work in whitened s-space (same stats as for q(theta|s))
    s_obs_w = _standardise(s_obs, S_mean, S_std)
    print("s_obs_w: ", s_obs_w)

    rng, k_mcmc = jax.random.split(rng)
    denoise_kwargs = {} if denoise_kwargs is None else dict(denoise_kwargs)
    mcmc_out = run_denoising_mcmc(
        k_mcmc,
        y_obs_w=s_obs_w,
        flow_s=s_flow,
        model=denoise_model,
        num_warmup=mcmc_num_warmup,
        num_samples=mcmc_num_samples,
        thinning=mcmc_thinning,
        **denoise_kwargs,
    )
    s_denoised_w = mcmc_out["x_samples"]  # (M, s_dim)

    # Robust posterior by Monte Carlo mixture over denoised s
    rng, k_mix = jax.random.split(rng)
    th_samps_robust = _sample_robust_posterior(k_mix, posterior_flow, s_denoised_w, run.n_posterior_draws)

    return RobustRunResult(
        theta_acc_precond=theta_acc,
        posterior_flow=posterior_flow,
        s_flow=s_flow,
        x_obs=x_obs,
        s_obs=s_obs,
        S_mean=S_mean,
        S_std=S_std,
        th_mean_post=th_mean,
        th_std_post=th_std,
        denoised_s_samples=s_denoised_w,
        posterior_samples_at_obs_robust=th_samps_robust,
        misspec_probs=mcmc_out.get("misspec_probs", None),
    )

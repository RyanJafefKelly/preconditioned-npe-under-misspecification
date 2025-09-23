"""Base code for preconditioned robust NPE (PNPE) experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.nn as jnn
import jax.numpy as jnp
from flowjax.distributions import Normal
from flowjax.distributions import Transformed as _Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

from precond_npe_misspec.pipelines.base_pnpe import ExperimentSpec, FlowConfig, RunConfig, preconditioning_step
from precond_npe_misspec.robust.denoise import run_denoising_mcmc
from precond_npe_misspec.utils.artifacts import save_artifacts

type Array = jax.Array
DistanceFn = Callable[[Array, Array], Array]
DistanceFactory = Callable[[Array], DistanceFn]


def _to_unconstrained(theta: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    p = (theta - lo) / (hi - lo)
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _from_unconstrained(u: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    return lo + (hi - lo) * jnn.sigmoid(u)


class _BoundedPosterior(eqx.Module):  # type: ignore[misc]
    base: eqx.Module  # flow over u (standardised)
    u_mean: jnp.ndarray  # (theta_dim,)
    u_std: jnp.ndarray  # (theta_dim,)
    lo: jnp.ndarray  # (theta_dim,)
    hi: jnp.ndarray  # (theta_dim,)

    def sample(self, key: Array, shape: tuple[int, ...], *, condition: Array) -> Array:
        u_proc = self.base.sample(key, shape, condition=condition)
        u = u_proc * self.u_std + self.u_mean
        return _from_unconstrained(u, self.lo, self.hi)


def default_summaries_flow_builder(
    s_dim: int,
) -> Callable[[Array, FlowConfig], eqx.Module]:
    def _builder(key: Array, cfg: FlowConfig) -> eqx.Module:
        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(s_dim)),  # base dist is summaries
            transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
            cond_dim=None,  # unconditional q(s) ... used in denoising step
            flow_layers=cfg.flow_layers,
            nn_width=cfg.nn_width,
        )

    return _builder


def _standardise(x: jnp.ndarray, m: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    return (x - m) / (s + 1e-8)


@dataclass
class RobustRunResult:
    theta_acc_precond: jnp.ndarray
    S_acc_precond: jnp.ndarray
    posterior_flow: eqx.Module  # q(theta | s)
    s_flow: eqx.Module  # q(s)
    x_obs: jnp.ndarray
    s_obs: jnp.ndarray
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean_post: jnp.ndarray
    th_std_post: jnp.ndarray
    denoised_s_samples: jnp.ndarray  # samples ~ p(s | y_obs)
    posterior_samples_at_obs_robust: jnp.ndarray
    misspec_probs: jnp.ndarray | None
    loss_history_theta: Any | None
    loss_history_s: Any | None


def _make_dataset_summaries(
    simulate_summaries: Callable[[Array, int], tuple[jnp.ndarray, jnp.ndarray]],
    n: int,
    batch_size: int | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate (theta, S) pairs in batches."""
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
) -> tuple[eqx.Module, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Standardise summaries
    S_mean, S_std = jnp.mean(S_acc, 0), jnp.std(S_acc, 0) + 1e-8
    S_proc = _standardise(S_acc, S_mean, S_std)

    k_build, k_fit = jax.random.split(key)
    flow0 = spec.build_posterior_flow(k_build, flow_cfg)

    # If bounds are provided, train flow in unconstrained space u
    if (getattr(spec, "theta_lo", None) is not None) and (getattr(spec, "theta_hi", None) is not None):
        lo = jnp.asarray(spec.theta_lo)
        hi = jnp.asarray(spec.theta_hi)

        u_acc = _to_unconstrained(theta_acc, lo, hi)
        u_mean, u_std = jnp.mean(u_acc, 0), jnp.std(u_acc, 0) + 1e-8
        u_proc = (u_acc - u_mean) / u_std

        flow_fit, losses_theta = fit_to_data(
            key=k_fit,
            dist=flow0,
            data=(u_proc, S_proc),
            learning_rate=flow_cfg.learning_rate,
            max_epochs=flow_cfg.max_epochs,
            max_patience=flow_cfg.max_patience,
            batch_size=flow_cfg.batch_size,
            show_progress=True,
        )
        posterior_flow = _BoundedPosterior(
            base=flow_fit,
            u_mean=u_mean,
            u_std=u_std,
            lo=lo,
            hi=hi,
        )
        # Keep θ-domain stats for reporting/artifacts
        th_mean, th_std = jnp.mean(theta_acc, 0), jnp.std(theta_acc, 0) + 1e-8
        return posterior_flow, S_mean, S_std, th_mean, th_std, losses_theta

    # Fallback: original unconstrained training on θ
    th_mean, th_std = jnp.mean(theta_acc, 0), jnp.std(theta_acc, 0) + 1e-8
    th_proc = _standardise(theta_acc, th_mean, th_std)

    flow_fit, losses_theta = fit_to_data(
        key=k_fit,
        dist=flow0,
        data=(th_proc, S_proc),
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )
    Invert = cast(Any, bij.Invert)
    TransformedD = cast(Any, _Transformed)
    affine_bij = bij.Affine(-th_mean / th_std, 1.0 / th_std)
    invert_bij = Invert(affine_bij)
    posterior_flow = TransformedD(base_dist=flow_fit, bijection=invert_bij)
    return posterior_flow, S_mean, S_std, th_mean, th_std, losses_theta


def _fit_summaries_flow(
    key: Array,
    s_dim: int,
    S_train: jnp.ndarray,
    flow_cfg: FlowConfig,
) -> tuple[eqx.Module, Any]:
    S_mean, S_std = jnp.mean(S_train, 0), jnp.std(S_train, 0) + 1e-8
    S_proc = _standardise(S_train, S_mean, S_std)

    k_build, k_fit = jax.random.split(key)
    s_flow0 = default_summaries_flow_builder(s_dim)(k_build, flow_cfg)
    s_flow_fit, losses_s = fit_to_data(
        key=k_fit,
        dist=s_flow0,
        data=S_proc,  # unconditional MLE on summaries
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )

    return s_flow_fit, losses_s


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
        def _draw(kk: Array, s_w: Array) -> Array:
            return cast(Array, posterior_flow.sample(kk, (1,), condition=s_w)[0])

        thetas = jax.vmap(_draw)(k_chunk, s_chunk)
        out.append(thetas)
    return jnp.concatenate(out, axis=0)


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
    obs_seed = jax.random.key(run.obs_seed)

    sim_kwargs = {} if run.sim_kwargs is None else dict(run.sim_kwargs)

    # Observed data
    rng, k_obs = jax.random.split(obs_seed)
    x_obs = spec.true_dgp(k_obs, jnp.asarray(run.theta_true), **sim_kwargs)
    s_obs = spec.summaries(x_obs)

    # Preconditioning ABC -> accepted (theta, S)
    rng, k_pre = jax.random.split(rng)
    theta_acc, S_acc = preconditioning_step(spec, k_pre, s_obs, run, flow_cfg)

    # Train q(theta|s) on accepted set
    rng, k_postfit = jax.random.split(rng)
    posterior_flow, S_mean, S_std, th_mean, th_std, losses_theta = _fit_posterior_flow(
        k_postfit, spec, theta_acc, S_acc, flow_cfg
    )

    rng, k_sfit = jax.random.split(rng)
    s_flow, losses_s = _fit_summaries_flow(k_sfit, spec.s_dim, S_acc, flow_cfg)

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

    # Robust posterior conditioned on denoised s
    rng, k_mix = jax.random.split(rng)
    th_samps_robust = _sample_robust_posterior(k_mix, posterior_flow, s_denoised_w, run.n_posterior_draws)

    result = RobustRunResult(
        theta_acc_precond=theta_acc,
        S_acc_precond=S_acc,
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
        loss_history_theta=losses_theta,
        loss_history_s=losses_s,
    )

    if run.outdir:
        save_artifacts(
            outdir=run.outdir,
            spec={
                "name": spec.name,
                "theta_dim": spec.theta_dim,
                "s_dim": spec.s_dim,
                "theta_labels": (
                    list(spec.theta_labels)  # type: ignore
                    if getattr(spec, "theta_labels", None)
                    else None
                ),
                "summary_labels": (
                    list(spec.summary_labels)  # type: ignore
                    if getattr(spec, "summary_labels", None)
                    else None
                ),
            },
            run_cfg=asdict(run),
            flow_cfg=asdict(flow_cfg),
            posterior_flow=result.posterior_flow,
            s_flow=result.s_flow,
            s_obs=result.s_obs,
            posterior_samples=None,  # PNPE path not used here
            robust_posterior_samples=result.posterior_samples_at_obs_robust,
            theta_acc=result.theta_acc_precond,
            S_acc=result.S_acc_precond,
            S_mean=result.S_mean,
            S_std=result.S_std,
            th_mean=result.th_mean_post,
            th_std=result.th_std_post,
            denoised_s_samples=result.denoised_s_samples,
            misspec_probs=result.misspec_probs,
            loss_history_theta=result.loss_history_theta,
            loss_history_s=result.loss_history_s,
            theta_labels=(
                list(spec.theta_labels) if getattr(spec, "theta_labels", None) else None  # type: ignore
            ),
            summary_labels=(
                list(spec.summary_labels)  # type: ignore
                if getattr(spec, "summary_labels", None)
                else None
            ),
        )
    return result

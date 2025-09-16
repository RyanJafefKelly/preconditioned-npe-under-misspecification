# src/precond_npe_misspec/engine/robust.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.numpy as jnp
from flowjax.distributions import Normal
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

from precond_npe_misspec.robust.denoise import run_denoising_mcmc

type Array = jax.Array
EPS = 1e-8
ModelName = Literal["laplace", "laplace_adaptive", "student_t", "cauchy", "spike_slab"]


def _standardise(x: jnp.ndarray, m: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    return (x - m) / (s + EPS)


def _default_summaries_flow_builder(
    s_dim: int,
) -> Callable[[Array, Any], eqx.Module]:
    def _builder(key: Array, cfg: Any) -> eqx.Module:
        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(s_dim)),
            transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
            cond_dim=None,  # unconditional q(s)
            flow_layers=cfg.flow_layers,
            nn_width=cfg.nn_width,
        )

    return _builder


def fit_s_flow(
    key: Array,
    s_dim: int,
    S_train: jnp.ndarray,
    flow_cfg: Any,
) -> tuple[eqx.Module, list[float]]:
    """
    Fit an unconditional flow q(s) on whitened summaries S_train.
    Returns (s_flow, losses). Whitening stats are internal.
    """
    # TODO: NOW ASSUMING PASSED IN WHITENED SUMMARIES
    S_mean = jnp.mean(S_train, 0)
    S_std = jnp.std(S_train, 0) + EPS
    S_proc = _standardise(S_train, S_mean, S_std)

    k_build, k_fit = jax.random.split(key)
    s_flow0 = _default_summaries_flow_builder(s_dim)(k_build, flow_cfg)

    s_flow_fit, losses_any = fit_to_data(
        key=k_fit,
        dist=s_flow0,
        data=S_proc,
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )
    losses_s = cast(list[float], losses_any)

    # Bind whitening stats into the model so it expects whitened inputs consistent with training.
    # We wrap by composing an Affine to map raw s -> whitened s seen by s_flow_fit.
    # y = (s - mean)/std  => y = A*s + b with A=1/std, b=-mean/std.
    class _WhitenedSummaries(eqx.Module):  # type: ignore[misc]
        base: eqx.Module
        mean: Array
        std: Array

        def log_prob(self, s: Array) -> Array:
            y = (s - self.mean) / self.std
            base_lp = cast(Array, self.base.log_prob(y))
            return base_lp - jnp.sum(jnp.log(self.std))

        def sample(self, key: Array, shape: tuple[int, ...]) -> Array:
            y = cast(Array, self.base.sample(key, shape))
            return y * self.std + self.mean

    s_flow_wrapped = _WhitenedSummaries(base=s_flow_fit, mean=S_mean, std=S_std)
    return s_flow_wrapped, losses_s


def denoise_s(
    key: Array,
    s_obs_w: Array,
    q_s: eqx.Module,
    robust_cfg: Any | None = None,
) -> tuple[Array, Array | None]:
    """
    Run denoising MCMC given whitened observed summaries s_obs_w and q(s).
    robust_cfg may provide:
      denoise_model, laplace_alpha, laplace_min_scale, student_t_scale, student_t_df,
      cauchy_scale, spike_std, slab_scale, misspecified_prob, learn_prob,
      mcmc_warmup, mcmc_samples, mcmc_thin.
    Falls back to sensible defaults when fields are missing.
    """
    rc = robust_cfg if robust_cfg is not None else object()

    model: ModelName = cast(ModelName, getattr(rc, "denoise_model", "spike_slab"))
    laplace_alpha: float = float(getattr(rc, "laplace_alpha", 0.3))
    laplace_min_scale: float = float(getattr(rc, "laplace_min_scale", 0.01))
    student_t_scale: float = float(getattr(rc, "student_t_scale", 0.05))
    student_t_df: float = float(getattr(rc, "student_t_df", 1.0))
    cauchy_scale: float = float(getattr(rc, "cauchy_scale", 0.05))
    spike_std: float = float(getattr(rc, "spike_std", 0.01))
    slab_scale: float = float(getattr(rc, "slab_scale", 0.25))
    misspecified_prob: float = float(getattr(rc, "misspecified_prob", 0.5))
    learn_prob: bool = bool(getattr(rc, "learn_prob", False))

    out = run_denoising_mcmc(
        key,
        y_obs_w=s_obs_w,
        flow_s=q_s,
        model=model,
        num_warmup=int(getattr(rc, "mcmc_warmup", 1000)),
        num_samples=int(getattr(rc, "mcmc_samples", 2000)),
        thinning=int(getattr(rc, "mcmc_thin", 1)),
        laplace_alpha=laplace_alpha,
        laplace_min_scale=laplace_min_scale,
        student_t_scale=student_t_scale,
        student_t_df=student_t_df,
        cauchy_scale=cauchy_scale,
        spike_std=spike_std,
        slab_scale=slab_scale,
        misspecified_prob=misspecified_prob,
        learn_prob=learn_prob,
    )
    s_denoised_w = out["x_samples"]
    misspec_probs = out.get("misspec_probs", None)
    return s_denoised_w, misspec_probs


def sample_robust_posterior(
    key: Array,
    posterior_flow: eqx.Module,  # q(theta | s_w)
    s_w_samples: Array,  # (M, s_dim), whitened
    n_draws: int,
    *,
    batch_size: int = 512,
) -> Array:
    """
    Monte Carlo mixture: draw θ by conditioning on randomly selected denoised s_w.
    """
    M = int(s_w_samples.shape[0])
    if M <= 0:
        raise ValueError("s_w_samples must have shape (M, s_dim) with M>0.")

    idx = jax.random.randint(key, (n_draws,), minval=0, maxval=M)
    out: list[Array] = []
    for start in range(0, n_draws, batch_size):
        end = min(start + batch_size, n_draws)
        k_chunk = jax.random.split(jax.random.fold_in(key, start), end - start)
        s_chunk = s_w_samples[idx[start:end]]

        def _draw(kk: Array, s_w: Array) -> Array:
            return cast(Array, posterior_flow.sample(kk, (1,), condition=s_w)[0])

        thetas = jax.vmap(_draw)(k_chunk, s_chunk)
        out.append(thetas)
    return jnp.concatenate(out, axis=0)

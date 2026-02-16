# src/precond_npe_misspec/engine/posterior.py
from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.numpy as jnp
from flowjax.distributions import Transformed as _Transformed
from flowjax.train import fit_to_data

type Array = jax.Array
EPS = 1e-8


def _to_unconstrained(theta: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Map componentwise (lo, hi) to R via logit."""
    p = (theta - lo) / (hi - lo)
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _from_unconstrained(u: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Inverse map from R to (lo, hi) via sigmoid."""
    return lo + (hi - lo) * jax.nn.sigmoid(u)


class _BoundedPosterior(eqx.Module):  # type: ignore[misc]
    """Wrap a conditional flow over whitened unconstrained params u_proc to sample θ in bounds."""

    base: eqx.Module
    u_mean: jnp.ndarray  # (θ_dim,)
    u_std: jnp.ndarray  # (θ_dim,)
    lo: jnp.ndarray  # (θ_dim,)
    hi: jnp.ndarray  # (θ_dim,)

    def sample(self, key: Array, shape: tuple[int, ...], *, condition: jnp.ndarray) -> jnp.ndarray:
        u_proc = self.base.sample(key, shape, condition=condition)
        u = u_proc * self.u_std + self.u_mean
        return _from_unconstrained(u, self.lo, self.hi)


def _standardise(x: jnp.ndarray, m: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    return (x - m) / (s + EPS)


def fit_posterior_flow(
    key: Array,
    spec: Any,
    theta_train: jnp.ndarray,
    S_train: jnp.ndarray,
    flow_cfg: Any,
) -> tuple[eqx.Module, Array, Array, Array, Array, list[float]]:
    """
    Fit conditional flow q(θ | s). Returns:
      flow, S_mean, S_std, th_mean, th_std, losses
    Flow samples accept whitened s via 'condition'.
    """
    # Standardise summaries for conditioning stability
    raw_mean = jnp.mean(S_train, axis=0)
    raw_std = jnp.std(S_train, axis=0)
    cap = jnp.asarray(1e30, dtype=raw_std.dtype)
    S_mean = jnp.clip(raw_mean, a_min=-cap, a_max=cap)
    S_std = jnp.clip(raw_std, a_min=None, a_max=cap) + EPS

    S_proc = _standardise(S_train, S_mean, S_std)

    k_build, k_fit = jax.random.split(key)
    flow0 = spec.build_posterior_flow(k_build, flow_cfg)

    # If bounds provided, train in unconstrained space then wrap
    has_bounds = (getattr(spec, "theta_lo", None) is not None) and (getattr(spec, "theta_hi", None) is not None)
    if has_bounds:
        lo = jnp.asarray(spec.theta_lo)
        hi = jnp.asarray(spec.theta_hi)

        u_train = _to_unconstrained(theta_train, lo, hi)
        u_mean = jnp.mean(u_train, axis=0)
        u_std = jnp.std(u_train, axis=0) + EPS
        u_proc = _standardise(u_train, u_mean, u_std)

        flow_fit, losses = fit_to_data(
            key=k_fit,
            dist=flow0,
            data=(u_proc, S_proc),
            learning_rate=flow_cfg.learning_rate,
            max_epochs=flow_cfg.max_epochs,
            max_patience=flow_cfg.max_patience,
            batch_size=flow_cfg.batch_size,
            show_progress=True,
        )
        posterior_flow = _BoundedPosterior(base=flow_fit, u_mean=u_mean, u_std=u_std, lo=lo, hi=hi)
        # Report θ-domain stats for artefacts
        th_mean = jnp.mean(theta_train, axis=0)
        th_std = jnp.std(theta_train, axis=0) + EPS
        return posterior_flow, S_mean, S_std, th_mean, th_std, losses

    # Fallback: train directly on θ, then un-standardise via an affine bijection
    th_mean = jnp.mean(theta_train, axis=0)
    th_std = jnp.std(theta_train, axis=0) + EPS
    th_proc = _standardise(theta_train, th_mean, th_std)

    flow_fit, losses = fit_to_data(
        key=k_fit,
        dist=flow0,
        data=(th_proc, S_proc),
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )

    # Map from standardised θ to raw θ: inverse of Affine(b=-μ/σ, s=1/σ) gives σ*y + μ
    Invert = cast(Any, bij.Invert)
    TransformedD = cast(Any, _Transformed)
    affine_bij = bij.Affine(-th_mean / th_std, 1.0 / th_std)
    posterior_flow = TransformedD(base_dist=flow_fit, bijection=Invert(affine_bij))
    return posterior_flow, S_mean, S_std, th_mean, th_std, losses


def sample_posterior(
    key: Array,
    posterior_flow: eqx.Module,
    s_obs_w: jnp.ndarray,
    n_draws: int,
    *,
    batch_size: int = 4096,
) -> jnp.ndarray:
    """
    Sample θ ~ q(θ | s_obs_w). Works for both bounded and unbounded wrappers.
    """
    out = []
    for start in range(0, n_draws, batch_size):
        end = min(start + batch_size, n_draws)
        k_chunk = jax.random.fold_in(key, start)
        out.append(posterior_flow.sample(k_chunk, (end - start,), condition=s_obs_w))
    return jnp.concatenate(out, axis=0)

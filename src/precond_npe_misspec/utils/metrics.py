# src/precond_npe_misspec/utils/metrics.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp

type Array = jax.Array


def _ensure_2d(x: Array) -> Array:
    x = jnp.asarray(x)
    return x if x.ndim == 2 else x.reshape((-1, 1))


def _q(a: Array, q: Array | float, *, axis: int, method: str = "linear") -> Array:
    """Quantile with JAX API compatibility."""
    try:
        return jnp.quantile(a, q, axis=axis, method=method)  # jax>=0.4.12
    except TypeError:  # older JAX
        return jnp.quantile(a, q, axis=axis, interpolation=method)


def posterior_mean_sd(samples: Array) -> tuple[Array, Array]:
    s = _ensure_2d(samples)  # (K,D)
    mu = jnp.mean(s, axis=0)  # (D,)
    sd = jnp.std(s, axis=0)  # population SD over draws; OK for posterior SD
    return mu, sd


def central_ci(samples: Array, alpha: float = 0.05) -> tuple[Array, Array]:
    """Quantile CI per dimension. samples: (K,D). Returns lo, hi shape (D,)."""
    s = _ensure_2d(samples)
    lo, hi = _q(s, jnp.array([alpha / 2, 1.0 - alpha / 2]), axis=0, method="linear")

    return lo, hi


def _hpdi_1d(x: Array, alpha: float) -> tuple[Array, Array]:
    xs = jnp.sort(x)
    n = xs.size
    m = jnp.int32(jnp.ceil((1.0 - alpha) * n))
    starts, ends = xs[: n - m + 1], xs[m - 1 :]
    i = jnp.argmin(ends - starts)
    return starts[i], ends[i]


def hpdi(samples: Array, alpha: float = 0.05) -> tuple[Array, Array]:
    """Minimum‑width HPDI per dimension from samples. Returns lo, hi shape (D,)."""
    s = _ensure_2d(samples)
    lo, hi = jax.vmap(lambda col: _hpdi_1d(col, alpha), in_axes=1, out_axes=(0, 0))(s)
    return lo, hi


def covered(lo: Array, hi: Array, theta: Array) -> Array:
    """Elementwise hit flags; all inputs broadcast to (D,)."""
    return (jnp.asarray(theta) >= jnp.asarray(lo)) & (jnp.asarray(theta) <= jnp.asarray(hi))


def compute_rep_metrics(
    posterior_samples: Array,
    theta_target: Array,
    *,
    level: float = 0.95,
    want_central: bool = True,
    want_hpdi: bool = True,
    # Optional joint HPD check (log‑density threshold)
    logpdf_samples: Array | None = None,  # shape (K,)
    logpdf_at_theta: float | None = None,
) -> dict[str, Any]:
    """
    Returns a JSON‑friendly dict with per‑dim intervals, widths, and hit flags.
    Keys (when available): ci_central, width_central, hit_central, ci_hpdi, width_hpdi, hit_hpdi,
    n_post, level, joint_hpd_hit.
    """
    alpha = 1.0 - level
    s = _ensure_2d(posterior_samples)
    out: dict[str, Any] = {
        "n_post": int(_ensure_2d(posterior_samples).shape[0]),
        "level": float(level),
    }
    th = jnp.asarray(theta_target)

    mu, sd = posterior_mean_sd(s)
    bias = jnp.abs(mu - th)
    var = jnp.square(sd)
    se_mean = jnp.square(bias)  # squared error of posterior mean
    post_mse = var + jnp.square(bias)  # E[(θ-θ†)^2 | posterior]

    out["post_mean"] = mu.tolist()
    out["post_sd"] = sd.tolist()
    out["bias"] = bias.tolist()
    out["se_mean"] = se_mean.tolist()
    out["post_mse"] = post_mse.tolist()

    if want_central:
        lo, hi = central_ci(posterior_samples, alpha)
        out["ci_central"] = {"lo": lo.tolist(), "hi": hi.tolist()}
        out["width_central"] = (hi - lo).tolist()
        out["hit_central"] = covered(lo, hi, th).tolist()

    if want_hpdi:
        lo, hi = hpdi(posterior_samples, alpha)
        out["ci_hpdi"] = {"lo": lo.tolist(), "hi": hi.tolist()}
        out["width_hpdi"] = (hi - lo).tolist()
        out["hit_hpdi"] = covered(lo, hi, th).tolist()

    if (logpdf_samples is not None) and (logpdf_at_theta is not None):
        # e.g., 5th pct for 95% set
        t = _q(jnp.asarray(logpdf_samples), alpha, axis=0, method="linear")
        out["joint_hpd_hit"] = bool(jnp.asarray(logpdf_at_theta) >= t)

    return out


def posterior_predictive_distance_on_summaries(
    *,
    key: Array,
    theta_samples: Array,  # (K, Dθ)
    simulate: Callable[[Array, Array, int], Array],
    summaries: Callable[[Array], Array],
    s_obs: Array,
    n_obs: int,
    n_rep: int = 1000,
    metric: Literal["l2", "l1"] = "l2",
) -> dict[str, Any]:
    """Sample y~p(y|θ), θ~posterior; distance between s(y) and s_obs."""
    K = theta_samples.shape[0]
    key_idx, key_sim = jax.random.split(key)
    idx = jax.random.randint(key_idx, (n_rep,), minval=0, maxval=K)
    th = theta_samples[idx]
    keys = jax.random.split(key_sim, n_rep)

    def _one(kk: Array, t: Array) -> Array:
        y = simulate(kk, t, n_obs)
        s = summaries(y)
        d = s - s_obs
        return jnp.linalg.norm(d, ord=2) if metric == "l2" else jnp.sum(jnp.abs(d))

    dists = jax.vmap(_one)(keys, th)
    finite = jnp.isfinite(dists)
    dists_f = dists[finite]
    return {
        "ppd_mean": float(jnp.mean(dists_f)),
        "ppd_sd": float(jnp.std(dists_f, ddof=1)),
        "ppd_q50": float(jnp.quantile(dists_f, 0.5)),
        "ppd_q90": float(jnp.quantile(dists_f, 0.9)),
    }


if __name__ == "__main__":
    key = jax.random.key(0)
    K, D = 20_000, 4
    th_true = jnp.array([0.0, 1.0, -1.0, 0.5])
    cov = jnp.array([1.0, 0.5, 2.0, 0.2])  # stds
    x = th_true + cov * jax.random.normal(key, (K, D))
    m = compute_rep_metrics(x, th_true, level=0.95, want_central=True, want_hpdi=True)
    print("central hit:", m["hit_central"])
    print("hpdi hit   :", m["hit_hpdi"])
    print("central width (mean):", sum(m["width_central"]) / D)
    print("hpdi    width (mean):", sum(m["width_hpdi"]) / D)

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

type Array = jax.Array


# model params = (sigma_rw, nu)
def prior_sample(key: Array) -> jnp.ndarray:
    """sigma_rw ~ Exp(50),  nu ~ Exp(0.1). Returns θ shape (2,)."""
    k1, k2 = jax.random.split(key)
    sigma_rw = jax.random.exponential(k1) / 50.0  # mean 0.02
    nu = jax.random.exponential(k2) / 0.1  # mean 10.0
    return jnp.array([sigma_rw, nu])


# def prior_sample(key: Array) -> jnp.ndarray:
#     """sigma_rw ~ Exp(50),  nu ~ Exp(0.1). Returns θ shape (2,)."""
#     k1, k2 = jax.random.split(key)
#     # sigma_rw = jax.random.gamma(k1, 5)
#     # nu = jax.random.gamma(k2, 5)  # mean 10.0
#     sigma_rw = dist.Gamma(5, 25).sample(k1)
#     nu = dist.Gamma(5, 1).sample(k2)
#     return jnp.array([sigma_rw, nu])


def assumed_dgp(key: Array, theta: jnp.ndarray, T: int = 100) -> Any:
    """
    Stochastic volatility:
      s_0 = 0,  s_t = s_{t-1} + eps_t,  eps_t ~ N(0, sigma_rw^2)
      r_t ~ StudentT(df=nu, loc=0, scale=exp(s_t))
    Returns r with shape (T,).
    """
    dtype = theta.dtype
    sigma_rw = jnp.asarray(theta[0], dtype=dtype)
    nu_raw = jnp.asarray(theta[1], dtype=dtype)

    # Clamp nu for numerical stability while preserving heavy tails.
    # NOTE: check below, maybe we want numerical instability to motivate robust methods? No clipping
    # NOTE2: different results if do or do not (wider posteriors) - reason if do or do not
    # nu = jnp.clip(nu_raw, a_min=jnp.asarray(0.2, dtype), a_max=jnp.asarray(2e2, dtype))
    nu = nu_raw

    k_rw, k_t = jax.random.split(key)
    # Log‑volatility random walk with s_0 = 0
    eps = sigma_rw * jax.random.normal(k_rw, (T - 1,), dtype=dtype)
    s = jnp.concatenate([jnp.zeros((1,), dtype=dtype), jnp.cumsum(eps)])
    # Heavy‑tailed returns
    t_noise = dist.StudentT(df=nu, loc=0.0, scale=1.0).sample(k_t, (T,)).astype(dtype)
    r = jnp.exp(s) * t_noise
    return r  # (T,)


def true_dgp(
    key: Array,
    theta: jnp.ndarray,
    T: int = 100,
    *,
    sigma_ms: int = 0,  # sigma in {0,1,2,3,4}
    block_1idx_inclusive: tuple[int, int] = (50, 65),
    **_: object,  # ignore other kwargs
) -> jnp.ndarray:
    """
    Misspecification: contiguous 'volmageddon' block scaled by 5**sigma_ms.
    S = {50,...,65} (1‑indexed). σ=0 returns identity.
    """
    x = assumed_dgp(key, theta, T=T)
    if sigma_ms <= 0:  #  i.e., when set to 0, should return identity
        return x
    s1, s2 = block_1idx_inclusive
    start = max(0, min(T, s1 - 1))
    end = max(start, min(T, s2))
    factor = 5.0**sigma_ms
    idx = jnp.arange(T)
    in_block = (idx >= start) & (idx < end)
    scale = jnp.where(in_block, factor, 1.0).astype(x.dtype)
    return x * scale


def _acf_x2(x: jnp.ndarray, lags: jnp.ndarray) -> jnp.ndarray:
    """ACF of x^2 at given lags via FFT. Shape (len(lags),)."""
    y = jnp.square(x)
    yc = y - jnp.mean(y)
    n = yc.shape[0]
    lags = jnp.asarray(lags, dtype=jnp.int32)

    if n < 2:
        return jnp.zeros((lags.shape[0],), dtype=x.dtype)

    nfft = 2 * n
    f = jnp.fft.rfft(yc, n=nfft)
    acf_full = jnp.fft.irfft(f * jnp.conj(f), n=nfft)[:n]  # unnormalised autocov
    acf_full = acf_full / (acf_full[0] + 1e-12)  # normalise so ρ(0)=1

    lags = jnp.clip(lags, 1, n - 1)  # ensure valid
    return acf_full[lags].astype(x.dtype)


# def summaries(x: jnp.ndarray, lags: tuple[int, ...] = (1, 2, 3, 4, 5)) -> jnp.ndarray:
#     # first compute full summaries
#     full_s_x = full_summaries(x, lags=lags)
#     subset_idx = [0, 1, 3, 6]
#     return full_s_x[jnp.array(subset_idx, dtype=jnp.int32)]


def summaries(x: jnp.ndarray, lags: tuple[int, ...] = (1, 2, 3, 4, 5)) -> jnp.ndarray:
    return x


def full_summaries(x: jnp.ndarray, lags: tuple[int, ...] = (2, 3, 4, 5)) -> jnp.ndarray:
    """
    # TODO: go through these and see which are most useful/simple
    """
    x = jnp.asarray(x)
    mu = jnp.mean(x)
    xc = x - mu
    var = jnp.mean(xc * xc)
    m4 = jnp.mean(xc**4)
    kurt = m4 / (var**2 + 1e-24)
    sd = jnp.sqrt(var + 1e-12)
    max_abs = jnp.max(jnp.abs(x))
    exceed_6sd = jnp.sum(jnp.abs(x) > 6.0 * sd).astype(x.dtype)
    acf = _acf_x2(x, jnp.asarray(lags, dtype=jnp.int32))
    return jnp.concatenate([jnp.array([var, m4, kurt, max_abs, exceed_6sd], dtype=x.dtype), acf])

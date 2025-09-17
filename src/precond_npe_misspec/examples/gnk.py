"""Implementation of the univariate g-and-k model."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.typing import ArrayLike

type Array = jax.Array
type PRNGKey = Array  # PRNG key array (uint32[2])


def prior_logpdf(
    theta: ArrayLike,
    A_min: float = 0.0,
    A_max: float = 10.0,
    B_min: float = 0.0,
    B_max: float = 10.0,
    g_min: float = 0.0,
    g_max: float = 10.0,
    k_min: float = 0.0,
    k_max: float = 10.0,
) -> Array:
    """Log p(θ) for independent Uniforms on (A,B,g,k).
    Vectorised over leading dims. Returns 0 on support, −inf outside.
    """
    th = jnp.asarray(theta)
    if th.shape[-1] != 4:
        raise ValueError("theta must have last dimension 4: (A,B,g,k).")
    A, B, g, k = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    in_box = (
        (A_min <= A)
        & (A <= A_max)
        & (B_min <= B)
        & (B <= B_max)
        & (g_min <= g)
        & (g <= g_max)
        & (k_min <= k)
        & (k <= k_max)
    )
    zero = jnp.zeros_like(A, dtype=th.dtype)
    neg_inf = jnp.full_like(A, -jnp.inf, dtype=th.dtype)
    return jnp.where(in_box, zero, neg_inf)


def gnk(
    z: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    g: ArrayLike,
    k: ArrayLike,
    c: float = 0.8,
) -> Array:
    """Quantile function for the g-and-k distribution."""
    z_ = jnp.asarray(z)
    A_ = jnp.asarray(A)
    B_ = jnp.asarray(B)
    g_ = jnp.asarray(g)
    k_ = jnp.asarray(k)
    return A_ + B_ * (1 + c * jnp.tanh(g_ * z_ / 2)) * (1 + z_**2) ** k_ * z_


def ss_octile(y: ArrayLike) -> Array:
    """Calculate octiles of the input data."""
    octiles = jnp.linspace(12.5, 87.5, 7)
    return jnp.percentile(y, octiles, axis=-1)


# def ss_duodecile(y):
#     """Calculate octiles of the input data."""
#     duodeciles = jnp.linspace(8.33, 91.67, 11)
#     return jnp.percentile(y, duodeciles, axis=-1)


# def ss_hexadeciles(y):
#     """Calculate hexadeciles of the input data."""
#     hexadeciles = jnp.linspace(6.25, 93.75, 15)
#     return jnp.percentile(y, hexadeciles, axis=-1)


# def gnk_density(x, A, B, g, k, c=0.8):
#     """Calculate the density of the g-and-k distribution."""
#     z = pgk(x, A, B, g, k, c, zscale=True)
#     return norm.pdf(z) / gnk_deriv(z, A, B, g, k, c)


# def gnk_density_from_z(z, A, B, g, k, c=0.8):
#     """Calculate the density of the g-and-k distribution given z."""
#     dQdz = gnk_deriv(z, A, B, g, k, c)
#     f_x = (norm.pdf(z) ** 2) / dQdz
#     return f_x


def true_dgp(
    key: PRNGKey,
    n_obs: int,
    # w: float = 0.1,
    # mu1: float = 3.0,
    # var1: float = 2.0,
    # mu2: float = 20.0,
    # var2: float = 16.0,
    w: float = 0.6,
    mu1: float = 1.0,
    var1: float = 2.0,
    mu2: float = 7.0,
    var2: float = 2.0,
) -> Array:
    """Generate n_obs iid draws from the misspecified mixture."""
    k_z, k1, k2 = random.split(key, 3)
    z = random.bernoulli(k_z, p=w, shape=(n_obs,))  # 1 -> comp 1, 0 -> comp 2
    y1 = mu1 + jnp.sqrt(var1) * random.normal(k1, (n_obs,))  # N(μ1, σ1^2)
    y2 = mu2 + jnp.sqrt(var2) * random.normal(k2, (n_obs,))  # N(μ2, σ2^2)
    return z * y1 + (1.0 - z) * y2


def get_summaries_batches(
    key: PRNGKey,
    A: ArrayLike,
    B: ArrayLike,
    g: ArrayLike,
    k: ArrayLike,
    n_obs: int,
    n_sims: int,
    batch_size: int,
    sum_fn: Callable[[Array], Array] = ss_octile,
) -> Array:
    """Generate batches of summary statistics for the g-and-k model."""
    num_batches = (n_sims + batch_size - 1) // batch_size
    all_octiles: list[Array] = []
    A_ = jnp.asarray(A)
    B_ = jnp.asarray(B)
    g_ = jnp.asarray(g)
    k_ = jnp.asarray(k)

    for i in range(num_batches):
        sub_key, key = random.split(key)
        start: int = i * batch_size
        stop: int = min(start + batch_size, n_sims)
        bs: int = stop - start

        z_batch = random.normal(sub_key, shape=(n_obs, bs))
        # Ensure parameters are indexable arrays
        x_batch = gnk(z_batch, A_[start:stop], B_[start:stop], g_[start:stop], k_[start:stop]).T  # (bs, n_obs)

        octiles_batch = sum_fn(x_batch).T  # (7, bs)
        all_octiles.append(octiles_batch)

    return jnp.concatenate(all_octiles, axis=1)


def ss_robust(y: ArrayLike) -> Array:
    """Compute robust summary statistics as in Drovandi 2011 #TODO."""
    y = jnp.asarray(y)
    # Ensure 2D: (batch, n_obs)
    y2 = y[None, :] if y.ndim == 1 else y
    if y2.ndim != 2:
        raise ValueError("y must be 1D or 2D of shape (batch, n_obs).")

    ss_A = _get_ss_A(y2)  # (batch, 1)
    ss_B = _get_ss_B(y2)  # (batch, 1)
    ss_g = _get_ss_g(y2)  # (batch, 1)
    ss_k = _get_ss_k(y2)  # (batch, 1)
    S = jnp.concatenate([ss_A, ss_B, ss_g, ss_k], axis=1)  # (batch, 4)
    return jnp.squeeze(S)  # -> (4,) if batch==1 else (batch, 4)


def _get_ss_A(y: Array) -> Array:
    """Compute the median as a summary statistic."""
    L2 = jnp.percentile(y, 50.0, axis=-1, keepdims=True)  # (batch, 1)
    return L2


def _get_ss_B(y: Array) -> Array:
    """Compute the interquartile range."""
    q = jnp.percentile(y, jnp.array([25.0, 75.0]), axis=-1, keepdims=True)  # (2,batch,1)
    L1, L3 = q[0], q[1]  # each (batch,1)
    iqr = L3 - L1
    eps: Array = jnp.asarray(np.finfo(np.dtype(y.dtype)).eps, dtype=y.dtype)
    iqr = jnp.where(iqr == 0, eps, iqr)
    return iqr


def _get_ss_g(y: Array) -> Array:
    """Compute a skewness-like summary statistic."""
    q = jnp.percentile(y, jnp.array([25.0, 50.0, 75.0]), axis=-1, keepdims=True)  # (3,batch,1)
    L1, L2, L3 = q[0], q[1], q[2]  # each (batch,1)
    iqr = _get_ss_B(y)  # (batch,1)
    sg = (L3 + L1 - 2.0 * L2) / iqr
    return sg


def _get_ss_k(y: Array) -> Array:
    """Compute a kurtosis-like summary statistic."""
    q = jnp.percentile(y, jnp.array([12.5, 37.5, 62.5, 87.5]), axis=-1, keepdims=True)  # (4,batch,1)
    E1, E3, E5, E7 = q[0], q[1], q[2], q[3]
    iqr = _get_ss_B(y)  # (batch,1)
    sk = (E7 - E5 + E3 - E1) / iqr
    return sk


def simulate(key: Array, theta: jnp.ndarray, *, n_obs: int) -> jnp.ndarray:
    z = jax.random.normal(key, (n_obs,), dtype=theta.dtype)
    A, B, g, k = theta
    return gnk(z, A, B, g, k)


def summaries_for_metrics(x: jnp.ndarray) -> jnp.ndarray:
    return ss_robust(x)

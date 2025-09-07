"""Implementation of the univariate g-and-k model."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike

type Array = jax.Array
type PRNGKey = Array  # PRNG key array (uint32[2])


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
    w: float = 0.9,
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


# def ss_robust(y):
#     """Compute robust summary statistics as in Drovandi 2011 #TODO."""
#     ss_A = _get_ss_A(y)
#     ss_B = _get_ss_B(y)
#     ss_g = _get_ss_g(y)
#     ss_k = _get_ss_k(y)

#     # Combine the summary statistics, (batch should be first dim)
#     ss_robust = jnp.concatenate(
#         [ss_A[:, None], ss_B[:, None], ss_g[:, None], ss_k[:, None]], axis=1
#     )
#     return jnp.squeeze(ss_robust)


# def _get_ss_A(y):
#     """Compute the median as a summary statistic."""
#     L2 = jnp.percentile(y, 50, axis=1)
#     ss_A = L2
#     return ss_A[:, None]


# def _get_ss_B(y):
#     """Compute the interquartile range."""
#     L1, L3 = jnp.percentile(y, jnp.array([25, 75]), axis=1)
#     ss_B = L3 - L1
#     ss_B = jnp.where(ss_B == 0, jnp.finfo(jnp.float32).eps, ss_B)
#     return ss_B[:, None]


# def _get_ss_g(y):
#     """Compute a skewness-like summary statistic."""
#     L1, L2, L3 = jnp.percentile(y, jnp.array([25, 50, 75]), axis=1)
#     ss_B = _get_ss_B(y).flatten()
#     ss_g = (L3 + L1 - 2 * L2) / ss_B
#     return ss_g[:, None]


# def _get_ss_k(y):
#     """Compute a kurtosis-like summary statistic."""
#     E1, E3, E5, E7 = jnp.percentile(y,
#                                     jnp.array([12.5, 37.5, 62.5, 87.5]),
#                                     axis=1)
#     ss_B = _get_ss_B(y).flatten()
#     ss_k = (E7 - E5 + E3 - E1) / ss_B
#     return ss_k[:, None]

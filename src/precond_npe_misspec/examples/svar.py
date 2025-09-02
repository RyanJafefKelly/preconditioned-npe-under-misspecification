from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def default_pairs(k: int = 6) -> jnp.ndarray:
    """Three disjoint symmetric couplings for k=6 by default."""
    assert k == 6, "default_pairs defined for k=6"
    return jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)


def prior_sample(key: Array, pairs: Array | None = None) -> jnp.ndarray:
    """θ ~ Uniform([-1,1]^{2m}) × Uniform([0,1]) for σ."""
    pairs = default_pairs() if pairs is None else pairs
    m = pairs.shape[0]
    k1, k2 = jax.random.split(key)
    off = jax.random.uniform(k1, (2 * m,), minval=-1.0, maxval=1.0)
    sigma = jax.random.uniform(k2, (), minval=0.0, maxval=1.0)
    return jnp.concatenate([off, jnp.array([sigma])])


def _theta_to_X(
    theta: jnp.ndarray, k: int, pairs: Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map θ -> (X, σ). σ is a JAX scalar."""
    m = pairs.shape[0]
    dtype = theta.dtype
    X = (-0.1) * jnp.eye(k, dtype=dtype)

    fwd = theta[0::2][:m]
    rev = theta[1::2][:m]
    i, j = pairs[:, 0], pairs[:, 1]
    X = X.at[i, j].set(fwd)
    X = X.at[j, i].set(rev)

    sigma = jnp.asarray(theta[-1], dtype=dtype)  # JAX scalar, no Python float
    return X, sigma


def true_dgp(
    key: Array,
    theta: jnp.ndarray,
    k: int = 6,
    T: int = 1000,
    pairs: Array | None = None,
) -> jnp.ndarray:
    """SVAR: y_t = X y_{t-1} + ε_t, ε_t ~ N(0, σ^2 I_k). Returns Y shape (T,k)."""
    pairs = default_pairs() if pairs is None else pairs
    X, sigma = _theta_to_X(theta, k, pairs)  # sigma is JAX scalar
    k0, k_draws = jax.random.split(key)
    y0 = sigma * jax.random.normal(k0, (k,), dtype=theta.dtype)

    def step(y, kk):  # type: ignore
        eps = sigma * jax.random.normal(kk, (k,), dtype=theta.dtype)
        y_new = X @ y + eps
        return y_new, y_new

    keys = jax.random.split(k_draws, T - 1)
    _, ys = jax.lax.scan(step, y0, keys)
    return jnp.vstack([y0[None, :], ys])


assumed_dgp = (
    true_dgp  # same model in this example  # TODO! CHANGE FOR MISSPECIFICATION
)


def summaries(Y: jnp.ndarray, pairs: Array | None = None) -> jnp.ndarray:
    """Lag-1 cross autocov for each directed pair, plus pooled std over all entries."""
    pairs = default_pairs() if pairs is None else pairs
    T = Y.shape[0]
    A = Y[1:, pairs[:, 0]] * Y[:-1, pairs[:, 1]]  # shape (T-1, m)
    B = Y[1:, pairs[:, 1]] * Y[:-1, pairs[:, 0]]  # shape (T-1, m)
    sdir = jnp.concatenate([A.sum(0), B.sum(0)]) / T  # use 1/T as in the paper
    s_sigma = jnp.std(Y.reshape(-1), ddof=0)
    return jnp.concatenate([sdir, jnp.array([s_sigma])])

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

# mypy: disable-error-code=no-untyped-def


def default_pairs(k: int = 6) -> jnp.ndarray:
    """Three disjoint symmetric couplings for k=6 by default."""
    assert k == 6, "default_pairs defined for k=6"
    return jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)


def prior_sample(key: Array, pairs: Array | None = None) -> jnp.ndarray:
    """theta ~ Uniform([-1,1]^6) × Uniform([0,1]) for sigma."""
    pairs = default_pairs() if pairs is None else pairs
    m = pairs.shape[0]
    k1, k2 = jax.random.split(key)
    off = jax.random.uniform(k1, (2 * m,), minval=-1.0, maxval=1.0)
    sigma = jax.random.uniform(k2, (), minval=0.0, maxval=1.0)
    return jnp.concatenate([off, jnp.array([sigma])])


def prior_logpdf(theta: jnp.ndarray, pairs: Array | None = None) -> jnp.ndarray:
    """Log p(θ) for Uniform([-1,1]^{2m}) × Uniform([0,1]) on σ."""
    pairs = default_pairs() if pairs is None else pairs
    m = pairs.shape[0]
    off = theta[: 2 * m]
    sigma = theta[-1]
    in_box = jnp.all((off >= -1.0) & (off <= 1.0)) & (0.0 <= sigma) & (sigma <= 1.0)
    return jnp.where(in_box, 0.0, -jnp.inf)


def _theta_to_X(
    theta: jnp.ndarray, k: int, pairs: Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map theta -> (X, sigma). sigma is a JAX scalar."""
    m = pairs.shape[0]
    dtype = theta.dtype
    X = (-0.1) * jnp.eye(k, dtype=dtype)

    fwd = theta[0::2][:m]
    rev = theta[1::2][:m]
    i, j = pairs[:, 0], pairs[:, 1]
    X = X.at[i, j].set(fwd)
    X = X.at[j, i].set(rev)

    sigma = jnp.asarray(theta[-1], dtype=dtype)
    return X, sigma


def assumed_dgp(
    key: Array,
    theta: jnp.ndarray,
    k: int = 6,
    T: int = 1000,
    pairs: Array | None = None,
) -> jnp.ndarray:
    """SVAR: y_t = X y_{t-1} + ε_t, ε_t ~ N(0, σ^2 I_k). Returns Y shape (T,k)."""
    pairs = default_pairs() if pairs is None else pairs
    X, sigma = _theta_to_X(theta, k, pairs)
    k0, k_draws = jax.random.split(key)
    y0 = sigma * jax.random.normal(k0, (k,), dtype=theta.dtype)

    def step(y, kk):
        eps = sigma * jax.random.normal(kk, (k,), dtype=theta.dtype)
        y_new = X @ y + eps
        return y_new, y_new

    keys = jax.random.split(k_draws, T - 1)
    _, ys = jax.lax.scan(step, y0, keys)
    return jnp.vstack([y0[None, :], ys])


def true_dgp(
    key: Array,
    theta: jnp.ndarray,
    k: int = 6,
    T: int = 1000,
    pairs: Array | None = None,
    mu: float = 0.05,
    eps: float = 0.02,  # fraction of contaminated time points
    kappa: float = 12.0,  # outlier scale multiplier
    df: float = 1.0,  # heavy tail (df=1 ⇒ Cauchy-like)
    per_channel: bool = True,  # contam each channel independently at a contaminated t
) -> jnp.ndarray:
    """Clean AR(1) then add mean-zero heavy-tailed spikes to a small fraction of rows."""
    Y = assumed_dgp(key, theta, k=k, T=T, pairs=pairs)  # clean SVAR
    # sigma = jnp.asarray(theta[-1], dtype=Y.dtype)

    # k_mask, k_eps = jax.random.split(jax.random.fold_in(key, 2024))
    # contam = jax.random.bernoulli(k_mask, eps, (T,))  # (T,)
    # e_t = kappa * sigma * _student_t_noise(k_eps, df, (T, 1))  # shared across channels

    # misspecify mean. # NOTE: more interesting misspecification?
    return Y + mu


def summaries(Y: jnp.ndarray, pairs: Array | None = None) -> jnp.ndarray:
    """Lag-1 cross autocov for each directed pair, plus pooled std over all entries."""
    pairs = default_pairs() if pairs is None else pairs
    T = Y.shape[0]
    Y_fwd = Y[1:, :]  # t = 2..T
    Y_lag = Y[:-1, :]  # t = 1..T-1
    Z_fwd = Y_fwd - jnp.mean(Y_fwd, axis=0, keepdims=True)
    Z_lag = Y_lag - jnp.mean(Y_lag, axis=0, keepdims=True)
    # centred lag-1 cross-covariances for directed pairs
    A = (Z_fwd[:, pairs[:, 0]] * Z_lag[:, pairs[:, 1]]).sum(axis=0) / (T - 1)
    B = (Z_fwd[:, pairs[:, 1]] * Z_lag[:, pairs[:, 0]]).sum(axis=0) / (T - 1)
    sdir = jnp.concatenate([A, B])
    s_sigma = jnp.std(Y.reshape(-1), ddof=0)
    Y_mean = jnp.mean(Y)
    return jnp.concatenate([sdir, jnp.array([s_sigma]), jnp.array([Y_mean])])


def simulate(
    key,
    theta,
    *,
    k: int,
    T: int,
    obs_model: str = "assumed",
    mu: float = 0.05,
    eps: float = 0.02,
    kappa: float = 12.0,
    df: float = 1.0,
    per_channel: bool = True,
):
    pairs = default_pairs(k)
    if obs_model == "assumed":
        return assumed_dgp(key, theta, k=k, T=T, pairs=pairs)
    return true_dgp(
        key,
        theta,
        k=k,
        T=T,
        pairs=pairs,
        mu=mu,
        eps=eps,
        kappa=kappa,
        df=df,
        per_channel=per_channel,
    )


def summaries_for_metrics(x, *, k: int):
    pairs = default_pairs(k)
    return summaries(x, pairs=pairs)

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

# mypy: disable-error-code=no-untyped-def

"""
contaminated_weibull.py

Simulation components for SBI under misspecification.
Assumed DGP: Weibull(scale=1.0, shape=k).
Prior: k ~ LogNormal(mu, sigma) with mu=-1.1, sigma=0.9 (≈28.6% mass below k<0.2).
True DGP: same clean Weibull draws, but an eps-fraction are replaced by
          negative outliers ~ Normal(mu_c=-1, sigma_c=0.2).

Exports:
- prior_sample(key, mu=-1.1, sigma=0.9) -> theta (shape (1,))
- prior_logpdf(theta, mu=-1.1, sigma=0.9) -> log p(theta)
- assumed_dgp(key, theta, n: int = 200) -> y (shape (n,))
- true_dgp(key, theta, n: int = 200, eps=0.05, mu_c=-1.0, sigma_c=0.2) -> y
- summaries(y) -> jnp.array([sample_variance, sample_mean, sample_min])
- simulate(..., obs_model={"assumed","true"}) convenience wrapper

Notes:
- theta is a length-1 vector with theta[0] = k > 0.
- All functions are JAX-pure and vectorised.
"""


# ------------------------------
# Helpers
# ------------------------------
def _theta_to_k(theta: Array) -> Array:
    k = jnp.asarray(theta[0])
    return k


def _weibull_sample(key: Array, k_shape: Array, n: int, *, lam: float = 1.0) -> Array:
    """Sample i.i.d. Weibull(shape=k, scale=lam) via inverse-CDF.
    If E ~ Exp(1), then X = lam * E**(1/k).
    """
    e = jax.random.exponential(key, shape=(n,), dtype=k_shape.dtype)
    x = lam * jnp.power(e, 1.0 / k_shape)
    return x


# ------------------------------
# Prior: LogNormal on k
# ------------------------------
def prior_sample(key: Array, mu: float = 0.0, sigma: float = 0.9) -> Array:
    """Return theta = [k], where k ~ LogNormal(mu, sigma) on log-scale."""
    z = jax.random.normal(key, (1,))
    k = jnp.exp(mu + sigma * z)[0]
    return jnp.array([k])


def prior_logpdf(theta: Array, mu: float = -1.1, sigma: float = 0.9) -> Array:
    """Log-density of LogNormal(mu, sigma) at k=theta[0]."""
    k = _theta_to_k(theta)
    valid = k > 0.0
    log_k = jnp.log(k)
    logpdf = (
        -jnp.log(k * sigma * jnp.sqrt(2.0 * jnp.pi)) - 0.5 * ((log_k - mu) / sigma) ** 2
    )
    return jnp.where(valid, logpdf, -jnp.inf)


# ------------------------------
# Assumed and True DGPs
# ------------------------------
def assumed_dgp(key: Array, theta: Array, n_obs: int = 200, lam: float = 1.0) -> Array:
    """y_i ~ Weibull(shape=k, scale=lam). Returns shape (n,)."""
    k_shape = _theta_to_k(theta)
    return _weibull_sample(key, k_shape, n_obs, lam=lam)


def true_dgp(
    key: Array,
    theta: Array,
    n_obs: int = 200,
    *,
    lam: float = 1.0,
    eps: float = 0.05,
    mu_c: float = -1.0,
    sigma_c: float = 0.2,
) -> Array:
    """Mixture contamination: with prob 1-eps use Weibull, else Normal(mu_c, sigma_c^2).
    Returns shape (n,).
    """
    k1, k2, k3 = jax.random.split(key, 3)
    clean = _weibull_sample(k1, _theta_to_k(theta), n_obs, lam=lam)
    contam_mask = jax.random.bernoulli(k2, eps, (n_obs,))
    outliers = mu_c + sigma_c * jax.random.normal(k3, (n_obs,), dtype=clean.dtype)
    return jnp.where(contam_mask, outliers, clean)


# ------------------------------
# Summaries
# ------------------------------
def summaries(y: Array) -> Array:
    """Return [sample_variance, sample_mean, sample_min].
    Variance uses ddof=1 (unbiased) when n>1. For n<=1 falls back to ddof=0.
    """
    # y = jnp.log(y)
    n = y.shape[0]
    ddof = jnp.where(n > 1, 1, 0)
    var = jnp.var(y, ddof=ddof)
    mean = jnp.mean(y)
    ymin = jnp.min(y)
    return jnp.stack([var, mean, ymin])


def ss_log(x: Array) -> Array:
    """Alias for summaries, for compatibility."""
    return jnp.log(summaries(x))


# ------------------------------
# Convenience
# ------------------------------
def simulate(
    key: Array,
    theta: Array,
    n_obs: int = 200,
    *,
    obs_model: str = "assumed",
    lam: float = 1.0,
    eps: float = 0.05,
    mu_c: float = -1.0,
    sigma_c: float = 0.2,
) -> Array:
    """Dispatch to assumed or true DGPs."""
    if obs_model == "assumed":
        return assumed_dgp(key, theta, n_obs=n_obs, lam=lam)
    elif obs_model == "true":
        return true_dgp(
            key, theta, n_obs=n_obs, lam=lam, eps=eps, mu_c=mu_c, sigma_c=sigma_c
        )
    else:
        raise ValueError("obs_model must be 'assumed' or 'true'.")


# ------------------------------
# Minimal sanity check
# ------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    th = prior_sample(jax.random.fold_in(key, 1))
    y_assumed = assumed_dgp(jax.random.fold_in(key, 2), th, n_obs=200)
    y_true = true_dgp(
        jax.random.fold_in(key, 3), th, n_obs=200, eps=0.05, mu_c=-1.0, sigma_c=0.2
    )

    s_assumed = summaries(y_assumed)
    s_true = summaries(y_true)

    print("theta[k] =", float(th[0]))
    print("assumed summaries [var, mean, min] =", [float(x) for x in s_assumed])
    print("true    summaries [var, mean, min] =", [float(x) for x in s_true])
    # Expect: true mean < assumed mean, true min < 0, variance slightly larger.

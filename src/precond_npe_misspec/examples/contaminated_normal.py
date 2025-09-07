from __future__ import annotations

import jax
import jax.numpy as jnp

type Array = jax.Array  # PRNG keys are also Arrays in JAX>=0.4


def true_dgp(key: Array, theta: float, stdev_err: float = 2.0, n_obs: int = 100) -> jnp.ndarray:
    """Mixture: 80% N(theta,1^2) + 20% N(theta,stdev_err^2)."""
    w = 0.8
    k1, k2 = jax.random.split(key)
    stds = jax.random.choice(k1, jnp.array([1.0, stdev_err]), shape=(n_obs,), p=jnp.array([w, 1 - w]))
    eps = jax.random.normal(k2, (n_obs,))
    return theta + stds * eps


def assumed_dgp(key: Array, theta: float, n_obs: int = 100) -> jnp.ndarray:
    return theta + jax.random.normal(key, (n_obs,))


def summaries(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.mean(x), jnp.var(x, ddof=1)])


def prior_sample(key: Array, mu0: float = 0.0, sigma0: float = 10.0) -> jnp.ndarray:
    # return a JAX scalar (shape=()) suitable for vmap/jit
    return jnp.asarray(mu0 + sigma0 * jax.random.normal(key, ()))


def conjugate_posterior_under_assumed(
    x: jnp.ndarray, mu0: float = 0.0, sigma0: float = 10.0, sigma: float = 1.0
) -> tuple[float, float]:
    """Posterior for theta under *assumed* N(theta, sigma^2) likelihood."""
    n = x.shape[0]
    xbar = jnp.mean(x)
    prec0, prec = 1.0 / (sigma0**2), n / (sigma**2)
    var = 1.0 / (prec0 + prec)
    mean = var * (prec0 * mu0 + prec * xbar)
    return float(mean), float(jnp.sqrt(var))

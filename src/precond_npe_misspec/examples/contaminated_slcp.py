# src/precond_npe_misspec/examples/contaminated_slcp.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

type Array = jax.Array


def _params_to_mean_and_cov(theta: Array) -> tuple[Array, Array]:
    t1, t2, t3, t4, t5 = theta
    mu = jnp.array([t1, t2])
    sigma1 = t3**2
    sigma2 = t4**2
    rho = jnp.tanh(t5)
    Sigma = jnp.array([[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]])

    # L = jnp.array(
    #     [
    #         [sigma1, 0.0],
    #         [rho * sigma2, sigma2 * jnp.sqrt(1.0 - rho**2)],
    #     ]
    # )
    return mu, Sigma


def base_dgp(key: Array, theta: Array, num_draws: int = 5) -> Array:
    """Clean SLCP: draw num_draws i.i.d. samples from N(μ(θ), Σ(θ)) and flatten."""
    mu, Sigma = _params_to_mean_and_cov(jnp.asarray(theta))
    # z = jax.random.normal(key, (num_draws, 2))
    # y = z @ L.T + mu  # (num_draws, 2)
    y = dist.MultivariateNormal(mu, Sigma).sample(key=key, sample_shape=(num_draws,))
    return jnp.asarray(y.reshape(-1))  # shape = (2 * num_draws,)


def assumed_dgp(key: Array, theta: Array, num_draws: int = 5) -> Array:
    """Assumed DGP = clean SLCP."""
    return base_dgp(key, theta, num_draws=num_draws)


def misspecification_transform(key: Array, x: Array, misspec_level: float = 1.0, noise_scale: float = 100.0) -> Array:
    """Contaminate only the last bivariate draw with large additive noise."""
    assert x.ndim == 1 and x.size % 2 == 0
    d = x.size
    mask = jnp.concatenate([jnp.zeros(d - 2), jnp.ones(2)])  # last two entries only
    noise = jax.random.normal(key, x.shape)
    return x + misspec_level * noise_scale * mask * noise


def true_dgp(key: Array, theta: Array, num_draws: int = 5, misspec_level: float = 1.0) -> Array:
    """Generate 4 clean draws and 1 contaminated draw (last bivariate sample)."""
    k_clean, k_noise = jax.random.split(key)
    x = base_dgp(k_clean, theta, num_draws=num_draws)
    return misspecification_transform(
        k_noise,
        x,
        misspec_level=misspec_level,
        noise_scale=100.0,
    )


def summaries(x: Array) -> Array:
    """Identity summaries (standard for SLCP)."""
    return x


def prior_sample(key: Array, low: float = -3.0, high: float = 3.0) -> Array:
    """Factorised uniform prior on θ∈[-3,3]^5."""
    return jax.random.uniform(key, (5,), minval=low, maxval=high)

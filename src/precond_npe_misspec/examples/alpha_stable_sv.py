# src/precond_npe_misspec/examples/alpha_stable_sv.py
from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.special import gammaln

EPS = 1e-8


# -----------------------------
# Prior over structural params θ = (θ2, θ3, θ4)  (persistence, vol-shock scale, tail index)
# Paper uses: θ2 ~ U(0.7,1), θ3 ~ U(0.01,1), θ4 ~ U(1,2)
# -----------------------------

# TODO: WIDE PRIOR?


def prior_sample(key: Array) -> jnp.ndarray:
    lo, hi = theta_bounds_3d()
    u = jax.random.uniform(key, shape=(3,))
    return lo + u * (hi - lo)  # (θ2, θ3, θ4)


def wide_prior_sample(key: Array) -> jnp.ndarray:
    # lo, hi = theta_bounds_3d()
    lo = jnp.array([-0.0, 0.0, 1.0], dtype=jnp.float32)
    hi = jnp.array([5.0, 10.0, 5.0], dtype=jnp.float32)
    u = jax.random.uniform(key, shape=(3,))
    return lo + u * (hi - lo)  # (θ2, θ3, θ4)


def theta_bounds_3d() -> tuple[jnp.ndarray, jnp.ndarray]:
    lo = jnp.array([-1.0, 0.01, 1.0], dtype=jnp.float32)
    hi = jnp.array([1.0, 1.0, 2.0], dtype=jnp.float32)
    return lo, hi


def theta_bounds_4d() -> tuple[jnp.ndarray, jnp.ndarray]:
    lo = jnp.array([-1.0, 0.01, 0.0, -5.0], dtype=jnp.float32)  # skew in [-0.8,0.8]
    hi = jnp.array([1.0, 1.0, 5.0, 5.0], dtype=jnp.float32)
    return lo, hi


def prior_sample_4d(key: Array) -> jnp.ndarray:
    lo, hi = theta_bounds_4d()
    k2, k3, k4, k5 = jax.random.split(key, 4)
    th2 = lo[0] + (hi[0] - lo[0]) * jax.random.uniform(k2)
    th3 = lo[1] + (hi[1] - lo[1]) * jax.random.uniform(k3)
    th4 = lo[2] + (hi[2] - lo[2]) * jax.random.uniform(k4)
    th5 = lo[3] + (hi[3] - lo[3]) * jax.random.uniform(k5)
    return jnp.array([th2, th3, th4, th5], dtype=jnp.float32)


def simulate_asv_with_optional_skew(
    key: Array,
    theta: jnp.ndarray,
    *,
    T: int,
    theta1: float,
    skew_if_missing: float = 0.0,
) -> jnp.ndarray:
    theta = jnp.asarray(theta)
    if theta.shape[0] == 4:
        th = theta[:3]
        skew = float(theta[3])
    else:
        th = theta
        skew = float(skew_if_missing)
    return assumed_dgp(key, th, T=T, theta1=theta1, skew=skew)


# -----------------------------
# α-stable sampler (Chambers–Mallows–Stuck; Zolotarev S0 parameterisation)
# β in [-1,1], α in (0,2]
# -----------------------------
def _stable_rvs(key: Array, alpha: Array, beta: float, shape: tuple[int, ...]) -> Array:
    """Chambers–Mallows–Stuck sampler in S0 parameterisation.
    Returns S0 variates for all α∈(0,2], β∈[-1,1].
    """
    eps = 1e-12
    key_u, key_w = jax.random.split(key)
    U = jax.random.uniform(key_u, shape=shape, minval=-jnp.pi / 2.0 + eps, maxval=jnp.pi / 2.0 - eps)
    W = jax.random.exponential(key_w, shape=shape)

    def _alpha_eq1(_: None) -> Array:
        denom = (jnp.pi / 2.0) + beta * U
        term = ((jnp.pi / 2.0) * W * jnp.cos(U)) / jnp.clip(denom, a_min=eps)
        # log term is clipped, not shifted by +EPS
        return (2.0 / jnp.pi) * (denom * jnp.tan(U) - beta * jnp.log(jnp.clip(term, a_min=eps)))

    def _alpha_ne1(_: None) -> Array:
        zeta = beta * jnp.tan(jnp.pi * alpha / 2.0)
        B = jnp.arctan(zeta) / alpha
        S = jnp.power(1.0 + zeta**2, 1.0 / (2.0 * alpha))
        part1 = jnp.sin(alpha * (U + B)) / jnp.power(jnp.cos(U), 1.0 / alpha)
        part2 = jnp.power(jnp.cos(U - alpha * (U + B)) / W, (1.0 - alpha) / alpha)
        X1 = S * part1 * part2  # S1 variate
        return X1 - zeta  # S1→S0 shift (β tan(πα/2) = zeta)

    return cast(Array, lax.cond(jnp.isclose(alpha, 1.0), _alpha_eq1, _alpha_ne1, operand=None))


# -----------------------------
# Simulate α-stable SV:
# ln σ_t^2 = θ1 + θ2 ln σ_{t-1}^2 + θ3 ν_t,  ν_t ~ N(0,1)
# r_t = σ_t * w_t,   w_t ~ S(α=θ4, β=skew, μ=0, σ=1)
# -----------------------------
def assumed_dgp(
    key: Array,
    theta: jnp.ndarray,
    T: int = 1000,
    theta1: float = 0.0,
    skew: float = 0.0,  # β in [-1,1]
    init_logvar: float = 0.0,
) -> jnp.ndarray:
    theta = jnp.asarray(theta)
    theta2, theta3, theta4 = theta[0], theta[1], theta[2]

    k_z, k_w = jax.random.split(key, 2)
    nu = jax.random.normal(k_z, (T,))

    # log-variance AR(1)
    def step(z_prev: Array, e: Array) -> tuple[Array, Array]:
        z_t = theta1 + theta2 * z_prev + theta3 * e
        return z_t, z_t

    _, z_path = lax.scan(step, init_logvar, nu)
    sigma = jnp.exp(0.5 * z_path)  # (T,)

    # α-stable innovations
    w = _stable_rvs(k_w, alpha=theta4, beta=skew, shape=(T,))
    r = sigma * w
    return r  # shape (T,)


# Optionally allow a slightly misspecified true DGP (skew/leverage switches)
true_dgp = assumed_dgp
# def true_dgp(
#     key: Array,
#     theta: jnp.ndarray,
#     T: int = 1000,
#     theta1: float = 0.0,
#     skew: float = -0.2,  # mild skew to force incompatibility
# ) -> jnp.ndarray:
#     return assumed_dgp(key, theta, T=T, theta1=theta1, skew=skew)


# -----------------------------
# Auxiliary GARCH(1,1)-t summaries: score vector S(y; β) / T
# Model: r_t = x_t ε_t;  x_t = β1 + β2 x_{t-1}|ε_{t-1}| + β3 x_{t-1};  ε_t ~ t_ν (std.)
# Summaries are ∂L/∂β_j / T at a fixed β (typically β̂(y_obs)); pass β via a closure.
# -----------------------------
def _student_t_logpdf(eps: Array, nu: float | Array) -> Array:
    """Log-pdf of unit-variance Student-t_ν.  Var=1 => scale s^2=(nu-2)/nu."""
    nu_arr = jnp.asarray(nu)
    c = gammaln((nu_arr + 1.0) / 2.0) - gammaln(nu_arr / 2.0) - 0.5 * (jnp.log(jnp.pi) + jnp.log(nu_arr - 2.0))
    return c - 0.5 * (nu_arr + 1.0) * jnp.log1p((eps * eps) / (nu_arr - 2.0))


@jax.jit  # type: ignore
def _garch_t_avg_loglik(r: Array, beta: Array) -> Array:
    b1, b2, b3, nu = beta[0], beta[1], beta[2], beta[3]

    def step(carry: tuple[Array, Array], rt: Array) -> tuple[tuple[Array, Array], Array]:
        x_prev, eps_prev = carry
        x_t = b1 + b2 * x_prev * jnp.abs(eps_prev) + b3 * x_prev
        x_t = jnp.maximum(x_t, 1e-8)
        eps_t = rt / x_t
        ll_t = _student_t_logpdf(eps_t, nu) - jnp.log(x_t)
        return (x_t, eps_t), ll_t

    x0 = jnp.maximum(jnp.std(r), 1e-3)
    (_, _), ll_seq = lax.scan(step, (x0, jnp.asarray(0.0)), r)
    return jnp.mean(ll_seq)


def make_summaries(aux_beta: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Build a summary function s(y) = (1/T) * ∂L/∂β at fixed aux_beta (β1,β2,β3,ν).

    Typical use:
      - Compute β̂ on the observed data once (optional helper below), then
        pass aux_beta=β̂ so that s(y_obs)≈0 (score at MLE).
      - To *lock df*, pick aux_beta[3] = ν_ref (e.g., 6 or 8) and keep it fixed.
    """
    aux_beta = jnp.asarray(aux_beta, dtype=jnp.float32)

    def summaries(y: jnp.ndarray) -> jnp.ndarray:
        g = jax.grad(lambda b: _garch_t_avg_loglik(y, b))
        return g(aux_beta)

    return summaries


# Optional: crude MLE for the auxiliary β on a single dataset (for y_obs)
def fit_aux_beta_mle(
    y: jnp.ndarray,
    init_beta: jnp.ndarray | None = None,
    steps: int = 300,
    lr: float = 5e-3,
) -> jnp.ndarray:
    """
    Lightweight gradient ascent to approximate β̂(y) for the *observed* data only.
    Constraints enforced by projection: β1>0, β2≥0, β3≥0, β2+β3<1, ν∈[2.1, 50].
    """
    if init_beta is None:
        # Reasonable defaults: small offset, persistent volatility, fat tails
        init_beta = jnp.array([0.05, 0.10, 0.85, 8.0])
    beta = init_beta.astype(jnp.float32)

    def project(b: jnp.ndarray) -> jnp.ndarray:
        b1 = jnp.maximum(b[0], 1e-6)
        b2 = jnp.clip(b[1], 1e-6, 0.98)
        b3 = jnp.clip(b[2], 1e-6, 0.98 - b2)
        nu = jnp.clip(b[3], 2.1, 50.0)
        return jnp.array([b1, b2, b3, nu])

    grad_ll = jax.grad(lambda b: _garch_t_avg_loglik(y, b))
    for _ in range(steps):
        beta = project(beta + lr * grad_ll(beta))
    return beta


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    key = jax.random.key(0)
    th = prior_sample(jax.random.fold_in(key, 1))  # (θ2, θ3, θ4)
    r = assumed_dgp(jax.random.fold_in(key, 2), th, T=512)  # synthetic returns

    # Option A: use a fixed β (locks df at 8)
    beta_ref = jnp.array([0.05, 0.10, 0.85, 8.0])
    summaries = make_summaries(beta_ref)
    s = summaries(r)
    print("score summaries @ fixed β:", s)

    # Option B: fit β̂ on this y, then recompute summaries; should be near 0
    beta_hat = fit_aux_beta_mle(r, beta_ref, steps=200, lr=3e-3)
    summaries_hat = make_summaries(beta_hat)
    s_hat = summaries_hat(r)
    print("β_hat:", beta_hat)
    print("score summaries @ β̂(y):", s_hat)

# src/precond_npe_misspec/examples/contaminated_weibull.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike

type Array = jax.Array
type PRNGKey = Array

EPS = 1e-8
LOG_EXP_SKEW_CONST = jnp.asarray(
    -1.1395470994046486, dtype=jnp.float32
)  # skew(log Exp(1))


# -----------------------------
# Weibull mechanics
# -----------------------------
def _weibull_sample(key: PRNGKey, k: ArrayLike, lam: ArrayLike, n: int) -> Array:
    """y = λ * Exp(1)^{1/k}.  Vectorised over leading dims of k, λ."""
    k = jnp.asarray(k)
    lam = jnp.asarray(lam)
    e = random.exponential(key, shape=(n,) + k.shape)
    return lam * e ** (1.0 / k)


def simulate(key: PRNGKey, theta: jnp.ndarray, *, n_obs: int) -> jnp.ndarray:
    """Assumed model: iid Weibull(k, λ).  theta=(k, λ)."""
    th = jnp.asarray(theta)
    k = th[..., 0]
    lam = th[..., 1]
    return _weibull_sample(key, k, lam, n_obs)


def true_dgp(key: PRNGKey, theta: ArrayLike, *, n_obs: int, eps: float = 0.00) -> Array:
    eps = 0.1
    k = jnp.asarray(theta)[..., 0]
    lam = jnp.asarray(theta)[..., 1]
    k1, k2, k3 = random.split(key, 3)
    e = random.exponential(k1, (n_obs,) + k.shape)
    x0 = jnp.log(lam) + (1.0 / k) * jnp.log(e + EPS)
    T = random.exponential(k2, (n_obs,) + k.shape)
    misspec_scale = 5.0
    u = random.uniform(k3, (n_obs,) + k.shape)
    x = jnp.where(u < eps, x0 + misspec_scale * T, x0)  # contaminate
    return jnp.exp(x)


# -----------------------------
# Summaries
# -----------------------------
def _log_skew(x: Array) -> Array:
    xm = x - jnp.mean(x)
    s = jnp.std(xm) + EPS
    return jnp.mean(xm**3) / (s**3)


def trimmed_quants(x: ArrayLike, tau: float, ps: ArrayLike) -> Array:
    x = jnp.asarray(x)
    ps = jnp.asarray(ps, dtype=x.dtype)

    xs = jnp.sort(x)  # ascending
    n: int = xs.shape[0]  # static Python int under jit
    m = jnp.ceil(tau * jnp.asarray(n, xs.dtype)).astype(jnp.int32)  # no Python int()
    keep = jnp.maximum(n - m, 1)  # JAX int32

    ps_t = jnp.clip(ps / (1.0 - tau), 0.0, 1.0)  # renormalise into kept mass
    pos = ps_t * (keep - 1).astype(xs.dtype)  # fractional index in [0, keep-1]
    lo = jnp.floor(pos).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, keep - 1)
    w = pos - lo.astype(xs.dtype)

    v0 = xs[lo]  # JAX gather, jit-safe
    v1 = xs[hi]
    return (1.0 - w) * v0 + w * v1


# def ss_log(x: Array, tau: float = 0.0) -> Array:
#     # pick tau=0.0 when eps=0; otherwise set tau>=eps (e.g., 0.05–0.10)
#     tau = 0.05
#     q01, q50, q99 = trimmed_quants(x, tau, jnp.array([0.01, 0.5, 0.99]))
#     med = q50  # x is log-data; do NOT take log again
#     iqr = q99 - q01  # log-width; do NOT exp()
#     skew = _log_skew(x)  # keep only for metrics; ignore in ABC distance
#     return jnp.asarray([med, iqr, skew], dtype=jnp.float32)


def ss_log(x: Array) -> Array:
    tau = 0.1
    ps = jnp.array([0.10, 0.20, 0.50, 0.80, 0.90], dtype=x.dtype)
    q10, q20, q50, q80, q90 = trimmed_quants(x, tau, (1.0 - tau) * ps)
    med = q50
    idr = q90 - q10  # inter-decile range on log-scale
    ior = q80 - q20  # inter-octile range on log-scale
    skew = _log_skew(x)  # incompatible; ignore in ABC distance
    return jnp.asarray([med, idr, ior, skew], dtype=jnp.float32)


def summaries_for_metrics(y: jnp.ndarray) -> jnp.ndarray:
    """Entry used by pipelines for PPD and metrics."""
    return ss_log(y)


def prior_sample(
    key: PRNGKey,
    logk_mu: float = -0.4,
    logk_sigma: float = 1.2,
    loglam_mu: float = -0.4,
    loglam_sigma: float = 1.2,
) -> jnp.ndarray:
    """Sample θ=(k,λ) with log‑normal marginals. Returns float64 for SMC stability."""
    k1, k2 = random.split(key)
    logk = logk_mu + logk_sigma * random.normal(k1)
    loglam = loglam_mu + loglam_sigma * random.normal(k2)
    k = jnp.exp(logk)
    lam = jnp.exp(loglam)
    return jnp.asarray([k, lam])


# # Optional: distance factory that ignores the incompatible skew component
# def make_distance_ignore_skew(_: Array) -> Callable[[Array, Array], Array]:
#     """
#     Return d(S_batch, s_obs) based only on [median, IQR] of log-scale summaries.
#     Shape support: S_batch (B,3) or (B,R,3); s_obs (3,).
#     """

#     def _dist(S: Array, s_obs: Array) -> Array:
#         S2 = jnp.asarray(S)[..., :2]
#         s2 = jnp.asarray(s_obs)[..., :2]
#         d = S2 - s2
#         # reduce over non-batch axes
#         if d.ndim == 1:
#             return jnp.linalg.norm(d)
#         return jnp.linalg.norm(d.reshape(d.shape[0], -1), axis=-1)

#     return _dist


# -----------------------------
# Sanity check
# -----------------------------
if __name__ == "__main__":
    key = random.key(0)
    n = 200
    k_true, lam_true = 0.8, 2.0
    eps, alpha = 0.05, 40.0

    # observed (misspecified)
    y_obs = true_dgp(key, jnp.array([k_true, lam_true]), n_obs=n, eps=eps)
    S_obs = ss_log(y_obs)
    print("obs [med_log, iqr_log, skew_log] =", S_obs)
    print("Weibull log‑skew constant (assumed model) =", LOG_EXP_SKEW_CONST)

    # crude ABC pilot to show acceptance ignoring skew
    M = 2000
    th_keys = jax.vmap(lambda i: random.fold_in(key, i))(
        jnp.arange(M, dtype=jnp.uint32)
    )
    thetas = jax.vmap(prior_sample)(th_keys)
    x_sim = jax.vmap(lambda kk, th: simulate(kk, th, n_obs=n))(th_keys, thetas)
    S_sim = jax.vmap(ss_log)(x_sim)

    d = jnp.linalg.norm(S_sim[:, :2] - S_obs[:2], axis=1)
    tau = jnp.quantile(d, 0.05)
    acc = d <= tau
    print("ABC accept rate (q=0.05) =", float(acc.mean()))

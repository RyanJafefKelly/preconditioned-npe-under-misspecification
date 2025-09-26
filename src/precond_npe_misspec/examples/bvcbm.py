from __future__ import annotations

from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

try:
    import tumourmodel as tm
except Exception as e:  # pragma: no cover
    raise ImportError("tumourmodel not installed or failed to import") from e

_SIM_FN = None  # set in initializer


def _init_bvcbm_worker(T: int, start_volume: float, page: int) -> None:
    global _SIM_FN
    _SIM_FN = simulator_biphasic(T=T, start_volume=start_volume, page=page)


def simulate_worker(theta: np.ndarray, seed: int) -> np.ndarray:
    if _SIM_FN is None:  # guard: should not be called in main process
        raise RuntimeError("_SIM_FN not initialised in worker")
    return np.asarray(_SIM_FN(np.asarray(theta, float), int(seed)), dtype=np.float32)


# ---------------- Simulators (host) ----------------
def simulator_monophasic(
    T: int, start_volume: float = 50.0, page: int = 5
) -> Callable[[Iterable[float], int], np.ndarray]:
    """theta = (p0, psc, dmax, gage). Returns daily tumour volumes of length T."""

    def f(theta: Iterable[float], seed: int) -> np.ndarray:
        p0, psc, dmax, gage = [float(x) for x in theta]
        return tm.simulate(
            [p0, psc, int(round(dmax)), int(round(gage)), int(page)],
            T=T,
            seed=int(seed),
            start_volume=float(start_volume),
        )

    return f


def simulator_biphasic(
    T: int, start_volume: float = 50.0, page: int = 5
) -> Callable[[Iterable[float], int], np.ndarray]:
    """theta = (p0_1, psc_1, dmax_1, gage_1, p0_2, psc_2, dmax_2, gage_2, tau)."""
    if not hasattr(tm, "simulate_biphasic"):
        raise RuntimeError(
            "tumourmodel.simulate_biphasic missing; add binding in tumourmodel-py"
        )

    def f(theta: Iterable[float], seed: int) -> np.ndarray:
        p0_1, psc_1, dmax_1, gage_1, p0_2, psc_2, dmax_2, gage_2, tau = [
            float(x) for x in theta
        ]
        return tm.simulate_biphasic(
            [p0_1, psc_1, int(round(dmax_1)), int(round(gage_1)), int(page)],
            [p0_2, psc_2, int(round(dmax_2)), int(round(gage_2)), int(page)],
            int(round(tau)),
            T=T,
            seed=int(seed),
            start_volume=float(start_volume),
        )

    return f


# ---------------- JAX-friendly summaries ----------------
def summary_identity(y: jax.Array) -> jax.Array:
    return jnp.asarray(y, dtype=jnp.float32)


def summary_log(y: jax.Array, eps: float = 1e-8) -> jax.Array:
    y = jnp.asarray(y, dtype=jnp.float32)
    return jnp.log(y + jnp.asarray(eps, dtype=jnp.float32))


# ---------------- Priors ----------------
def theta_bounds_monophasic(T: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    lo = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    hi = jnp.array([1.0, 1.0, 50.0, 24.0 * float(T)], dtype=jnp.float32)
    return lo, hi


def theta_bounds_biphasic(T: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    lo = jnp.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        dtype=jnp.float32,
    )
    hi = jnp.array(
        [
            1.0,
            1.0,
            50.0,
            24.0 * float(T),
            1.0,
            1.0,
            50.0,
            24.0 * float(T),
            float(T - 1),
        ],
        dtype=jnp.float32,
    )
    return lo, hi


def prior_monophasic(T: int) -> dist.Distribution:
    lo, hi = theta_bounds_monophasic(T)
    return dist.Independent(dist.Uniform(low=lo, high=hi), 1)


def prior_biphasic(T: int) -> dist.Distribution:
    lo, hi = theta_bounds_biphasic(T)
    return dist.Independent(dist.Uniform(low=lo, high=hi), 1)


# ---------------- Adaptors for pipeline paths (non-jitted) ----------------
def simulate(
    key: jax.Array,
    theta: jax.Array,
    *,
    T: int,
    start_volume: float = 50.0,
    page: int = 5,
) -> jax.Array:
    """Monophasic adaptor (kept for completeness)."""
    fn = simulator_monophasic(T=T, start_volume=start_volume, page=page)
    seed = int(
        jax.device_get(jax.random.randint(key, (), 0, 2**31 - 1, dtype=jnp.uint32))
    )
    return jnp.asarray(fn(theta, seed), dtype=jnp.float32)


def simulate_biphasic(
    key: jax.Array,
    theta: jax.Array,
    *,
    T: int,
    start_volume: float = 50.0,
    page: int = 5,
) -> jax.Array:
    """Biphasic adaptor used by pipeline simulate_path. Not for use under jit/vmap."""
    fn = simulator_biphasic(T=T, start_volume=start_volume, page=page)
    seed = int(
        jax.device_get(jax.random.randint(key, (), 0, 2**31 - 1, dtype=jnp.uint32))
    )
    return jnp.asarray(fn(theta, seed), dtype=jnp.float32)


__all__ = [
    "simulator_monophasic",
    "simulator_biphasic",
    "summary_identity",
    "summary_log",
    "theta_bounds_monophasic",
    "theta_bounds_biphasic",
    "prior_monophasic",
    "prior_biphasic",
    "simulate",
    "simulate_biphasic",
]

if __name__ == "__main__":  # tiny smoke test
    sim = simulator_monophasic(T=5)
    y = sim([0.05, 0.01, 30, 48], seed=123)
    print("shape:", y.shape)

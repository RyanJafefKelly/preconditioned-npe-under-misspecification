# src/precond_npe_misspec/engine/preconditioning.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from precond_npe_misspec.algorithms.smc_abc import run_smc_abc
from precond_npe_misspec.utils import distances as dist

type Array = jax.Array
DistanceFn = Callable[[Array, Array], Array]
DistanceFactory = Callable[[Array], DistanceFn]  # input: pilot S -> distance fn


# mypy: disable-error-code=misc


def _to_vec(d: jnp.ndarray) -> jnp.ndarray:
    """Ensure distances reduce to shape (B,)."""
    return d if d.ndim == 1 else jnp.mean(d.reshape(d.shape[0], -1), axis=-1)


def _make_dataset(
    spec: Any,
    key: Array,
    n: int,
    *,
    batch_size: int | None = None,
    sim_kwargs: dict[str, Any] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate (theta, S, x) batches from the prior."""
    if batch_size is None:
        batch_size = max(1, min(n, 2048))
    sim_kwargs = {} if sim_kwargs is None else sim_kwargs

    k_theta, k_sim = jax.random.split(key)

    @eqx.filter_jit  # compile two shapes at most (full and last partial)
    def _simulate_batch(th_keys: Array, sm_keys: Array) -> tuple[Array, Array, Array]:
        thetas_b = jax.vmap(spec.prior_sample)(th_keys)  # (B, θ)
        xs_b = jax.vmap(lambda kk, th: spec.simulate(kk, th, **sim_kwargs))(sm_keys, thetas_b)  # (B, n_obs)
        S_b = jax.vmap(spec.summaries)(xs_b)  # (B, d)
        return thetas_b, S_b, xs_b

    th_parts, S_parts, x_parts = [], [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = jnp.arange(start, end, dtype=jnp.uint32)
        th_keys = jax.vmap(lambda i: jax.random.fold_in(k_theta, i))(idx)
        sm_keys = jax.vmap(lambda i: jax.random.fold_in(k_sim, i))(idx)
        th_b, S_b, x_b = _simulate_batch(th_keys, sm_keys)
        th_parts.append(th_b)
        S_parts.append(S_b)
        x_parts.append(x_b)

    thetas = jnp.concatenate(th_parts, axis=0)
    S = jnp.concatenate(S_parts, axis=0)
    xs = jnp.concatenate(x_parts, axis=0)

    mask_theta = jnp.all(jnp.isfinite(thetas), axis=1)
    mask_S = jnp.all(jnp.isfinite(S), axis=1)
    S_MAX = jnp.asarray(1e30, dtype=S.dtype)
    mask_mag = jnp.all(jnp.abs(S) < S_MAX, axis=1)

    mask = mask_theta & mask_S & mask_mag

    thetas = thetas[mask]
    S = S[mask]
    xs = xs[mask]

    print(f"jnp.max(S)={jnp.max(S)}, jnp.min(thetas)={jnp.min(S)}")
    print(f"Is nan: {jnp.sum(jnp.isnan(thetas))}")

    return thetas, S, xs


def _abc_rejection_with_sim(
    spec: Any,
    key: Array,
    s_obs: jnp.ndarray,
    *,
    n_sims: int,
    q: float,
    dist_fn: DistanceFn,
    batch_size: int | None,
    sim_kwargs: dict[str, Any] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rejection ABC with on‑the‑fly simulation. Returns accepted (θ, S, x)."""
    if batch_size is None:
        batch_size = max(1, min(n_sims, 2048))
    sim_kwargs = {} if sim_kwargs is None else sim_kwargs

    k_th_base, k_sm_base = jax.random.split(key)
    th_chunks, S_chunks, x_chunks, d_chunks = [], [], [], []

    for start in range(0, n_sims, batch_size):
        end = min(start + batch_size, n_sims)
        idx = jnp.arange(start, end, dtype=jnp.uint32)

        th_keys = jax.vmap(lambda i: jax.random.fold_in(k_th_base, i))(idx)
        th_b = jax.vmap(spec.prior_sample)(th_keys)  # (B, θ)

        sm_keys = jax.vmap(lambda i: jax.random.fold_in(k_sm_base, i))(idx)
        xs_b = jax.vmap(lambda kk, th: spec.simulate(kk, th, **sim_kwargs))(sm_keys, th_b)
        S_b = jax.vmap(spec.summaries)(xs_b)  # (B, d)

        d_b = _to_vec(dist_fn(S_b, s_obs))  # (B,)
        th_chunks.append(th_b)
        S_chunks.append(S_b)
        x_chunks.append(xs_b)
        d_chunks.append(d_b)

    thetas_all = jnp.concatenate(th_chunks, axis=0)
    S_all = jnp.concatenate(S_chunks, axis=0)
    x_all = jnp.concatenate(x_chunks, axis=0)
    d_all = jnp.concatenate(d_chunks, axis=0)

    finite = jnp.isfinite(d_all)
    thetas_all = thetas_all[finite]
    S_all = S_all[finite]
    x_all = x_all[finite]
    d_all = d_all[finite]
    n_tot = int(d_all.shape[0])

    # Keep exactly n_keep: q in (0,1] => fraction; q>=1 => absolute count.
    n_keep = max(1, min(n_tot, int(jnp.ceil(q * n_tot)))) if 0 < q <= 1.0 else max(1, min(n_tot, int(q)))

    idx = jnp.argpartition(d_all, n_keep - 1)[:n_keep]
    idx = idx[jnp.argsort(d_all[idx])]
    eps_star = float(d_all[idx][-1])

    print(f"Preconditioning: rejection | kept={n_keep}/{n_tot} | eps*={eps_star:0.6g}")
    return thetas_all[idx], S_all[idx], x_all[idx]


def run_preconditioning(
    key: Array,
    spec: Any,
    s_obs: jnp.ndarray,
    run: Any,
    flow_cfg: Any,  # unused here, kept for a stable call signature
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return training triples (theta_train, S_train, X_train) according to:
      - method="none"       → prior draws
      - method="rejection"  → rejection ABC
      - method="smc_abc"    → SMC‑ABC
    """
    sim_kwargs = {} if getattr(run, "sim_kwargs", None) is None else dict(run.sim_kwargs)
    batch_size = int(getattr(run, "batch_size", 512))
    method = run.precond.method

    # Optional pilot set for distance factory
    S_pilot: jnp.ndarray | None = None
    if getattr(spec, "make_distance", None) is not None and method in {
        "rejection",
        "smc_abc",
    }:
        key, k_pilot = jax.random.split(key)
        _, S_pilot, _ = _make_dataset(
            spec,
            k_pilot,
            run.precond.n_sims,
            batch_size=batch_size,
            sim_kwargs=sim_kwargs,
        )

    if method == "none":
        key, k_data = jax.random.split(key)
        theta_train, S_train, x_train = _make_dataset(
            spec,
            k_data,
            run.precond.n_sims,
            batch_size=batch_size,
            sim_kwargs=sim_kwargs,
        )
        theta_train = theta_train.astype(jnp.float32)
        S_train = S_train.astype(jnp.float32)
        print(f"Preconditioning: none    | training pairs: {int(theta_train.shape[0])}")
        return theta_train, S_train, x_train

    if method == "rejection":
        # Distance selection
        if getattr(spec, "make_distance", None) is None:
            dist_fn: DistanceFn = dist.euclidean
        else:
            assert S_pilot is not None
            dist_fn = spec.make_distance(S_pilot)
        key, k_abc = jax.random.split(key)
        th, S, X = _abc_rejection_with_sim(
            spec,
            k_abc,
            s_obs,
            n_sims=run.precond.n_sims,
            q=run.precond.q_precond,
            dist_fn=dist_fn,
            batch_size=batch_size,
            sim_kwargs=sim_kwargs,
        )
        return (
            th.astype(jnp.float32),
            S.astype(jnp.float32),
            X,
        )

    if method == "smc_abc":
        key, k_smc = jax.random.split(key)
        theta_particles, S_particles, x_particles = run_smc_abc(
            key=k_smc,
            n_particles=run.precond.smc_n_particles,
            epsilon0=run.precond.smc_epsilon0,
            alpha=run.precond.smc_alpha,
            eps_min=run.precond.smc_eps_min,
            acc_min=run.precond.smc_acc_min,
            max_iters=run.precond.smc_max_iters,
            initial_R=run.precond.smc_initial_R,
            c_tuning=run.precond.smc_c_tuning,
            B_sim=run.precond.smc_B_sim,
            spec=spec,
            s_obs=s_obs,
            sim_kwargs=sim_kwargs,
            S_pilot_for_distance=S_pilot,
        )
        theta_particles = theta_particles.astype(jnp.float32)
        S_particles = S_particles.astype(jnp.float32)
        print(f"Preconditioning: smc_abc | particles: {int(theta_particles.shape[0])}")
        return theta_particles, S_particles, x_particles

    raise ValueError(f"Unknown preconditioning method: {method!r}")

"""Base code for NLE + ABC pipelines under model misspecification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.numpy as jnp
from flowjax.distributions import Normal, Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

from precond_npe_misspec.utils import distances as dist

type Array = jax.Array

DistanceFn = dist.DistanceFn

DistanceFactory = Callable[[jnp.ndarray], DistanceFn]  # input: S_tr_w (n,d)


@dataclass(frozen=True)
class FlowConfig:
    # NOTE: trialing increasing these
    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 8.0
    learning_rate: float = 5e-4
    max_epochs: int = 500
    max_patience: int = 10
    batch_size: int = 512


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    theta_dim: int
    s_dim: int
    prior_sample: Callable[[Array], jnp.ndarray]
    true_dgp: Callable[..., jnp.ndarray]  # true_dgp(key, theta, **sim_kwargs) -> x
    simulate: Callable[..., jnp.ndarray]  # simulate(key, theta, **sim_kwargs) -> x
    summaries: Callable[[jnp.ndarray], jnp.ndarray]  # s = summaries(x)
    build_flow: Callable[[Array, FlowConfig], eqx.Module]
    baseline_posterior: Callable[[jnp.ndarray], tuple[float, float]] | None = None
    make_distance: DistanceFactory | None = None


def default_coupling_rqs_builder(s_dim: int, theta_dim: int) -> Callable[[Array, FlowConfig], eqx.Module]:
    def _builder(key: Array, cfg: FlowConfig) -> eqx.Module:
        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(s_dim)),
            transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
            cond_dim=theta_dim,
            flow_layers=cfg.flow_layers,
            nn_width=cfg.nn_width,
        )

    return _builder


# ---------------- Core ops ----------------


def make_dataset(spec: ExperimentSpec, key: Array, n: int, **sim_kwargs: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    k_theta, k_sim = jax.random.split(key)
    thetas = jax.vmap(spec.prior_sample)(jax.random.split(k_theta, n))  # (n, ...) or (n,)
    xs = jax.vmap(lambda k, th: spec.simulate(k, th, **sim_kwargs))(jax.random.split(k_sim, n), thetas)
    S = jax.vmap(spec.summaries)(xs)
    return thetas, S


@dataclass
class TrainedFlow:
    flow: eqx.Module
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean: jnp.ndarray
    th_std: jnp.ndarray
    losses: jnp.ndarray


def fit_conditional_likelihood_flow(
    spec: ExperimentSpec,
    key: Array,
    S: jnp.ndarray,
    theta: jnp.ndarray,
    cfg: FlowConfig,
) -> TrainedFlow:
    S_mean, S_std = jnp.mean(S, 0), jnp.std(S, 0) + 1e-6
    th_mean, th_std = jnp.mean(theta, 0), jnp.std(theta, 0) + 1e-6
    S_proc = (S - S_mean) / S_std
    th_proc = (theta - th_mean) / th_std
    if th_proc.ndim == 1:
        th_proc = th_proc[:, None]

    flow0 = spec.build_flow(key, cfg)
    key, sub = jax.random.split(key)
    flow, losses = fit_to_data(
        key=sub,
        dist=flow0,
        data=(S_proc, th_proc),  # MLE
        learning_rate=cfg.learning_rate,
        max_epochs=cfg.max_epochs,
        max_patience=cfg.max_patience,
        batch_size=cfg.batch_size,
        show_progress=True,
    )
    s_shape: tuple[int, ...] = tuple(S_proc.shape[1:])
    cond_shape: tuple[int, ...] = tuple(th_proc.shape[1:])

    flow = Transformed(
        base_dist=flow,
        bijection=bij.Affine(S_mean, S_std),  # maps s_w -> s = mean + std * s_w
        shape=s_shape,
        cond_shape=cond_shape,
    )
    return TrainedFlow(flow, S_mean, S_std, th_mean, th_std, losses)


def _standardise(S: jnp.ndarray, m: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    return (S - m) / s


def abc_rejection_with_sim(
    spec: ExperimentSpec,
    key: Array,
    s_obs: jnp.ndarray,
    n_props: int,
    q: float,
    S_mean: jnp.ndarray,
    S_std: jnp.ndarray,
    dist_fn: Callable[[Array, Array], Array],
    n_rep_summaries: int = 1,
    batch_size: int | None = None,
    **sim_kwargs: Any,
) -> jnp.ndarray:
    """Batched ABC rejection using the true simulator. Deterministic via fold_in."""
    R = max(1, int(n_rep_summaries))
    if batch_size is None:
        batch_size = max(1, min(n_props, 16384 // R))

    k_th_base, k_sm_base = jax.random.split(key)
    s_obs_w = _standardise(s_obs, S_mean, S_std)

    def _theta_batch(idx_b: jnp.ndarray) -> Array:
        th_keys_b = jax.vmap(lambda i: jax.random.fold_in(k_th_base, i))(idx_b)
        return jax.vmap(spec.prior_sample)(th_keys_b)

    def _simkeys_batch(idx_b: jnp.ndarray) -> Array:
        if R == 1:
            return jax.vmap(lambda i: jax.random.fold_in(k_sm_base, i))(idx_b)  # (B, key)
        r_idx = jnp.arange(R, dtype=jnp.uint32)

        def keys_for_i(i: jnp.ndarray) -> Array:
            return jax.vmap(lambda r: jax.random.fold_in(k_sm_base, i * R + r))(r_idx)  # (R,key)

        return jax.vmap(keys_for_i)(idx_b)  # (B,R,key)

    def _simulate_batch(idx_b: jnp.ndarray) -> tuple[Array, Array]:
        th_b = _theta_batch(idx_b)  # (B,θ) or (B,)
        ks_b = _simkeys_batch(idx_b)
        if R == 1:
            xs = jax.vmap(lambda k, th: spec.simulate(k, th, **sim_kwargs))(ks_b, th_b)  # (B,...)
            S = jax.vmap(spec.summaries)(xs)  # (B,d)
        else:

            def sim_r(krow: Array, th: Array) -> Array:
                xs = jax.vmap(lambda kk: spec.simulate(kk, th, **sim_kwargs))(krow)  # (R,...)
                return jax.vmap(spec.summaries)(xs)  # (R,d)

            S = jax.vmap(sim_r)(ks_b, th_b)  # (B,R,d)
        return th_b, _standardise(S, S_mean, S_std)

    simulate_batch = eqx.filter_jit(_simulate_batch)

    def _to_vector(x: jnp.ndarray) -> jnp.ndarray:
        return x if x.ndim == 1 else jnp.mean(x.reshape(x.shape[0], -1), axis=-1)

    # distances
    d_chunks: list[jnp.ndarray] = []
    batch_slices: list[tuple[int, int]] = []
    for start in range(0, n_props, batch_size):
        end = min(start + batch_size, n_props)
        idx_b = jnp.arange(start, end, dtype=jnp.uint32)
        _th_b, Sw = simulate_batch(idx_b)
        d_b = dist_fn(Sw, s_obs_w)
        d_chunks.append(_to_vector(d_b))
        batch_slices.append((start, end))

    d_all = jnp.concatenate(d_chunks, axis=0)  # (N,)
    eps = jnp.quantile(d_all, q)

    # acc thetas
    acc_parts: list[jnp.ndarray] = []
    for (start, end), d_b in zip(batch_slices, d_chunks, strict=True):
        idx_b = jnp.arange(start, end, dtype=jnp.uint32)
        th_b = _theta_batch(idx_b)
        mask = _to_vector(d_b) <= eps
        acc_parts.append(th_b[mask])

    return (
        jnp.concatenate(acc_parts, axis=0)
        if any(p.shape[0] for p in acc_parts)
        else _theta_batch(jnp.arange(0, 0, dtype=jnp.uint32))
    )


def smc_abc_with_surrogate() -> None:
    pass


def abc_rejection_with_surrogate(
    spec: ExperimentSpec,
    trained: TrainedFlow,
    key: Array,
    s_obs: jnp.ndarray,
    n_props: int,
    q: float,
    dist_fn: Callable[[Array, Array], Array],
    n_rep_summaries: int = 1,
    batch_size: int | None = None,
) -> jnp.ndarray:
    """Batched ABC with neural surrogate. Deterministic via fold_in. Memory safe."""
    # TODO: jit anything
    R = max(1, int(n_rep_summaries))
    if batch_size is None:
        batch_size = max(1, min(n_props, 16384 // R))

    k_th_base, k_flow_base = jax.random.split(key)
    s_obs_w = _standardise(s_obs, trained.S_mean, trained.S_std)

    def _theta_batch(idx_b: jnp.ndarray) -> Array:
        th_keys_b = jax.vmap(lambda i: jax.random.fold_in(k_th_base, i))(idx_b)
        return jax.vmap(spec.prior_sample)(th_keys_b)

    def _cond_batch(th_b: Array) -> Array:
        thc_b = (th_b - trained.th_mean) / trained.th_std
        return thc_b[:, None] if thc_b.ndim == 1 else thc_b

    def _flowkeys_batch(idx_b: jnp.ndarray) -> Array:
        return jax.vmap(lambda i: jax.random.fold_in(k_flow_base, i))(idx_b)  # (B,key)

    def _sample_standardise_batch(idx_b: jnp.ndarray) -> tuple[Array, Array]:
        th_b = _theta_batch(idx_b)  # (B,θ) or (B,)
        thc_b = _cond_batch(th_b)  # (B,cond)
        fk_b = _flowkeys_batch(idx_b)  # (B,key)
        if R == 1:
            S = jax.vmap(lambda k, c: trained.flow.sample(k, condition=c))(fk_b, thc_b)  # (B,d)
        else:
            S = jax.vmap(lambda k, c: trained.flow.sample(k, (R,), condition=c))(fk_b, thc_b)  # (B,R,d)
        return th_b, _standardise(S, trained.S_mean, trained.S_std)

    def _to_vector(x: jnp.ndarray) -> jnp.ndarray:
        return x if x.ndim == 1 else jnp.mean(x.reshape(x.shape[0], -1), axis=-1)

    def _jit_dist(fn: DistanceFn) -> DistanceFn:
        def _eval(Sw: Array, s_obs_w: Array) -> Array:
            return _to_vector(fn(Sw, s_obs_w))

        return cast(DistanceFn, eqx.filter_jit(_eval))

    dist_eval: DistanceFn = _jit_dist(dist_fn)

    # distances
    th_chunks: list[jnp.ndarray] = []
    d_chunks: list[jnp.ndarray] = []
    batch_slices: list[tuple[int, int]] = []
    # TODO? batch via scan function
    for start in range(0, n_props, batch_size):
        end = min(start + batch_size, n_props)
        idx_b = jnp.arange(start, end, dtype=jnp.uint32)
        th_b, Sw = _sample_standardise_batch(idx_b)  # keep θ
        d_b = dist_eval(Sw, s_obs_w)
        th_chunks.append(th_b)
        d_chunks.append(_to_vector(d_b))  # (B,)
        batch_slices.append((start, end))

    d_all = jnp.concatenate(d_chunks, axis=0)  # (N,)
    eps = jnp.quantile(d_all, q)
    acc_parts = [th_b[d_b <= eps] for th_b, d_b in zip(th_chunks, d_chunks, strict=True)]

    return (
        jnp.concatenate(acc_parts, axis=0)
        if any(p.shape[0] for p in acc_parts)
        else _theta_batch(jnp.arange(0, 0, dtype=jnp.uint32))
    )


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    theta_true: float | jnp.ndarray = 0.0
    n_train: int = 4000
    n_props: int = 20000
    q_accept: float = 0.01
    sim_kwargs: dict[str, Any] | None = None  # e.g. {"n_obs": 100, "stdev_err": 2.0}
    n_rep_summaries: int = 1
    batch_size: int = 256


@dataclass
class RunResult:
    acc_true: jnp.ndarray
    acc_surr: jnp.ndarray
    x_obs: jnp.ndarray
    s_obs: jnp.ndarray
    trained: TrainedFlow


def run_experiment(spec: ExperimentSpec, run: RunConfig, flow_cfg: FlowConfig) -> RunResult:
    rng = jax.random.key(run.seed)
    sim_kwargs = {} if run.sim_kwargs is None else dict(run.sim_kwargs)

    rng, k_obs = jax.random.split(rng)

    # TODO: should also have option to pass in real data
    x_obs = spec.true_dgp(k_obs, jnp.asarray(run.theta_true), **sim_kwargs)
    s_obs = spec.summaries(x_obs)

    # 2) Training data
    rng, k_tr = jax.random.split(rng)
    theta_tr, S_tr = make_dataset(spec, k_tr, run.n_train, **sim_kwargs)

    # 3) Fit conditional likelihood flow
    rng, k_flow = jax.random.split(rng)
    trained = fit_conditional_likelihood_flow(spec, k_flow, S_tr, theta_tr, flow_cfg)

    # 4) Memory‑safe distance function construction
    # For MMD with median heuristic, using all S_tr_w is O(n^2) memory.
    # Subsample a reference set for bandwidth / covariance estimation.
    rng, k_ref = jax.random.split(rng)
    ref_size_default = 2048
    ref_size = int(getattr(run, "distance_ref_size", ref_size_default))
    ref_size = max(1, min(ref_size, S_tr.shape[0]))

    if spec.make_distance is None:
        dist_fn = dist.euclidean
    else:
        if ref_size == S_tr.shape[0]:
            S_ref_w = _standardise(S_tr, trained.S_mean, trained.S_std)  # (N,d)  # type: ignore
        else:
            idx = jax.random.choice(k_ref, S_tr.shape[0], (ref_size,), replace=False)
            S_ref = jnp.take(S_tr, idx, axis=0)  # (ref_size,d)
            S_ref_w = _standardise(S_ref, trained.S_mean, trained.S_std)
        dist_fn = spec.make_distance(S_ref_w)  # type: ignore

    # 5) ABC with true simulator (batched)
    rng, k_true = jax.random.split(rng)
    acc_true = abc_rejection_with_sim(
        spec,
        k_true,
        s_obs,
        run.n_props,
        run.q_accept,
        trained.S_mean,
        trained.S_std,
        dist_fn,
        n_rep_summaries=run.n_rep_summaries,
        batch_size=getattr(run, "batch_size", None),
        **sim_kwargs,
    )

    # 6) ABC with surrogate (batched)
    rng, k_surr = jax.random.split(rng)
    acc_surr = abc_rejection_with_surrogate(
        spec,
        trained,
        k_surr,
        s_obs,
        run.n_props,
        run.q_accept,
        dist_fn,
        n_rep_summaries=run.n_rep_summaries,
        batch_size=getattr(run, "batch_size", None),
    )

    return RunResult(acc_true, acc_surr, x_obs, s_obs, trained)

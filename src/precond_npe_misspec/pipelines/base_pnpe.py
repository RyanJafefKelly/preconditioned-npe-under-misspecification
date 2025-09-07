from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.numpy as jnp
from flowjax.distributions import Normal, Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

from precond_npe_misspec.utils import distances as dist

type Array = jax.Array
DistanceFn = Callable[[Array, Array], Array]
DistanceFactory = Callable[[Array], DistanceFn]  # input: S_ref_raw (n,d)


# ---------------- Spec and configs ----------------


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    theta_dim: int
    s_dim: int
    prior_sample: Callable[[Array], jnp.ndarray]
    true_dgp: Callable[..., jnp.ndarray]  # true_dgp(key, theta, **sim_kwargs) -> x
    simulate: Callable[..., jnp.ndarray]  # simulate(key, theta, **sim_kwargs) -> x
    summaries: Callable[[jnp.ndarray], jnp.ndarray]  # s = summaries(x)
    # Builders
    build_theta_flow: Callable[[Array, FlowConfig], eqx.Module]
    build_posterior_flow: Callable[[Array, FlowConfig], eqx.Module]
    # Optional distance factory (as in base_nle_abc)
    make_distance: DistanceFactory | None = None


@dataclass(frozen=True)
class FlowConfig:
    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 8.0
    learning_rate: float = 5e-4
    max_epochs: int = 500
    max_patience: int = 10
    batch_size: int = 512


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    theta_true: float | jnp.ndarray = 0.0
    # Preconditioning ABC
    n_sims: int = 200_000  # number of simulations to run
    q_precond: float = 0.1  # acceptance quantile
    # Post-processing
    n_posterior_draws: int = 5000
    # Simulation kwargs
    sim_kwargs: dict[str, Any] | None = None
    # Distances
    n_rep_summaries: int = 1
    batch_size: int = 256


@dataclass
class RunResult:
    theta_acc_precond: jnp.ndarray
    posterior_flow: eqx.Module
    x_obs: jnp.ndarray
    s_obs: jnp.ndarray
    # stats for conditioning
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean_post: jnp.ndarray
    th_std_post: jnp.ndarray
    posterior_samples_at_obs: jnp.ndarray


# ---------------- Helpers ----------------


# NOTE: DO I ACTUALLY NEED THIS FOR PNPE? I dont think so...
def default_theta_flow_builder(
    theta_dim: int,
) -> Callable[[Array, FlowConfig], eqx.Module]:
    def _builder(key: Array, cfg: FlowConfig) -> eqx.Module:
        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(theta_dim)),
            transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
            cond_dim=None,
            flow_layers=cfg.flow_layers,
            nn_width=cfg.nn_width,
        )

    return _builder


def default_posterior_flow_builder(theta_dim: int, s_dim: int) -> Callable[[Array, FlowConfig], eqx.Module]:
    def _builder(key: Array, cfg: FlowConfig) -> eqx.Module:
        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(theta_dim)),  # random variable is θ
            transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
            cond_dim=s_dim,  # condition on s
            flow_layers=cfg.flow_layers,
            nn_width=cfg.nn_width,
        )

    return _builder


def _standardise(x: jnp.ndarray, m: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    return (x - m) / (s + 1e-6)


def _make_dataset(
    spec: ExperimentSpec,
    key: Array,
    n: int,
    batch_size: int | None = None,
    **sim_kwargs: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Memory‑safe dataset: generate θ and summaries S in batches."""
    if batch_size is None:
        batch_size = max(1, min(n, 2048))

    k_theta, k_sim = jax.random.split(key)

    @eqx.filter_jit  # type: ignore[misc]
    def _simulate_batch(th_keys: Array, sm_keys: Array) -> tuple[Array, Array]:
        thetas_b = jax.vmap(spec.prior_sample)(th_keys)  # (B, θ)
        xs_b = jax.vmap(lambda kk, th: spec.simulate(kk, th, **sim_kwargs))(sm_keys, thetas_b)
        S_b = jax.vmap(spec.summaries)(xs_b)  # (B, d)
        return thetas_b, S_b

    th_parts: list[jnp.ndarray] = []
    S_parts: list[jnp.ndarray] = []

    # Two shapes -> at most two compilations (full batch and last partial).
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = jnp.arange(start, end, dtype=jnp.uint32)
        th_keys = jax.vmap(lambda i: jax.random.fold_in(k_theta, i))(idx)
        sm_keys = jax.vmap(lambda i: jax.random.fold_in(k_sim, i))(idx)
        th_b, S_b = _simulate_batch(th_keys, sm_keys)
        th_parts.append(th_b)
        S_parts.append(S_b)

    thetas = jnp.concatenate(th_parts, axis=0)
    S = jnp.concatenate(S_parts, axis=0)
    return thetas, S


def _abc_rejection_with_sim(
    spec: ExperimentSpec,
    key: Array,
    s_obs: jnp.ndarray,
    n_sims: int,
    q: float,
    # S_mean: jnp.ndarray,
    # S_std: jnp.ndarray,
    dist_fn: DistanceFn,
    batch_size: int | None,
    **sim_kwargs: Any,
) -> tuple[Array, Array]:
    if batch_size is None:
        batch_size = max(1, min(n_sims, 2048))

    k_th_base, k_sm_base = jax.random.split(key)

    def _to_vec(d: jnp.ndarray) -> jnp.ndarray:
        # Support distances returning (B,) or (B,R,...). Reduce over non-batch axes.
        return d if d.ndim == 1 else jnp.mean(d.reshape(d.shape[0], -1), axis=-1)

    th_chunks: list[jnp.ndarray] = []
    S_chunks: list[jnp.ndarray] = []
    d_chunks: list[jnp.ndarray] = []

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
        d_chunks.append(d_b)

    # Concatenate once.
    thetas_all = jnp.concatenate(th_chunks, axis=0)  # (N, θ)
    S_all = jnp.concatenate(S_chunks, axis=0)  # (N, d)
    d_all = jnp.concatenate(d_chunks, axis=0)  # (N,)

    # Drop non-finite distances defensively.
    finite = jnp.isfinite(d_all)
    thetas_all = thetas_all[finite]
    S_all = S_all[finite]
    d_all = d_all[finite]
    n_tot = int(d_all.shape[0])

    # Keep exactly n_keep items: q as fraction in (0,1], or integer count if q>=1.
    n_keep = max(1, min(n_tot, math.ceil(q * n_tot))) if 0 < q <= 1.0 else max(1, min(n_tot, int(q)))

    # Indices of the n_keep smallest distances.
    idx = jnp.argpartition(d_all, n_keep - 1)[:n_keep]
    # Optional: sort accepted by distance.
    idx = idx[jnp.argsort(d_all[idx])]

    theta_acc = thetas_all[idx]
    S_acc = S_all[idx]
    return theta_acc, S_acc


def preconditioning_step(
    spec: ExperimentSpec,
    key: Array,
    s_obs: jnp.ndarray,
    run: RunConfig,
    flow_cfg: FlowConfig,
) -> tuple[Array, Array]:
    sim_kwargs = {} if run.sim_kwargs is None else dict(run.sim_kwargs)

    # 1) Pilot set to summaries for ABC distance
    key, k_pilot = jax.random.split(key)
    _, S_pilot = _make_dataset(spec, k_pilot, run.n_sims, **sim_kwargs)

    # 2) Distance
    if spec.make_distance is None:
        dist_fn: DistanceFn = dist.euclidean  # no cast
    else:
        dist_fn = spec.make_distance(S_pilot)  # returns DistanceFn

    # 3) ABC rejection to get concentrated θ set
    key, k_abc = jax.random.split(key)
    theta_acc, S_acc = _abc_rejection_with_sim(
        spec,
        k_abc,
        s_obs,
        run.n_sims,
        run.q_precond,
        # S_mean,
        # S_std,
        dist_fn,
        batch_size=run.batch_size,
        **sim_kwargs,
    )
    assert theta_acc.shape[0] > 0, "ABC returned zero accepted parameters."

    print("Number of accepted parameters:", theta_acc.shape[0])

    return theta_acc, S_acc


@dataclass
class _PosteriorTrained:
    flow: eqx.Module
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean: jnp.ndarray
    th_std: jnp.ndarray


def npe_step(
    spec: ExperimentSpec,
    key: Array,
    theta_acc: Array,
    S_pilot: Array,
    run: RunConfig,
    flow_cfg: FlowConfig,
) -> _PosteriorTrained:
    # sim_kwargs = {} if run.sim_kwargs is None else dict(run.sim_kwargs)

    # TODO:

    # 2) Standardise for stable training
    S_mean, S_std = jnp.mean(S_pilot, 0), jnp.std(S_pilot, 0) + 1e-8
    th_mean, th_std = jnp.mean(theta_acc, 0), jnp.std(theta_acc, 0) + 1e-8
    S_proc = _standardise(S_pilot, S_mean, S_std)
    th_proc = _standardise(theta_acc, th_mean, th_std)

    # 3) Fit conditional flow for p(θ | s)
    key, k_build, k_fit = jax.random.split(key, 3)
    flow0 = spec.build_posterior_flow(k_build, flow_cfg)
    flow_fit, _losses = fit_to_data(
        key=k_fit,
        dist=flow0,
        data=(th_proc, S_proc),  # conditional MLE
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )
    posterior_flow = Transformed(flow_fit, bij.Invert(bij.Affine(-th_mean / th_std, 1.0 / th_std)))
    return _PosteriorTrained(posterior_flow, S_mean, S_std, th_mean, th_std)


def run_experiment(spec: ExperimentSpec, run: RunConfig, flow_cfg: FlowConfig) -> RunResult:
    rng = jax.random.key(run.seed)
    sim_kwargs = {} if run.sim_kwargs is None else dict(run.sim_kwargs)

    # Observed data
    rng, k_obs = jax.random.split(rng)
    x_obs = spec.true_dgp(k_obs, jnp.asarray(run.theta_true), **sim_kwargs)
    s_obs = spec.summaries(x_obs)

    # Preconditioning
    rng, k_pre = jax.random.split(rng)

    theta_acc, S_acc = preconditioning_step(spec, k_pre, s_obs, run, flow_cfg)

    rng, k_npe = jax.random.split(rng)
    posterior = npe_step(spec, k_npe, theta_acc, S_acc, run, flow_cfg)

    rng, k_post = jax.random.split(rng)
    s_obs_w = _standardise(s_obs, posterior.S_mean, posterior.S_std)
    print("s_obs_w: ", s_obs_w)
    th_samps = posterior.flow.sample(k_post, (run.n_posterior_draws,), condition=s_obs_w)

    return RunResult(
        theta_acc_precond=theta_acc,
        posterior_flow=posterior.flow,
        x_obs=x_obs,
        s_obs=s_obs,
        S_mean=posterior.S_mean,
        S_std=posterior.S_std,
        th_mean_post=posterior.th_mean,
        th_std_post=posterior.th_std,
        posterior_samples_at_obs=th_samps,
    )

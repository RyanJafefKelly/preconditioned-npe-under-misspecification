"""Base code for preconditioned NPE (PNPE) experiments."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib
from flowjax.distributions import Normal
from flowjax.distributions import Transformed as _Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

from precond_npe_misspec.algorithms.smc_abc import run_smc_abc
from precond_npe_misspec.examples.embeddings import EmbedBuilder
from precond_npe_misspec.utils import distances as dist
from precond_npe_misspec.utils.artifacts import save_artifacts

type Array = jax.Array
DistanceFn = Callable[[Array, Array], Array]
DistanceFactory = Callable[[Array], DistanceFn]  # input: S_ref_raw (n,d)

if TYPE_CHECKING:

    class _EqxModule:  # pragma: no cover - typing shim for eqx.Module
        def sample(self, key: Array, shape: tuple[int, ...], *, condition: Array) -> Array: ...

else:
    _EqxModule = eqx.Module

matplotlib.use("Agg")

EPS = 1e-8


def _to_unconstrained(theta: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    # map (lo,hi) -> R via logit
    p = (theta - lo) / (hi - lo)
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _from_unconstrained(u: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    # map R -> (lo,hi) via sigmoid
    return lo + (hi - lo) * jnn.sigmoid(u)


class _BoundedPosterior(_EqxModule):
    base: eqx.Module  # flow over u_proc
    u_mean: jnp.ndarray  # (theta_dim,)
    u_std: jnp.ndarray  # (theta_dim,)
    lo: jnp.ndarray  # (theta_dim,)
    hi: jnp.ndarray  # (theta_dim,)

    def __init__(
        self,
        *,
        base: eqx.Module,
        u_mean: jnp.ndarray,
        u_std: jnp.ndarray,
        lo: jnp.ndarray,
        hi: jnp.ndarray,
    ) -> None:
        self.base = base
        self.u_mean = u_mean
        self.u_std = u_std
        self.lo = lo
        self.hi = hi

    def sample(self, key: Array, shape: tuple[int, ...], *, condition: Array) -> Array:
        u_proc = cast(Array, self.base.sample(key, shape, condition=condition))
        u = u_proc * self.u_std + self.u_mean
        return _from_unconstrained(u, self.lo, self.hi)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    theta_dim: int
    s_dim: int
    prior_sample: Callable[[Array], jnp.ndarray]
    prior_logpdf: Callable[[Array], Array] | None
    true_dgp: Callable[..., jnp.ndarray]  # true_dgp(key, theta, **sim_kwargs) -> x
    simulate: Callable[..., jnp.ndarray]  # simulate(key, theta, **sim_kwargs) -> x
    summaries: Callable[[jnp.ndarray], jnp.ndarray]  # s = summaries(x)
    # Builders
    # build_theta_flow: Callable[[Array, FlowConfig], eqx.Module]
    build_posterior_flow: Callable[[Array, FlowConfig], eqx.Module]
    build_embedder: EmbedBuilder | None = None
    # Optional distance factory (as in base_nle_abc)
    make_distance: DistanceFactory | None = None
    theta_labels: tuple[str, ...] | None = None
    summary_labels: tuple[str, ...] | None = None
    # param range ... fall back to unconstrained if None
    theta_lo: jnp.ndarray | None = None  # shape (theta_dim,)
    theta_hi: jnp.ndarray | None = None  # shape (theta_dim,)
    # simulate, summary path - so can call for posterior predictive checks
    simulate_path: str | None = None
    summaries_path: str | None = None


@dataclass(frozen=True)
class FlowConfig:
    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 8.0
    learning_rate: float = 5e-4
    max_epochs: int = 50
    max_patience: int = 10
    batch_size: int = 512


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    obs_seed: int = 1234
    theta_true: float | jnp.ndarray = 0.0
    outdir: str | None = None
    # Preconditioning ABC
    precond_method: Literal["rejection", "smc_abc"] = "rejection"
    n_sims: int = 200_000  # number of simulations to run
    q_precond: float = 0.1  # acceptance quantile
    # Post-processing
    n_posterior_draws: int = 5000
    # Simulation kwargs
    sim_kwargs: dict[str, Any] | None = None
    # Distances
    n_rep_summaries: int = 1
    batch_size: int = 256
    # Plot/save options
    fig_dpi: int = 160
    fig_format: str = "pdf"
    # SMC‑ABC params
    smc_n_particles: int = 1_000
    smc_alpha: float = 0.5
    smc_epsilon0: float = 1e6
    smc_eps_min: float = 1e-3
    smc_acc_min: float = 0.1
    smc_max_iters: int = 5
    smc_initial_R: int = 1
    smc_c_tuning: float = 0.01
    smc_B_sim: int = 1


@dataclass
class RunResult:
    theta_acc_precond: jnp.ndarray
    S_acc_precond: jnp.ndarray
    posterior_flow: _EqxModule
    x_obs: jnp.ndarray
    s_obs: jnp.ndarray
    # stats for conditioning
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean_post: jnp.ndarray
    th_std_post: jnp.ndarray
    posterior_samples_at_obs: jnp.ndarray
    loss_history: Any


# NOTE: DO I ACTUALLY NEED THIS FOR PNPE? I dont think so...
# def default_theta_flow_builder(
#     theta_dim: int,
# ) -> Callable[[Array, FlowConfig], eqx.Module]:
#     def _builder(key: Array, cfg: FlowConfig) -> eqx.Module:
#         return coupling_flow(
#             key=key,
#             base_dist=Normal(jnp.zeros(theta_dim)),
#             transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
#             cond_dim=None,
#             flow_layers=cfg.flow_layers,
#             nn_width=cfg.nn_width,
#         )

#     return _builder


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
    return (x - m) / s


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

    def _simulate_batch(th_keys: Array, sm_keys: Array) -> tuple[Array, Array]:
        thetas_b = jax.vmap(spec.prior_sample)(th_keys)  # (B, θ)
        xs_b = jax.vmap(lambda kk, th: spec.simulate(kk, th, **sim_kwargs))(sm_keys, thetas_b)
        S_b = jax.vmap(spec.summaries)(xs_b)  # (B, d)
        return thetas_b, S_b

    _simulate_batch_compiled = cast(
        Callable[[Array, Array], tuple[Array, Array]],
        eqx.filter_jit(_simulate_batch),
    )

    th_parts: list[jnp.ndarray] = []
    S_parts: list[jnp.ndarray] = []

    # Two shapes -> at most two compilations (full batch and last partial).
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = jnp.arange(start, end, dtype=jnp.uint32)
        th_keys = jax.vmap(lambda i: jax.random.fold_in(k_theta, i))(idx)
        sm_keys = jax.vmap(lambda i: jax.random.fold_in(k_sim, i))(idx)
        th_b, S_b = _simulate_batch_compiled(th_keys, sm_keys)
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

    # Drop non-finite distances
    finite = jnp.isfinite(d_all)
    thetas_all = thetas_all[finite]
    S_all = S_all[finite]
    d_all = d_all[finite]
    n_tot = int(d_all.shape[0])

    # Keep exactly n_keep items: q as fraction in (0,1], or integer count if q>=1.
    n_keep = max(1, min(n_tot, math.ceil(q * n_tot))) if 0 < q <= 1.0 else max(1, min(n_tot, int(q)))

    # Indices of the n_keep smallest distances.
    idx = jnp.argpartition(d_all, n_keep - 1)[:n_keep]
    # Sort accepted by distance.
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

    # pilot set solely for distance construction if needed
    S_pilot: jnp.ndarray | None = None
    if spec.make_distance is not None:
        key, k_pilot = jax.random.split(key)
        _, S_pilot = _make_dataset(spec, k_pilot, run.n_sims, **sim_kwargs)

    if run.precond_method == "rejection":
        # fall back to existing rejection ABC
        if spec.make_distance is None:
            dist_fn: DistanceFn = dist.euclidean
        else:
            assert S_pilot is not None
            dist_fn = spec.make_distance(S_pilot)

        key, k_abc = jax.random.split(key)
        theta_acc, S_acc = _abc_rejection_with_sim(
            spec,
            k_abc,
            s_obs,
            run.n_sims,
            run.q_precond,
            dist_fn,
            batch_size=run.batch_size,
            **sim_kwargs,
        )
        print("Number of accepted parameters:", theta_acc.shape[0])
        return theta_acc, S_acc

    # SMC‑ABC path
    key, k_smc = jax.random.split(key)
    theta_particles, S_particles = run_smc_abc(
        key=k_smc,
        n_particles=run.smc_n_particles,
        epsilon0=run.smc_epsilon0,
        alpha=run.smc_alpha,
        eps_min=run.smc_eps_min,
        acc_min=run.smc_acc_min,
        max_iters=run.smc_max_iters,
        initial_R=run.smc_initial_R,
        c_tuning=run.smc_c_tuning,
        B_sim=run.smc_B_sim,
        spec=spec,
        s_obs=s_obs,
        sim_kwargs=sim_kwargs,
        S_pilot_for_distance=S_pilot,
    )
    print("Number of SMC particles:", theta_particles.shape[0])
    return theta_particles, S_particles


@dataclass
class _PosteriorTrained:
    flow: _EqxModule
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean: jnp.ndarray
    th_std: jnp.ndarray
    losses: Any


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
    S_mean, S_std = jnp.mean(S_pilot, 0), jnp.std(S_pilot, 0) + EPS
    th_mean, th_std = jnp.mean(theta_acc, 0), jnp.std(theta_acc, 0) + EPS
    S_proc = _standardise(S_pilot, S_mean, S_std)
    th_proc = _standardise(theta_acc, th_mean, th_std)

    # 3) Fit conditional flow for p(θ | s)
    key, k_build, k_fit = jax.random.split(key, 3)
    flow0 = spec.build_posterior_flow(k_build, flow_cfg)
    if (spec.theta_lo is not None) and (spec.theta_hi is not None):
        lo = jnp.asarray(spec.theta_lo)
        hi = jnp.asarray(spec.theta_hi)
        # unconstrained params
        u_acc = _to_unconstrained(theta_acc, lo, hi)
        u_mean, u_std = jnp.mean(u_acc, 0), jnp.std(u_acc, 0) + EPS
        u_proc = (u_acc - u_mean) / u_std

        flow_fit, _losses = fit_to_data(
            key=k_fit,
            dist=flow0,
            data=(u_proc, S_proc),
            learning_rate=flow_cfg.learning_rate,
            max_epochs=flow_cfg.max_epochs,
            max_patience=flow_cfg.max_patience,
            batch_size=flow_cfg.batch_size,
            show_progress=True,
        )
        # Wrap with sampler that maps back into (lo,hi)
        posterior_flow = _BoundedPosterior(
            base=flow_fit,
            u_mean=u_mean,
            u_std=u_std,
            lo=lo,
            hi=hi,
        )
        # Keep θ‑domain summary stats for reporting/artifacts
        th_mean, th_std = jnp.mean(theta_acc, 0), jnp.std(theta_acc, 0) + EPS
        return _PosteriorTrained(posterior_flow, S_mean, S_std, th_mean, th_std, _losses)

    # -------- fallback: original unconstrained training on θ --------
    th_mean, th_std = jnp.mean(theta_acc, 0), jnp.std(theta_acc, 0) + EPS
    th_proc = _standardise(theta_acc, th_mean, th_std)

    flow_fit, _losses = fit_to_data(
        key=k_fit,
        dist=flow0,
        data=(th_proc, S_proc),
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )

    Invert = cast(Any, bij.Invert)
    TransformedD = cast(Any, _Transformed)
    affine_bij = bij.Affine(-th_mean / th_std, 1.0 / th_std)
    invert_bij = Invert(affine_bij)
    posterior_flow = TransformedD(base_dist=flow_fit, bijection=invert_bij)
    return _PosteriorTrained(posterior_flow, S_mean, S_std, th_mean, th_std, _losses)


def run_experiment(spec: ExperimentSpec, run: RunConfig, flow_cfg: FlowConfig) -> RunResult:
    rng = jax.random.key(run.seed)
    obs_seed = jax.random.key(run.obs_seed)

    sim_kwargs = {} if run.sim_kwargs is None else dict(run.sim_kwargs)

    # Observed data
    rng, k_obs = jax.random.split(obs_seed)
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

    result = RunResult(
        theta_acc_precond=theta_acc,
        S_acc_precond=S_acc,
        posterior_flow=posterior.flow,
        x_obs=x_obs,
        s_obs=s_obs,
        S_mean=posterior.S_mean,
        S_std=posterior.S_std,
        th_mean_post=posterior.th_mean,
        th_std_post=posterior.th_std,
        posterior_samples_at_obs=th_samps,
        loss_history=posterior.losses,
    )

    if run.outdir:
        save_artifacts(
            outdir=run.outdir,
            spec={
                "name": spec.name,
                "theta_dim": spec.theta_dim,
                "s_dim": spec.s_dim,
                "theta_labels": list(spec.theta_labels) if spec.theta_labels else None,
            },
            run_cfg=asdict(run),
            flow_cfg=asdict(flow_cfg),
            posterior_flow=result.posterior_flow,
            s_obs=result.s_obs,
            posterior_samples=result.posterior_samples_at_obs,
            theta_acc=result.theta_acc_precond,
            S_acc=result.S_acc_precond,
            S_mean=result.S_mean,
            S_std=result.S_std,
            th_mean=result.th_mean_post,
            th_std=result.th_std_post,
            loss_history=result.loss_history,
            theta_labels=list(spec.theta_labels) if spec.theta_labels else None,
        )

    return result

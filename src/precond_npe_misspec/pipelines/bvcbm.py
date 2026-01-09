# src/precond_npe_misspec/pipelines/bvcbm.py
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from jax import ShapeDtypeStruct

from precond_npe_misspec.engine.run import (
    NpeRsConfig,
    PosteriorConfig,
    PrecondConfig,
    RobustConfig,
    RunConfig,
    run_experiment,
)
from precond_npe_misspec.engine.spec import (
    ExperimentSpec,
    FlowConfig,
    default_posterior_flow_builder,
)
from precond_npe_misspec.examples import bvcbm as ex
from precond_npe_misspec.examples.embeddings import build as get_embedder
from precond_npe_misspec.utils.debug_paths import ensure_debug_outdir


def _uniform_logpdf_box(
    theta: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray
) -> jnp.ndarray:
    th = jnp.asarray(theta)
    inside = jnp.all((th >= lo) & (th <= hi))
    return jnp.where(inside, 0.0, -jnp.inf)


# ---------- Public config ----------
@dataclass
class Config:
    # Data + model
    seed: int = 0
    obs_seed: int = 1234
    outdir: str | None = None
    T: int = 19
    start_volume: float = 50.0
    page: int = 5
    dataset: Literal["breast", "pancreatic"] = "pancreatic"
    patient_idx: int = 0  # 0..3

    # 3-parameter biphasic: (gage1[h], tau[days], gage2[h])
    theta_true: tuple[float, float, float] = (200.0, 12.0, 50.0)

    obs_model: Literal["synthetic", "real"] = "real"
    summary: Literal["identity", "log"] = "identity"
    embedder: str = "asv_tcn"

    # Engines
    precond: PrecondConfig = PrecondConfig()
    posterior: PosteriorConfig = PosteriorConfig()
    robust: RobustConfig = RobustConfig()
    flow: FlowConfig = FlowConfig()
    npers: NpeRsConfig = NpeRsConfig()


# ---------- Spec builder ----------
def _make_spec(cfg: Config, y_obs: jnp.ndarray | None, T_sim: int) -> ExperimentSpec:
    theta_dim = 3

    # --- host simulator via callback (JAX-safe) ---
    T_fixed = int(T_sim)

    class _SimFn(Protocol):
        def __call__(self, theta: np.ndarray, seed: int) -> np.ndarray: ...

    sim_py: _SimFn = cast(
        _SimFn,
        ex.simulator_biphasic(T=T_fixed, start_volume=cfg.start_volume, page=cfg.page),
    )

    _N_WORKERS = int(os.getenv("BVCBM_WORKERS", "1"))
    _POOL = os.getenv("BVCBM_POOL", "process")  # "process" | "thread"
    _START = os.getenv("BVCBM_START_METHOD", "spawn")  # spawn|forkserver|fork

    _EXEC: ProcessPoolExecutor | ThreadPoolExecutor | None = None

    if _N_WORKERS > 1 and _POOL == "process":
        ctx = mp.get_context(_START)
        _EXEC = ProcessPoolExecutor(
            max_workers=_N_WORKERS,
            mp_context=ctx,
            initializer=ex._init_bvcbm_worker,
            initargs=(T_fixed, cfg.start_volume, cfg.page),
        )
        _WORKER = ex.simulate_worker  # (theta, seed) -> (T,)
    elif _N_WORKERS > 1 and _POOL == "thread":
        _EXEC = ThreadPoolExecutor(max_workers=_N_WORKERS)

        def _WORKER(theta: np.ndarray, seed: int) -> np.ndarray:
            return sim_py(theta, int(seed))

    else:
        _EXEC = None

        def _WORKER(theta: np.ndarray, seed: int) -> np.ndarray:
            return sim_py(theta, int(seed))

    def _simulate_np(theta_np: np.ndarray, seed_np: np.ndarray) -> np.ndarray:
        th = np.asarray(theta_np, dtype=float)
        sd = np.asarray(seed_np, dtype=np.uint32)
        # unbatched -------------------------------------------------------------
        if th.ndim == 1:
            if _EXEC is None:
                y = sim_py(th, int(sd))
            elif _POOL == "process":
                y = _EXEC.submit(_WORKER, th, int(sd)).result()
            else:  # thread pool
                y = _WORKER(th, int(sd))
            return np.asarray(y, dtype=np.float32)  # (T_fixed,)
        # batched ---------------------------------------------------------------
        B = th.shape[0]
        seeds = sd.reshape(B).astype(np.uint32)
        if _EXEC is None:
            ys = [_WORKER(th[i], int(seeds[i])) for i in range(B)]
        elif _POOL == "process":
            ys = list(_EXEC.map(_WORKER, list(th), list(seeds.astype(int))))
        else:
            ys = list(_EXEC.map(_WORKER, list(th), list(seeds.astype(int))))
        return np.asarray(ys, dtype=np.float32)  # (B, T_fixed)

    def simulate(key: jax.Array, theta: jax.Array, **kw: Any) -> jax.Array:
        seed = jax.random.randint(key, (), 0, 2**31 - 1, dtype=jnp.uint32)
        # shape-aware result spec: (T,) if unbatched, else (B, T)
        is_batched = len(theta.shape) == 2
        shape: tuple[int, ...] = (
            (int(theta.shape[0]), T_fixed) if is_batched else (T_fixed,)
        )
        out_shape: ShapeDtypeStruct = ShapeDtypeStruct(shape, jnp.float32)
        result: jax.Array = cast(
            jax.Array,
            jax.pure_callback(
                _simulate_np, out_shape, theta, seed, vmap_method="broadcast_all"
            ),
        )
        return result

    # Summaries
    if cfg.summary == "log":

        def summaries(x: jax.Array | np.ndarray) -> jax.Array:
            return jnp.asarray(ex.summary_log(jnp.asarray(x)))

    else:

        def summaries(x: jax.Array | np.ndarray) -> jax.Array:
            return jnp.asarray(ex.summary_identity(jnp.asarray(x)))

    # Probe to infer summary dimension
    x_probe = jnp.asarray(
        sim_py(np.asarray(cfg.theta_true, float), seed=0), jnp.float32
    )
    s_dim = int(summaries(x_probe).shape[-1])

    # Priors (Uniforms; g_age in hours, τ in days)
    lo, hi = ex.theta_bounds_biphasic(T_sim)
    prior = ex.prior_biphasic(T_sim)

    def prior_sample(key: jax.Array) -> jax.Array:
        return cast(jax.Array, prior.sample(key))

    def prior_logpdf(th: jax.Array) -> jax.Array:  # or _uniform_logpdf_box(th, lo, hi)
        return _uniform_logpdf_box(th, lo, hi)

    # Observed DGP
    if cfg.obs_model == "synthetic":

        def true_dgp(key: jax.Array, theta: jax.Array, **kw: Any) -> jax.Array:
            return simulate(key, theta, T=T_sim)

    else:
        assert y_obs is not None, "y_obs must be provided when obs_model='real'"

        def true_dgp(key: jax.Array, theta: jax.Array, **kw: Any) -> jax.Array:
            del theta  # avoid unused-arg warnings
            return y_obs

    return ExperimentSpec(
        name="bvcbm_biphasic",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=prior_sample,
        prior_logpdf=prior_logpdf,
        true_dgp=true_dgp,
        simulate=simulate,
        summaries=summaries,
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
        build_embedder=get_embedder(cfg.embedder),
        theta_labels=(
            r"$g_{\mathrm{age}}^{(1)}\,[\mathrm{h}]$",
            r"$\tau\,[\mathrm{days}]$",
            r"$g_{\mathrm{age}}^{(2)}\,[\mathrm{h}]$",
        ),
        summary_labels=tuple(
            f"{'logV' if cfg.summary == 'log' else 'V'}[t={t}]" for t in range(s_dim)
        ),
        theta_lo=lo,
        theta_hi=hi,
        simulate_path="precond_npe_misspec.examples.bvcbm:simulate_biphasic",  # adaptor to add in examples
        summaries_path="precond_npe_misspec.examples.bvcbm:summary_identity",
    )


# ---------- CLI ----------
def _load_real_observations(T: int, *, dataset: str, patient_idx: int) -> jnp.ndarray:
    from scipy.io import loadmat

    mat_path = os.getenv("CANCER_DATASETS_MAT")
    if mat_path:
        data_path = Path(mat_path)
    else:
        data_path = (
            Path(__file__).resolve().parents[2]
            / "precond_npe_misspec"
            / "data"
            / "CancerDatasets.mat"
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing real data file at {data_path}")

    key = {"breast": "Breast_data", "pancreatic": "Pancreatic_data"}[dataset.lower()]
    mat = loadmat(data_path, squeeze_me=True)
    arr = np.asarray(mat.get(key), dtype=float)
    if arr is None:
        raise KeyError(f"'{key}' missing from {data_path.name}")
    if arr.ndim < 2 or patient_idx >= arr.shape[1]:
        raise ValueError(
            f"Unexpected shape for {key}; need ≥4 columns, got {arr.shape}"
        )
    if T > arr.shape[0]:
        raise ValueError(f"T={T} exceeds available samples {arr.shape[0]}")
    series = np.squeeze(arr[:T, patient_idx])
    return jnp.asarray(series, dtype=jnp.float32)


def main(cfg: Config) -> None:
    if cfg.outdir is None:
        cfg.outdir = ensure_debug_outdir("bvcbm", seed=cfg.seed)
        print(f"[debug] saving artefacts to {cfg.outdir}")
    if cfg.obs_model == "real":
        y_obs = _load_real_observations(
            cfg.T, dataset=cfg.dataset, patient_idx=cfg.patient_idx
        )
        T_sim = int(y_obs.shape[0])
    else:
        y_obs = None
        T_sim = cfg.T

    spec = _make_spec(cfg, y_obs, T_sim)
    run_experiment(
        spec,
        RunConfig(
            seed=cfg.seed,
            obs_seed=cfg.obs_seed,
            theta_true=cfg.theta_true,
            sim_kwargs={"T": T_sim},
            summaries_kwargs={},
            outdir=cfg.outdir,
            precond=cfg.precond,
            posterior=cfg.posterior,
            robust=cfg.robust,
            batch_size=cfg.flow.batch_size,
        ),
        cfg.flow,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))

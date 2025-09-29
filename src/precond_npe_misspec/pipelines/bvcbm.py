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

from precond_npe_misspec.engine.run import (NpeRsConfig, PosteriorConfig,
                                            PrecondConfig, RobustConfig,
                                            RunConfig, run_experiment)
from precond_npe_misspec.examples import bvcbm as ex
from precond_npe_misspec.examples.embeddings import build as get_embedder
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec, FlowConfig, default_posterior_flow_builder)

# def _uniform_logpdf_box(
#     theta: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray
# ) -> jnp.ndarray:
#     th = jnp.asarray(theta)
#     inside = jnp.all((th >= lo) & (th <= hi))
#     return jnp.where(inside, 0.0, -jnp.inf)


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

    # (p0_1, psc_1, dmax_1, gage_1[h], p0_2, psc_2, dmax_2, gage_2[h], tau[days])
    theta_true: tuple[float, float, float, float, float, float, float, float, float] = (
        0.05,
        0.01,
        30.0,
        48.0,
        0.04,
        0.008,
        30.0,
        48.0,
        7.0,
    )

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
    theta_dim = 9

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
        out_shape: ShapeDtypeStruct = ShapeDtypeStruct(  # type: ignore[no-untyped-call]
            shape, jnp.float32
        )
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
        return cast(jax.Array, prior.log_prob(th))

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
            r"$p_0^{(1)}$",
            r"$p_{\mathrm{psc}}^{(1)}$",
            r"$d_{\max}^{(1)}$",
            r"$g_{\mathrm{age}}^{(1)}\,[\mathrm{h}]$",
            r"$p_0^{(2)}$",
            r"$p_{\mathrm{psc}}^{(2)}$",
            r"$d_{\max}^{(2)}$",
            r"$g_{\mathrm{age}}^{(2)}\,[\mathrm{h}]$",
            r"$\tau\,[\mathrm{days}]$",
        ),
        summary_labels=tuple(
            f"{'logV' if cfg.summary == 'log' else 'V'}[t={t}]" for t in range(s_dim)
        ),
        theta_lo=lo,
        theta_hi=hi,
        simulate_path="precond_npe_misspec.examples.bvcbm:simulate_biphasic",  # adaptor to add in examples
        summaries_path="precond_npe_misspec.examples.bvcbm:summary_log",
    )


# ---------- CLI ----------
def _load_real_observations(T: int) -> jnp.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("scipy is required for obs_model='real'") from exc

    data_path = (
        Path(__file__).resolve().parents[2]
        / "precond_npe_misspec"
        / "data"
        / "CancerDatasets.mat"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing real data file at {data_path}")

    mat = loadmat(data_path, squeeze_me=True)
    breast_data = mat.get("Breast_data")
    if breast_data is None:
        raise KeyError("'Breast_data' missing from CancerDatasets.mat")

    control_growth = np.asarray(breast_data, dtype=float)
    if control_growth.shape[0] < T:
        raise ValueError(
            f"Requested T={T} exceeds available samples {control_growth.shape[0]}"
        )
    if control_growth.ndim < 2 or control_growth.shape[1] <= 3:
        raise ValueError(
            "Unexpected shape for Breast_data; expected at least four columns"
        )

    series = np.squeeze(control_growth[:T, 3])
    return jnp.asarray(series, dtype=jnp.float32)


def main(cfg: Config) -> None:
    if cfg.obs_model == "real":
        y_obs = _load_real_observations(cfg.T)
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

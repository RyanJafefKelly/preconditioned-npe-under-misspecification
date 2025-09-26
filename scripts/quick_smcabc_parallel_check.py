#!/usr/bin/env python3
import os, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ.setdefault("BVCBM_POOL", "process")

import jax, jax.numpy as jnp
from precond_npe_misspec.pipelines.bvcbm import _make_spec, Config
from precond_npe_misspec.algorithms.smc_abc import run_smc_abc


def bench(workers: int, N: int, T: int, seed: int = 0) -> float:
    os.environ["BVCBM_WORKERS"] = str(workers)
    cfg = Config(T=T, seed=seed, obs_seed=seed + 1)
    spec = _make_spec(cfg, None, T)
    theta_true = jnp.asarray(cfg.theta_true, jnp.float32)
    y_obs = spec.true_dgp(jax.random.key(cfg.obs_seed), theta_true, T=T)
    s_obs = spec.summaries(y_obs)
    # warm-up (compile + pool spin-up; keep R≈1 to avoid recompiles)
    _ = run_smc_abc(
        key=jax.random.key(seed),
        n_particles=N,
        epsilon0=1e6,
        alpha=0.5,
        eps_min=0.0,
        acc_min=0.0,
        max_iters=1,
        initial_R=1,
        c_tuning=0.9,
        B_sim=1,
        spec=spec,
        s_obs=s_obs,
        sim_kwargs={"T": T},
        S_pilot_for_distance=None,
    )
    jax.block_until_ready(_[0])
    # timed run
    t0 = time.time()
    out = run_smc_abc(
        key=jax.random.key(seed + 123),
        n_particles=N,
        epsilon0=1e6,
        alpha=0.5,
        eps_min=0.0,
        acc_min=0.0,
        max_iters=2,
        initial_R=1,
        c_tuning=0.9,
        B_sim=1,
        spec=spec,
        s_obs=s_obs,
        sim_kwargs={"T": T},
        S_pilot_for_distance=None,
    )
    jax.block_until_ready(out[0])
    return time.time() - t0


if __name__ == "__main__":
    N, T = 32, 19
    t1 = bench(1, N, T)
    print(f"SMCABC workers=1  time={t1:.2f}s")
    t2 = bench(2, N, T)
    print(f"SMCABC workers=2  time={t2:.2f}s  speedup={t1/t2:.2f}×")
    t4 = bench(4, N, T)
    print(f"SMCABC workers=4  time={t4:.2f}s  speedup={t1/t4:.2f}×")

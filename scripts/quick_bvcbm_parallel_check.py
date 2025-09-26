#!/usr/bin/env python3
import os, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ.setdefault("BVCBM_POOL", "process")  # force processes

import jax, jax.numpy as jnp
from precond_npe_misspec.pipelines.bvcbm import _make_spec, Config


def timed_run(workers: int, B: int, T: int, seed: int = 0) -> float:
    os.environ["BVCBM_WORKERS"] = str(workers)
    spec = _make_spec(Config(T=T), None, T)
    key = jax.random.key(seed)
    keys = jax.random.split(key, B)
    thetas = jax.vmap(lambda k: spec.prior_sample(k))(keys)

    @jax.jit
    def sim_batch(kk, th):
        return jax.vmap(lambda k, t: spec.simulate(k, t, T=T))(kk, th)

    sim_batch(keys, thetas).block_until_ready()  # compile + warm
    t0 = time.time()
    sim_batch(keys, thetas).block_until_ready()
    return time.time() - t0


if __name__ == "__main__":
    B, T = 16, 19  # bump B to amortise pool overhead
    t1 = timed_run(1, B, T)
    print(f"workers=1  time={t1:.2f}s")
    t2 = timed_run(2, B, T)
    print(f"workers=2  time={t2:.2f}s  speedup={t1/t2:.2f}×")
    t4 = timed_run(4, B, T)
    print(f"workers=4  time={t4:.2f}s  speedup={t1/t4:.2f}×")

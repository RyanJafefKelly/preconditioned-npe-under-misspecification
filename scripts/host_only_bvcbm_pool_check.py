#!/usr/bin/env python3
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from precond_npe_misspec.examples import bvcbm as ex


def run(workers: int, B: int, T: int):
    with ProcessPoolExecutor(max_workers=workers, initializer=ex._init_bvcbm_worker, initargs=(T, 50.0, 5)) as pool:
        thetas = np.random.rand(B, 9).astype(float)
        seeds = np.arange(B, dtype=int)
        t0 = time.time()
        list(pool.map(ex.simulate_worker, list(thetas), list(seeds)))
        return time.time() - t0


if __name__ == "__main__":
    B, T = 128, 19
    for w in [1, 2, 4]:
        print(f"workers={w} time={run(w, B, T):.2f}s")

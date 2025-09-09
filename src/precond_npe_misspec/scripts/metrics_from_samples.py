"""Run after attaining posterior samples, calculate coverage and more."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import tyro

from precond_npe_misspec.utils.metrics import compute_rep_metrics


@dataclass
class Args:
    outdir: str
    theta_target: tuple[float, float, float, float]  # true or pseudo-true
    level: float = 0.95
    want_hpdi: bool = True
    want_central: bool = True
    samples_file: str | None = None  # override; else auto-detect
    method: str | None = None  # optional label in metrics


def _load_samples(outdir: Path, override: str | None) -> np.ndarray:
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(p)
        with np.load(p) as z:
            return z["samples"]
    # prefer robust, else standard
    for name in ("posterior_samples_robust.npz", "posterior_samples.npz"):
        p = outdir / name
        if p.is_file():
            with np.load(p) as z:
                return z["samples"]
    raise FileNotFoundError("No posterior samples found in " + str(outdir))


def main(a: Args) -> None:
    outdir = Path(a.outdir)
    samples = _load_samples(outdir, a.samples_file)
    th = jnp.asarray(a.theta_target, dtype=samples.dtype)

    m = compute_rep_metrics(
        posterior_samples=jnp.asarray(samples),
        theta_target=th,
        level=a.level,
        want_central=a.want_central,
        want_hpdi=a.want_hpdi,
    )
    # attach minimal metadata
    m["theta_target"] = list(map(float, a.theta_target))
    m["method"] = a.method
    m["outdir"] = str(outdir)

    # try to add seeds and config if present
    cfg_path = outdir / "config.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            m["run_cfg"] = cfg.get("run", {})
            m["spec"] = cfg.get("spec", {})
        except Exception:
            pass

    with open(outdir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

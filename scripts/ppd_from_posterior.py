from __future__ import annotations

import importlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


def _first_array_from_npz(p: Path) -> np.ndarray:
    with np.load(p, allow_pickle=False) as z:
        # prefer common keys
        for k in ("samples", "posterior_samples", "theta", "thetas"):
            if k in z and z[k].ndim == 2 and z[k].shape[1] == 3:
                return z[k]
        # else best-effort: first (N,3)
        for k in z.files:
            arr = z[k]
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
    raise FileNotFoundError(f"no (N,3) array in {p}")


def _load_samples(run_dir: Path) -> tuple[np.ndarray, Path]:
    cands = [
        "posterior_samples_robust.npz",
        "robust_posterior_samples.npz",
        "posterior_samples.npz",
        "posterior_samples.npy",
    ]
    for name in cands:
        p = run_dir / name
        if p.exists():
            if p.suffix == ".npz":
                return _first_array_from_npz(p), p
            arr = np.load(p)
            assert arr.ndim == 2 and arr.shape[1] == 3
            return arr, p
    raise FileNotFoundError(f"no posterior samples found in {run_dir}")


def _import_by_path(path: str):
    mod, func = path.split(":")
    m = importlib.import_module(mod)
    return getattr(m, func), m


def _pick_latest_run(method_dir: Path, group_pred, required_dim: int = 3) -> Path:
    """Return the newest run that has posterior samples with shape (N, required_dim)."""
    candidates: list[Path] = []
    groups = [g for g in method_dir.iterdir() if g.is_dir() and group_pred(g.name)]
    if not groups:
        raise FileNotFoundError("no matching groups")
    for g in groups:
        seed_dirs = sorted([d for d in g.iterdir() if d.is_dir() and d.name.startswith("seed-")])
        if not seed_dirs:
            continue
        s = next((d for d in seed_dirs if d.name == "seed-0"), seed_dirs[0])
        for r in sorted([r for r in s.iterdir() if r.is_dir()]):
            try:
                arr, _ = _load_samples(r)
                if arr.ndim == 2 and arr.shape[1] == required_dim:
                    candidates.append(r)
            except Exception:
                continue
    if not candidates:
        raise FileNotFoundError(f"no runs with posterior samples of shape (N,{required_dim})")
    return sorted(candidates)[-1]


@dataclass
class Args:
    results_root: str = "results/bvcbm"
    method: str = "rnpe"
    dataset: str = "pancreatic"
    patient_idx: int = 0
    T: int = 32
    page: int = 5
    summary: str = "identity"  # or "log"
    obs_model: str = "real"
    M: int = 1000
    seed: int = 0
    out_override: str | None = None


def main(a: Args) -> None:
    # handle rf_abc_rnpe alias
    method_choices = [a.method]
    if a.method == "rf_abc_rnpe":
        method_choices.append("rf_abc_pnpe")
    run_dir = None
    for m in method_choices:
        base = Path(a.results_root) / m
        if not base.exists():
            continue

        def pred(name: str) -> bool:
            want = f"ds_{a.dataset}-p{a.patient_idx}-T_{a.T}-"
            return (
                name.startswith(want)
                and f"-sum_{a.summary}-" in name
                and f"-page_{a.page}-" in name
                and f"-obs_{a.obs_model}-" in name
            )

        try:
            run_dir = _pick_latest_run(base, pred, required_dim=3)

            break
        except FileNotFoundError:
            continue
    if run_dir is None:
        raise FileNotFoundError(
            f"no run found for args: method={a.method} dataset={a.dataset} p={a.patient_idx} "
            f"T={a.T} page={a.page} summary={a.summary} obs={a.obs_model} "
            f"under {a.results_root}/{a.method} (need posterior_samples with shape (N,3))"
        )

    thetas, src = _load_samples(run_dir)
    M = min(a.M, thetas.shape[0])

    ep = json.loads((run_dir / "entrypoints.json").read_text())
    sim_path: str = ep["simulate"]
    sim_kwargs: dict = ep["sim_kwargs"]
    sim_fn_adaptor, module = _import_by_path(sim_path)
    # Prefer host simulator if available
    sim_host = getattr(module, "simulator_biphasic", None)
    if sim_host is not None:
        sim = sim_host(**sim_kwargs)

        def simulate_one(th: Iterable[float], sd: int) -> np.ndarray:
            return sim(th, sd)

    else:
        import jax
        import jax.numpy as jnp

        def simulate_one(th: Iterable[float], sd: int) -> np.ndarray:
            key = jax.random.PRNGKey(sd)
            y = sim_fn_adaptor(key, jnp.asarray(th, jnp.float32), **sim_kwargs)
            return np.asarray(y, np.float32)

    rng = np.random.default_rng(a.seed)
    idx = rng.choice(thetas.shape[0], size=M, replace=False)
    seeds = rng.integers(0, 2**31 - 1, size=M, dtype=np.uint32)
    Y = np.stack([simulate_one(thetas[i], int(seeds[i])) for i in range(M)], axis=0).astype(np.float32)

    # Optional summaries
    S = None
    if "summaries" in ep and ep["summaries"]:
        sum_fn, _ = _import_by_path(ep["summaries"])
        try:
            import jax.numpy as jnp

            S = np.stack([np.asarray(sum_fn(Y[i]), np.float32) for i in range(M)], 0)
        except Exception:
            pass

    outdir = Path(a.out_override) if a.out_override else run_dir
    np.save(outdir / "ppd_y.npy", Y)
    np.save(outdir / "ppd_theta_idx.npy", idx)
    if S is not None:
        np.save(outdir / "ppd_S.npy", S)
    q05, q50, q95 = np.quantile(Y, [0.05, 0.5, 0.95], axis=0)
    np.savez(outdir / "ppd_quants.npz", q05=q05, q50=q50, q95=q95)
    (outdir / "ppd_info.json").write_text(
        json.dumps(
            {
                "M": int(M),
                "seed": int(a.seed),
                "src_samples": str(src.name),
                "sim_path": sim_path,
                "sim_kwargs": sim_kwargs,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )
    print(f"[ppd] wrote {M} draws to {outdir}")


if __name__ == "__main__":
    main(tyro.cli(Args))

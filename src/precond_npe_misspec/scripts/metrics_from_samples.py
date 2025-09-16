"""Compute posterior metrics from saved artefacts: coverage, logpdf at θ†, PPD."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from precond_npe_misspec.pipelines.base_pnpe import (
    FlowConfig,
    default_posterior_flow_builder,
)
from precond_npe_misspec.utils.metrics import (
    compute_rep_metrics,
    posterior_predictive_distance_on_summaries,
)


@dataclass
class Args:
    outdir: str
    theta_target: list[float]
    level: float = 0.95
    want_hpdi: bool = True
    want_central: bool = True
    samples_file: str | None = None
    method: str | None = None

    # Neural-posterior log density
    compute_logpdf: bool = True
    logpdf_batch: int = 8192

    # Posterior predictive distance on summaries
    compute_ppd: bool = False
    ppd_n: int = 1000
    ppd_metric: str = "l2"  # "l2" | "l1"
    ppd_rng_seed: int = 0
    n_obs_override: int | None = None
    ppd_entrypoints: str | None = None  # JSON with {"simulate": "mod:fn", "summaries": "mod:fn", "sim_kwargs": {...}}
    ppd_simulate: str | None = None  # dotted path as fallback
    ppd_summaries: str | None = None  # dotted path as fallback


# ---------- helpers ----------
def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        return cast(dict[str, Any], json.loads(p.read_text()))
    except Exception:
        return None


def _load_samples(outdir: Path, override: str | None) -> np.ndarray:
    if override:
        with np.load(override) as z:
            return cast(np.ndarray, z["samples"])
    for name in ("posterior_samples_robust.npz", "posterior_samples.npz"):
        p = outdir / name
        if p.is_file():
            with np.load(p) as z:
                return cast(np.ndarray, z["samples"])
    raise FileNotFoundError(f"No posterior samples in {outdir}")


def _load_array_stem(outdir: Path, stem: str) -> jnp.ndarray | None:
    for ext in (".npy", ".npz"):
        p = outdir / f"{stem}{ext}"
        if p.is_file():
            try:
                if ext == ".npy":
                    return jnp.asarray(np.load(p))
                with np.load(p) as z:
                    k = "arr_0" if "arr_0" in z else next(iter(z.keys()))
                    return jnp.asarray(z[k])
            except Exception:
                pass
    return None


def _load_standardisers(outdir: Path) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    # Prefer explicit S_mean/S_std; fallback to standardisers.npz with keys
    S_mean = _load_array_stem(outdir, "S_mean")
    S_std = _load_array_stem(outdir, "S_std")
    if S_mean is not None and S_std is not None:
        return S_mean, S_std
    p = outdir / "standardisers.npz"
    if p.is_file():
        try:
            with np.load(p) as z:
                Sm = z.get("S_mean")
                Ss = z.get("S_std")
                if Sm is not None and Ss is not None:
                    return jnp.asarray(Sm), jnp.asarray(Ss)
        except Exception:
            pass
    return None, None


def _shape_to_vec(x: jnp.ndarray | None, s_dim: int) -> jnp.ndarray | None:
    """Coerce x to shape (s_dim,) if possible; else return None."""
    if x is None:
        return None
    x = jnp.asarray(x)
    if x.ndim == 1 and x.size == s_dim:
        return x
    if x.ndim == 2:
        # common cases: (R, s_dim) or (s_dim, R) → average over R
        if x.shape[1] == s_dim:
            return jnp.mean(x, axis=0)
        if x.shape[0] == s_dim:
            return jnp.mean(x, axis=1)
    return None


def _load_context(outdir: Path, cfg: dict[str, Any]) -> jnp.ndarray:
    """Prefer denoised context if available; fallback to standardised s_obs."""
    spec = cfg.get("spec", {}) or {}
    s_dim = int(spec["s_dim"])
    # 1) robust path
    s_dn = _load_array_stem(outdir, "s_obs_denoised_w")
    s_ctx = _shape_to_vec(s_dn, s_dim)
    if s_ctx is not None:
        return s_ctx
    # 2) standardised s_obs
    s_obs = _load_array_stem(outdir, "s_obs")
    S_mean, S_std = _load_standardisers(outdir)
    if s_obs is None or S_mean is None or S_std is None:
        raise FileNotFoundError("Missing s_obs and/or standardisers")
    s_obs_w = (s_obs - S_mean) / (S_std + 1e-8)
    s_ctx = _shape_to_vec(s_obs_w, s_dim)
    if s_ctx is not None:
        return s_ctx
    raise ValueError(f"Context shape incompatible with s_dim={s_dim}")


def _load_callable(spec: str) -> Callable[..., Any]:
    mod, fn = spec.split(":", 1) if ":" in spec else spec.rsplit(".", 1)
    return cast(Callable[..., Any], getattr(import_module(mod), fn))


def _logprob_conditional(q: Any, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
    try:
        return cast(jnp.ndarray, q.log_prob(x, context=context))
    except TypeError:
        return cast(jnp.ndarray, q.log_prob(x, context))


def _batched_logprob(q: Any, x: jnp.ndarray, context: jnp.ndarray, batch: int) -> jnp.ndarray:
    n = x.shape[0]
    if n <= batch:
        return _logprob_conditional(q, x, context)
    parts: list[jnp.ndarray] = []
    for i in range(0, n, batch):
        parts.append(_logprob_conditional(q, x[i : i + batch], context))
    return jnp.concatenate(parts, axis=0)


# ---------- main ----------
def main(a: Args) -> None:
    outdir = Path(a.outdir)
    samples = jnp.asarray(_load_samples(outdir, a.samples_file))
    theta_dag = jnp.asarray(a.theta_target, dtype=samples.dtype)
    cfg = _load_json(outdir / "config.json") or {}

    m = compute_rep_metrics(
        posterior_samples=samples,
        theta_target=theta_dag,
        level=a.level,
        want_central=a.want_central,
        want_hpdi=a.want_hpdi,
    )
    m.update(
        {
            "theta_target": [float(x) for x in a.theta_target],
            "method": a.method,
            "outdir": str(outdir),
        }
    )
    if "spec" in cfg:
        m["spec"] = cfg["spec"]
    run_cfg = cfg.get("run") or {}
    if run_cfg:
        m["run_cfg"] = run_cfg

    # Log density at θ† under q(θ|s_obs)
    if a.compute_logpdf:
        try:
            spec = cfg.get("spec", {}) or {}
            theta_dim, s_dim = int(spec["theta_dim"]), int(spec["s_dim"])
            s_ctx = _load_context(outdir, cfg)
            flow_cfg_dict = cfg.get("flow_cfg", {}) or {}
            q_blank = default_posterior_flow_builder(theta_dim, s_dim)(jax.random.key(0), FlowConfig(**flow_cfg_dict))
            eqx_path = outdir / "posterior_flow.eqx"
            if not eqx_path.is_file():
                raise FileNotFoundError("posterior_flow.eqx not found")
            q_theta_s = eqx.tree_deserialise_leaves(eqx_path, q_blank)

            # 1) choose context s_ctx (RNPE: denoised-whitened if available)
            s_ctx = _load_array_stem(outdir, "s_obs_denoised_w")
            if s_ctx is None:
                den = _load_array_stem(outdir, "denoised_s_samples")  # (M, d) optional
                if den is not None and den.ndim == 2:
                    s_ctx = jnp.mean(den, axis=0)
            if s_ctx is None:
                s_obs = _load_array_stem(outdir, "s_obs")
                S_mean, S_std = _load_standardisers(outdir)
                if s_obs is None or S_mean is None or S_std is None:
                    raise FileNotFoundError("Missing s_obs and/or standardisers")
                s_ctx = (s_obs - S_mean) / (S_std + 1e-8)
            # 2) compute log-probs regardless of source of s_ctx
            lp_samples = _batched_logprob(q_theta_s, samples, s_ctx, a.logpdf_batch)
            lp_theta = _logprob_conditional(q_theta_s, theta_dag[None, :], s_ctx)[0]

            # lp_samples = _batched_logprob(q_theta_s, samples, s_obs_w, a.logpdf_batch)
            # lp_theta = _logprob_conditional(q_theta_s, theta_dag[None, :], s_obs_w)[0]
            q05, q50, q95 = np.quantile(np.asarray(lp_samples), [0.05, 0.5, 0.95])
            m["post_logpdf_at_theta"] = float(lp_theta)
            m["post_logpdf_quantiles"] = {
                "q05": float(q05),
                "q50": float(q50),
                "q95": float(q95),
            }
            alpha = 1.0 - float(a.level)
            thr = np.quantile(np.asarray(lp_samples), alpha)
            m["joint_hpd_hit"] = bool(float(lp_theta) >= float(thr))
        except Exception as e:
            m["post_logpdf_error"] = f"{type(e).__name__}: {e}"

    # Posterior predictive distance on summaries
    if a.compute_ppd:
        try:
            # resolve entrypoints + kwargs
            ep = None
            if a.ppd_entrypoints:
                ep = _load_json(Path(a.ppd_entrypoints))
            if ep is None:
                ep = _load_json(outdir / "entrypoints.json")
            simulate_path = a.ppd_simulate or (ep or {}).get("simulate")
            summaries_path = a.ppd_summaries or (ep or {}).get("summaries")
            sim_kwargs = dict(run_cfg.get("sim_kwargs") or {})
            sim_kwargs.update((ep or {}).get("sim_kwargs") or {})
            summary_kwargs = (ep or {}).get("summaries_kwargs") or {}

            if not simulate_path or not summaries_path:
                raise FileNotFoundError("PPD simulate/summaries entrypoints not found")
            simulate = _load_callable(simulate_path)
            summaries = _load_callable(summaries_path)

            def _sim_wrap(k: jax.Array, th: jax.Array, _n: int) -> jax.Array:  # ignore third arg
                return cast(jax.Array, simulate(k, th, **sim_kwargs))

            def _sum_wrap(x: jax.Array) -> jax.Array:
                return cast(jax.Array, summaries(x, **summary_kwargs))

            s_obs = _load_array_stem(outdir, "s_obs")
            if s_obs is None:
                raise FileNotFoundError("s_obs not found")
            if a.n_obs_override is not None:
                sim_kwargs["n_obs"] = int(a.n_obs_override)
            # avoid n_obs=0 → NaNs in some metrics
            n_obs_val = int(sim_kwargs.get("n_obs", 0))
            if n_obs_val <= 0:
                n_obs_val = 1

            key = jax.random.key(a.ppd_rng_seed)
            ppd = posterior_predictive_distance_on_summaries(
                key=key,
                theta_samples=samples,
                simulate=_sim_wrap,
                summaries=_sum_wrap,
                s_obs=s_obs,
                n_obs=n_obs_val,
                n_rep=int(a.ppd_n),
                metric=("l2" if a.ppd_metric == "l2" else "l1"),
            )
            # Drop non-finite distances defensively
            for k in ("ppd_mean", "ppd_sd", "ppd_q50", "ppd_q90"):
                if not np.isfinite(ppd[k]):
                    ppd[k] = float("nan")

            ppd["ppd_q50"] = float(ppd.get("ppd_q50", np.nan))
            m.update(ppd)
        except Exception as e:
            m["ppd_error"] = f"{type(e).__name__}: {e}"

    (outdir / "metrics.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

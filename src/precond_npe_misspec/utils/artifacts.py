from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, jnp.ndarray):
        return np.asarray(jax.device_get(x))
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, jnp.ndarray | np.ndarray):
        return _to_np(obj).tolist()
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _plot_marginals(
    samples: np.ndarray,
    labels: Sequence[str],
    path: Path,
    dpi: int,
    ref: np.ndarray | None = None,
) -> None:
    if plt is None:
        return
    D = samples.shape[1]
    fig, axes = plt.subplots(1, D, figsize=(2.6 * D, 2.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for i in range(D):
        ax = axes[i]
        ax.hist(samples[:, i], bins=60, density=True)
        ax.set_xlabel(labels[i])
        if ref is not None and i < ref.shape[0]:
            rv = float(ref[i])
            ax.axvline(rv, color="red", linestyle="--", linewidth=1.0)
            # Add a red 'x' at y=0 for the marginal
            ax.scatter([rv], [0.0], color="red", marker="x", zorder=5)
        if i:
            ax.set_yticks([])
        else:
            ax.set_ylabel("density")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_pairgrid(
    samples: np.ndarray,
    labels: Sequence[str],
    path: Path,
    dpi: int,
    ref: np.ndarray | None = None,
) -> None:
    if plt is None:
        return
    D = samples.shape[1]
    if D < 2:
        return
    fig, axes = plt.subplots(D, D, figsize=(2.2 * D, 2.2 * D), constrained_layout=True)
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, j], bins=60, density=True)
                if ref is not None and j < ref.shape[0]:
                    rv = float(ref[j])
                    ax.axvline(rv, color="red", linestyle="--", linewidth=1.0)
                    # Optional: also mark the baseline for density
                    ax.scatter([rv], [0.0], color="red", marker="x", zorder=5)
            elif i > j:
                ax.hist2d(samples[:, j], samples[:, i], bins=80)
                if ref is not None and i < ref.shape[0] and j < ref.shape[0]:
                    rj = float(ref[j])
                    ri = float(ref[i])
                    ax.axvline(rj, color="red", linestyle="--", linewidth=1.0)
                    ax.axhline(ri, color="red", linestyle="--", linewidth=1.0)
                    ax.scatter([rj], [ri], color="red", marker="x", s=20, zorder=5)
            else:
                ax.axis("off")
            if i == D - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticks([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticks([])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_losses(losses: Any, path: Path, dpi: int, title: str | None = None) -> None:
    if plt is None:
        return
    import numpy as np

    fig, ax = plt.subplots(figsize=(4.0, 2.8), constrained_layout=True)
    if isinstance(losses, dict):
        for k, v in losses.items():
            ax.plot(np.asarray(v), label=str(k))
        ax.legend(frameon=False)
    else:
        ax.plot(np.asarray(losses))
    if title:
        ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_artifacts(
    *,
    outdir: str | Path,
    spec: Mapping[str, Any],
    run_cfg: Mapping[str, Any],
    flow_cfg: Mapping[str, Any],
    posterior_flow: Any,
    s_obs: jnp.ndarray,
    # PNPE
    posterior_samples: jnp.ndarray | None = None,
    theta_acc: jnp.ndarray | None = None,
    S_acc: jnp.ndarray | None = None,
    S_mean: jnp.ndarray | None = None,
    S_std: jnp.ndarray | None = None,
    th_mean: jnp.ndarray | None = None,
    th_std: jnp.ndarray | None = None,
    loss_history: Any | None = None,
    # RNPE extras
    s_flow: Any | None = None,
    robust_posterior_samples: jnp.ndarray | None = None,
    denoised_s_samples: jnp.ndarray | None = None,  # whitened s
    misspec_probs: jnp.ndarray | None = None,
    loss_history_theta: Any | None = None,
    loss_history_s: Any | None = None,
    # Labels
    theta_labels: Sequence[str] | None = None,
    summary_labels: Sequence[str] | None = None,
    # Reference values (theta true / pseudo-true)
    theta_ref: Sequence[float] | None = None,
) -> None:
    out = Path(outdir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Arrays
    # Pull reference theta from explicit arg or from run config if present
    ref_theta_arr: np.ndarray | None = None
    if theta_ref is not None:
        ref_theta_arr = _to_np(theta_ref)
    else:
        try:
            rt = run_cfg.get("theta_true")
            if rt is not None:
                ref_theta_arr = _to_np(rt)
        except Exception:
            ref_theta_arr = None

    if posterior_samples is not None:
        np.savez_compressed(out / "posterior_samples.npz", samples=_to_np(posterior_samples))
    if robust_posterior_samples is not None:
        np.savez_compressed(
            out / "posterior_samples_robust.npz",
            samples=_to_np(robust_posterior_samples),
        )
    if theta_acc is not None or S_acc is not None:
        np.savez_compressed(
            out / "accepted_preconditioning.npz",
            theta=None if theta_acc is None else _to_np(theta_acc),
            S=None if S_acc is None else _to_np(S_acc),
        )
    np.save(out / "s_obs.npy", _to_np(s_obs))
    if denoised_s_samples is not None:
        np.save(out / "s_denoised_w.npy", _to_np(denoised_s_samples))
    if misspec_probs is not None:
        np.save(out / "misspec_probs.npy", _to_np(misspec_probs))
    if S_mean is not None and S_std is not None:
        np.savez_compressed(
            out / "standardisers.npz",
            S_mean=_to_np(S_mean),
            S_std=_to_np(S_std),
            th_mean=None if th_mean is None else _to_np(th_mean),
            th_std=None if th_std is None else _to_np(th_std),
        )

    # Checkpoints
    try:
        eqx.tree_serialise_leaves(out / "posterior_flow.eqx", posterior_flow)
    except Exception:
        pass
    if s_flow is not None:
        try:
            eqx.tree_serialise_leaves(out / "summaries_flow.eqx", s_flow)
        except Exception:
            pass

    # Metadata
    with open(out / "config.json", "w") as f:
        json.dump(
            _jsonable(
                {
                    "spec": dict(spec),
                    "run": dict(run_cfg),
                    "flow": dict(flow_cfg),
                    "n_accepted_precond": (int(theta_acc.shape[0]) if theta_acc is not None else 0),
                }
            ),
            f,
            indent=2,
        )

    # Loss histories
    if loss_history is not None:
        with open(out / "loss_history.json", "w") as f:
            json.dump(_jsonable(loss_history), f, indent=2)
    if loss_history_theta is not None:
        with open(out / "loss_history_theta.json", "w") as f:
            json.dump(_jsonable(loss_history_theta), f, indent=2)
    if loss_history_s is not None:
        with open(out / "loss_history_s.json", "w") as f:
            json.dump(_jsonable(loss_history_s), f, indent=2)

    # Plots
    dpi = int(run_cfg.get("fig_dpi", 160))
    fmt = str(run_cfg.get("fig_format", "pdf"))
    if posterior_samples is not None:
        smp = _to_np(posterior_samples)
        th_labels = list(theta_labels) if theta_labels else [f"theta[{i}]" for i in range(smp.shape[1])]
        _plot_marginals(smp, th_labels, out / f"posterior_marginals.{fmt}", dpi, ref_theta_arr)
        _plot_pairgrid(smp, th_labels, out / f"posterior_pairplot.{fmt}", dpi, ref_theta_arr)
    if robust_posterior_samples is not None:
        smp_r = _to_np(robust_posterior_samples)
        th_labels = list(theta_labels) if theta_labels else [f"theta[{i}]" for i in range(smp_r.shape[1])]
        _plot_marginals(smp_r, th_labels, out / f"posterior_robust_marginals.{fmt}", dpi, ref_theta_arr)
        _plot_pairgrid(smp_r, th_labels, out / f"posterior_robust_pairplot.{fmt}", dpi, ref_theta_arr)
    if denoised_s_samples is not None:
        sden = _to_np(denoised_s_samples)
        s_labels = list(summary_labels) if summary_labels else [f"s[{i}]" for i in range(sden.shape[1])]
        _plot_marginals(sden, s_labels, out / f"s_denoised_marginals.{fmt}", dpi)
        _plot_pairgrid(sden, s_labels, out / f"s_denoised_pairplot.{fmt}", dpi)
    if loss_history is not None:
        _plot_losses(loss_history, out / f"training_losses.{fmt}", dpi, title="q(theta|s)")
    if loss_history_theta is not None:
        _plot_losses(
            loss_history_theta,
            out / f"training_losses_theta.{fmt}",
            dpi,
            title="q(theta|s)",
        )
    if loss_history_s is not None:
        _plot_losses(loss_history_s, out / f"training_losses_s.{fmt}", dpi, title="q(s)")

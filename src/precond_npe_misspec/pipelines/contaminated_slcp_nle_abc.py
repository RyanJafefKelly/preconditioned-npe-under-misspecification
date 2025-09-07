"""Contaminated SLCP experiment with NLE + ABC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import colormaps as cm
from matplotlib.lines import Line2D

from precond_npe_misspec.examples.contaminated_slcp import assumed_dgp, prior_sample, summaries, true_dgp
from precond_npe_misspec.pipelines.base_nle_abc import (
    ExperimentSpec,
    FlowConfig,
    RunConfig,
    default_coupling_rqs_builder,
    run_experiment,
)
from precond_npe_misspec.utils import distances as dist

type Array = jax.Array


# ---- Distance factory ----


def _distance_factory_from_cfg(cfg: Config) -> dist.DistanceFn:
    if cfg.distance == "euclidean":
        return lambda S_tr_w: dist.euclidean  # type: ignore
    if cfg.distance == "l1":
        return lambda S_tr_w: dist.l1  # type: ignore
    if cfg.distance == "mahalanobis":
        return lambda S_tr_w: dist.mahalanobis_from_whitened(  # type: ignore
            S_tr_w, ridge=cfg.mahalanobis_ridge
        )
    if cfg.distance == "mmd":
        if cfg.mmd_bandwidth is None:
            return lambda S_tr_w: dist.mmd_rbf_with_median(  # type: ignore
                S_tr_w, sqrt=True, unbiased=cfg.mmd_unbiased
            )
        else:
            return lambda _S_tr_w: dist.mmd_rbf_factory(  # type: ignore
                cfg.mmd_bandwidth, sqrt=True, unbiased=cfg.mmd_unbiased
            )
    raise ValueError(f"Unknown distance {cfg.distance}")


# ---- CLI config and main ----


@dataclass
class Config:
    seed: int = 0
    theta_true: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    num_draws: int = 5
    misspec_level: float = 1.0
    n_train: int = 4000
    n_props: int = 20000
    q_accept: float = 0.01
    # Flow
    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 5.0
    learning_rate: float = 5e-4
    max_epochs: int = 500
    max_patience: int = 20
    batch_size: int = 2048
    # Distance options
    distance: Literal["euclidean", "l1", "mahalanobis", "mmd"] = "euclidean"
    mahalanobis_ridge: float = 1e-3
    n_rep_summaries: int = 1  # use >1 for MMD
    mmd_unbiased: bool = False
    mmd_bandwidth: float | None = None  # None -> median heuristic
    # save results
    outdir: str = ""


def main(cfg: Config) -> None:
    s_dim = 2 * cfg.num_draws
    flow_cfg = FlowConfig(
        flow_layers=cfg.flow_layers,
        nn_width=cfg.nn_width,
        knots=cfg.knots,
        interval=cfg.interval,
        learning_rate=cfg.learning_rate,
        max_epochs=cfg.max_epochs,
        max_patience=cfg.max_patience,
        batch_size=cfg.batch_size,
    )
    run_cfg = RunConfig(
        seed=cfg.seed,
        theta_true=jnp.asarray(cfg.theta_true),
        n_train=cfg.n_train,
        n_props=cfg.n_props,
        q_accept=cfg.q_accept,
        sim_kwargs={"num_draws": cfg.num_draws, "misspec_level": cfg.misspec_level},
        n_rep_summaries=cfg.n_rep_summaries,
    )

    spec = ExperimentSpec(
        name="contaminated_slcp",
        theta_dim=5,
        s_dim=s_dim,
        prior_sample=prior_sample,
        true_dgp=lambda key, theta, **kw: true_dgp(key, theta, num_draws=kw["num_draws"]),
        simulate=lambda key, theta, **kw: assumed_dgp(key, theta, num_draws=kw["num_draws"]),
        summaries=summaries,
        build_flow=default_coupling_rqs_builder(s_dim=s_dim, theta_dim=5),
        baseline_posterior=None,
        # make_distance=_distance_factory_from_cfg(cfg),
    )

    result = run_experiment(spec, run_cfg, flow_cfg)

    def vec_q(arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        if arr.size == 0:
            return jnp.array([]), jnp.array([]), jnp.array([]), 0
        q = jnp.quantile(arr, jnp.array([0.025, 0.5, 0.975]), axis=0)
        return q[0], q[1], q[2], int(arr.shape[0])

    lo_t, med_t, hi_t, n_t = vec_q(result.acc_true)
    lo_s, med_s, hi_s, n_s = vec_q(result.acc_surr)

    th_star = jnp.asarray(cfg.theta_true)
    l2_true = float(jnp.linalg.norm(med_t - th_star)) if n_t > 0 else float("nan")
    l2_surr = float(jnp.linalg.norm(med_s - th_star)) if n_s > 0 else float("nan")

    # plot marginals
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    labels = [f"$\\theta_{i}$" for i in range(5)]
    for i in range(5):
        axes[i].hist(result.acc_true[:, i], bins=30, alpha=0.5, label="Normal ABC", color="C0")
        axes[i].hist(result.acc_surr[:, i], bins=30, alpha=0.5, label="Surrogate", color="C1")
        axes[i].axvline(cfg.theta_true[i], color="k", linestyle="--", label="θ*")
        axes[i].set_title(labels[i])
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.outdir}/marginals_{i + 1}.png")

    # plot bivariate plots (5x5 grid)
    if cfg.outdir:
        os.makedirs(cfg.outdir, exist_ok=True)

    th_star = jnp.asarray(cfg.theta_true)
    var_names = [f"theta{i}" for i in range(1, 6)]
    # var_name_map = {vn: rf"$\theta_{i}$" for i, vn in enumerate(var_names, start=1)}
    reference_values = {f"theta{i + 1}": float(th_star[i]) for i in range(5)}

    def to_idata(arr: jnp.ndarray) -> object:
        a = np.asarray(arr)
        if a.size == 0:
            return None
        if a.ndim == 1:
            a = a[:, None]
        post = {f"theta{i + 1}": a[:, i][None, :] for i in range(a.shape[1])}
        return az.from_dict(posterior=post)

    # id_true = to_idata(result.acc_true)
    id_surr = to_idata(result.acc_surr)

    fig, axes = plt.subplots(5, 5, sharey=False, figsize=(16, 16))

    # Layer 1: ABC with true simulator
    # NOTE: do upper triangle
    # if id_true is not None:
    #     axes = az.plot_pair(
    #         id_true,
    #         var_names=var_names,
    #         kind="kde",
    #         reference_values=reference_values,
    #         reference_values_kwargs={"color": "red", "marker": "X", "markersize": 10},
    #         kde_kwargs={
    #             "hdi_probs": [0.05, 0.25, 0.5, 0.75, 0.95],
    #             "contour_kwargs": {"colors": None, "cmap": plt.cm.viridis},
    #             "contourf_kwargs": {"alpha": 0},
    #         },
    #         ax=axes,
    #         labeller=az.labels.MapLabeller(var_name_map=var_name_map),
    #         marginals=True,
    #         marginal_kwargs={"label": "ABC true"},
    #     )

    # Layer 2: surrogate ABC (dashed contours)
    if id_surr is not None:
        axes = az.plot_pair(
            id_surr,
            var_names=var_names,
            kind="kde",
            reference_values=reference_values,
            reference_values_kwargs={"color": "red", "marker": "X", "markersize": 10},
            kde_kwargs={
                "hdi_probs": [0.05, 0.25, 0.5, 0.75, 0.95],
                "contour_kwargs": {
                    "colors": None,
                    # "cmap": cm.cividis,
                    "linestyles": "dashed",
                },
                "contourf_kwargs": {"alpha": 0},
            },
            ax=axes,
            # labeller=az.labels.MapLabeller(var_name_map=var_name_map),
            marginals=True,
            marginal_kwargs={"label": "Surrogate"},  # no linestyle here
        )

    # Safe formatting: skip removed axes
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if ax is None or ax.get_figure() is None:
                continue
            ax.set_xlim(-3, 3)
            ax.set_xticks([-3, 0, 3])
            if i != j:
                ax.set_ylim(-3, 3)
                ax.set_yticks([-3, 0, 3])

    # Make diagonal second overlay dashed and coloured
    for i in range(5):
        ax = axes[i, i]
        if ax is None or ax.get_figure() is None:
            continue
        lines = ax.get_lines()
        if len(lines) >= 2:
            lines[-1].set_linestyle("--")
            lines[-1].set_color("C1")

    handles = [
        # Line2D([0], [0], color="C0", lw=2, label="ABC true"),
        Line2D([0], [0], color="C1", lw=2, ls="--", label="Surrogate"),
        Line2D([0], [0], color="red", marker="X", linestyle="None", label=r"$\theta^\star$"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(f"{cfg.outdir}/slcp_joint_all.pdf", bbox_inches="tight")
    plt.close(fig)

    def fmt(v: jnp.ndarray) -> str:
        return jnp.array_str(jnp.asarray(v), precision=3)  # type: ignore

    print("Observed summaries:", fmt(result.s_obs))
    print(f"ABC true-sim   n={n_t}  median={fmt(med_t)}  95% CI=[{fmt(lo_t)} , {fmt(hi_t)}]  L2_err={l2_true:.3f}")
    print(f"ABC surrogate  n={n_s}  median={fmt(med_s)}  95% CI=[{fmt(lo_s)} , {fmt(hi_s)}]  L2_err={l2_surr:.3f}")
    print(f"θ*={fmt(th_star)}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))

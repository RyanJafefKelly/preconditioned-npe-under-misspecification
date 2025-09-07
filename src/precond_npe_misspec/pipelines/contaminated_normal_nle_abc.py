"""Run NLE and ABC on the contaminated normal example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# import equinox as eqx
import jax
import jax.numpy as jnp

from precond_npe_misspec.examples.contaminated_normal import (
    assumed_dgp,
    conjugate_posterior_under_assumed,
    prior_sample,
    summaries,
    true_dgp,
)
from precond_npe_misspec.pipelines.base_nle_abc import (
    ExperimentSpec,
    FlowConfig,
    RunConfig,
    default_coupling_rqs_builder,
    run_experiment,
)
from precond_npe_misspec.utils import distances as dist

type Array = jax.Array


def _distance_factory_from_cfg(cfg: Config) -> dist.DistanceFn:
    if cfg.distance == "euclidean":
        return lambda S_tr_w: dist.euclidean  # type: ignore
    if cfg.distance == "l1":
        return lambda S_tr_w: dist.l1  # type: ignore
    if cfg.distance == "mahalanobis":
        return lambda S_tr_w: dist.mahalanobis_from_whitened(  # type: ignore
            S_tr_w,
            ridge=cfg.mahalanobis_ridge,  # type: ignore
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


@dataclass
class Config:
    seed: int = 0
    theta_true: float = 2.0
    n_obs: int = 100
    stdev_err: float = 2.0
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
    max_patience: int = 10
    batch_size: int = 512
    # Distance options
    distance: Literal["euclidean", "l1", "mahalanobis", "mmd"] = "euclidean"
    n_rep_summaries: int = 1  # use >1 for MMD
    # mahalanobis_ridge: float = 1e-3
    mmd_unbiased: bool = False
    mmd_bandwidth: float | None = None  # None -> median heuristic


def main(cfg: Config) -> None:
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
        theta_true=cfg.theta_true,
        n_train=cfg.n_train,
        n_props=cfg.n_props,
        q_accept=cfg.q_accept,
        sim_kwargs={"n_obs": cfg.n_obs, "stdev_err": cfg.stdev_err},
    )

    spec = ExperimentSpec(
        name="contaminated_normal",
        theta_dim=1,
        s_dim=2,
        prior_sample=prior_sample,
        true_dgp=lambda key, theta, **kw: true_dgp(key, theta, kw["stdev_err"], kw["n_obs"]),
        simulate=lambda key, theta, **kw: assumed_dgp(key, theta, kw["n_obs"]),
        summaries=summaries,
        build_flow=default_coupling_rqs_builder(s_dim=2, theta_dim=1),
        baseline_posterior=conjugate_posterior_under_assumed,
        # make_distance=_distance_factory_from_cfg(cfg),
    )

    result = run_experiment(spec, run_cfg, flow_cfg)

    def quantiles(arr: jnp.ndarray) -> tuple[float, float, float, int]:
        q = jnp.quantile(arr, jnp.array([0.025, 0.5, 0.975]))
        return float(q[0]), float(q[1]), float(q[2]), int(arr.size)

    lo_t, med_t, hi_t, n_t = quantiles(result.acc_true)
    lo_s, med_s, hi_s, n_s = quantiles(result.acc_surr)
    # mean_assumed, sd_assumed = spec.baseline_posterior(result.x_obs)

    print("Observed summaries:", result.s_obs)
    print(f"ABC true-sim   n={n_t}  median={med_t:.3f}  95% CI=({lo_t:.3f}, {hi_t:.3f})")
    print(f"ABC surrogate  n={n_s}  median={med_s:.3f}  95% CI=({lo_s:.3f}, {hi_s:.3f})")
    # print(f"Assumed conjugate posterior  mean={mean_assumed:.3f}  sd={sd_assumed:.3f}")
    print(f"θ*={cfg.theta_true:.3f}  |  |median_surr−θ*|={abs(med_s - cfg.theta_true):.3f}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))

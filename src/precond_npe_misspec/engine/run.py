# engine/run.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as _np

from precond_npe_misspec.utils.artifacts import save_artifacts

from .npe_rs import fit_posterior_flow_npe_rs
from .posterior import fit_posterior_flow, sample_posterior
from .preconditioning import run_preconditioning
from .robust import denoise_s, fit_s_flow, sample_robust_posterior

type Array = jax.Array
type LossHistory = dict[str, list[float]]


@dataclass(frozen=True)
class PrecondConfig:
    method: Literal["none", "rejection", "smc_abc", "rf_abc"] = "rf_abc"
    n_sims: int = 20_000
    q_precond: float = 0.1
    store_raw_data: bool = False
    # SMC‑ABC
    smc_n_particles: int = 40
    smc_alpha: float = 0.5
    smc_epsilon0: float = 1e6
    smc_eps_min: float = 1e-3
    smc_acc_min: float = 0.1
    smc_max_iters: int = 3
    smc_initial_R: int = 1
    smc_c_tuning: float = 0.01
    smc_B_sim: int = 1
    # RF-ABC
    abc_rf_mode: Literal["multi", "per_param"] = "per_param"
    rf_n_estimators: int = 800
    rf_min_leaf: int = 40
    rf_max_depth: int | None = 10
    rf_train_frac: float = 1.0
    rf_random_state: int = 0
    rf_n_jobs: int = -1


@dataclass(frozen=True)
class PosteriorConfig:
    method: Literal["npe", "rnpe", "npe_rs"] = "rnpe"
    n_posterior_draws: int = 20_000


@dataclass(frozen=True)
class RobustConfig:
    denoise_model: Literal["laplace", "laplace_adaptive", "student_t", "cauchy", "spike_slab"] = "spike_slab"
    laplace_alpha: float = 0.3
    laplace_min_scale: float = 0.01
    student_t_scale: float = 0.05
    student_t_df: float = 1.0
    cauchy_scale: float = 0.05
    spike_std: float = 0.01
    slab_scale: float = 0.25
    misspecified_prob: float = 0.5
    learn_prob: bool = False
    mcmc_warmup: int = 1000
    mcmc_samples: int = 2000
    mcmc_thin: int = 1


@dataclass(frozen=True)
class NpeRsConfig:
    """Config for NPE‑RS."""

    embed_dim: int = 4
    embed_width: int = 128
    embed_depth: int = 2
    activation: Literal["relu", "tanh"] = "relu"
    mmd_weight: float = 1.0
    kernel: Literal["rbf", "imq"] = "rbf"
    mmd_subsample: int | None = 256
    bandwidth: float | Literal["median"] = "median"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    obs_seed: int = 1234
    theta_true: tuple[float, ...] = (0.0,)  # set by pipeline
    sim_kwargs: dict[str, Any] | None = None
    summaries_kwargs: dict[str, Any] | None = None
    outdir: str | None = None
    precond: PrecondConfig = field(default_factory=PrecondConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    batch_size: int = 512
    robust: RobustConfig = field(default_factory=RobustConfig)
    npers: NpeRsConfig = field(default_factory=NpeRsConfig)


@dataclass
class Result:
    theta_train: jnp.ndarray
    S_train: jnp.ndarray
    posterior_flow: Any
    x_obs: jnp.ndarray
    s_obs: jnp.ndarray
    S_mean: jnp.ndarray
    S_std: jnp.ndarray
    th_mean_post: jnp.ndarray
    th_std_post: jnp.ndarray
    posterior_samples_at_obs: jnp.ndarray
    loss_history_theta: Any
    # Robust extras (RNPE/PRNPE)
    denoised_s_samples: jnp.ndarray | None = None
    misspec_probs: jnp.ndarray | None = None


def run_experiment(spec: Any, run: RunConfig, flow_cfg: Any) -> Result:
    rng = jax.random.key(run.seed)
    obs_rng = jax.random.key(run.obs_seed)
    print("0")
    # Observed data
    x_obs = spec.true_dgp(obs_rng, jnp.asarray(run.theta_true), **(run.sim_kwargs or {}))
    s_obs = spec.summaries(x_obs)

    # Preconditioning → training set
    rng, k_pre = jax.random.split(rng)
    theta_tr, S_tr, X_tr = run_preconditioning(k_pre, spec, s_obs, run, flow_cfg)
    if run.posterior.method == "npe_rs" and X_tr is None:
        raise ValueError(
            "Preconditioning returned no raw data but NPE-RS posterior requires it. "
            "Set run.precond.store_raw_data=True."
        )
    print(f"jnp.max(S_tr)={jnp.max(S_tr)}, jnp.min(S_tr)={jnp.min(S_tr)}")

    # Fit q(theta | s) or q(theta | eta(s)) depending on method
    rng, k_fit = jax.random.split(rng)
    if run.posterior.method == "npe_rs":
        # rng, k_sim = jax.random.split(rng)

        # def _sim_one(k: Array, th: Array) -> Array:
        #     return cast(Array, spec.simulate(k, th, **(run.sim_kwargs or {})))

        # keys = jax.random.split(k_sim, theta_tr.shape[0])
        # X_tr = jax.vmap(_sim_one)(keys, theta_tr)
        q_theta_s_rs, X_mean, X_std, th_mean, th_std, losses_dict = fit_posterior_flow_npe_rs(
            k_fit, spec, theta_tr, X_tr, x_obs, flow_cfg, run.npers
        )
        q_theta_s = cast(Any, q_theta_s_rs)
        # For downstream compatibility, reuse S_* slots for x-whitening stats.
        S_mean, S_std = X_mean, X_std
        # loss_history: LossHistory = cast(LossHistory, losses_dict)

    else:
        q_theta_s_std, S_mean, S_std, th_mean, th_std, losses_list = fit_posterior_flow(
            k_fit, spec, theta_tr, S_tr, flow_cfg
        )
        q_theta_s = cast(Any, q_theta_s_std)
        if isinstance(losses_list, dict):
            losses_dict = {str(split): list(_np.asarray(values)) for split, values in losses_list.items()}
        else:
            losses_dict = {"nll": list(_np.asarray(losses_list))}
        # loss_history = cast(LossHistory, losses_dict)

        # losses_theta = cast(LossHistory, {"nll": list(losses_list)})
    if run.posterior.method == "npe_rs":
        x_obs_w = (x_obs - X_mean) / (X_std + 1e-8)
        print("x_obs_w: ", x_obs_w)
    else:
        s_obs_w = (s_obs - S_mean) / (S_std + 1e-8)
        S_tr_w = (S_tr - S_mean) / (S_std + 1e-8)
        print("s_obs_w: ", s_obs_w)

    # NPE/NPE-RS sampling
    if run.posterior.method in ("npe", "npe_rs"):
        rng, k_post = jax.random.split(rng)
        condition_vec = x_obs_w if run.posterior.method == "npe_rs" else s_obs_w
        theta_samps = sample_posterior(k_post, q_theta_s, condition_vec, run.posterior.n_posterior_draws)

        res = Result(
            theta_train=theta_tr,
            S_train=S_tr,
            posterior_flow=q_theta_s,
            x_obs=x_obs,
            s_obs=s_obs,
            S_mean=S_mean,
            S_std=S_std,
            th_mean_post=th_mean,
            th_std_post=th_std,
            posterior_samples_at_obs=theta_samps,
            loss_history_theta=losses_dict,
        )
    elif run.posterior.method == "rnpe":
        # RNPE: fit q(s), denoise, then mix q(theta|s)
        rng, k_sfit = jax.random.split(rng)
        q_s_w, _ = fit_s_flow(k_sfit, spec.s_dim, S_tr_w, flow_cfg)  # TODO: IDEA - TRAIN ON FULL S
        rng, k_mcmc = jax.random.split(rng)

        s_denoised_w, misspec_probs = denoise_s(k_mcmc, s_obs_w, q_s_w, run.robust)

        if run.outdir:
            _od = _Path(run.outdir)
            _od.mkdir(parents=True, exist_ok=True)
            # Save all denoised samples (M, s_dim) and their mean context vector (s_dim,)
            _np.savez_compressed(_od / "denoised_s_samples.npz", samples=_np.asarray(s_denoised_w))
            _np.save(_od / "s_obs_denoised_w.npy", _np.asarray(s_denoised_w).mean(axis=0))
            if misspec_probs is not None:
                _np.save(_od / "misspec_probs.npy", _np.asarray(misspec_probs))

        rng, k_mix = jax.random.split(rng)
        theta_samps_robust = sample_robust_posterior(k_mix, q_theta_s, s_denoised_w, run.posterior.n_posterior_draws)
        res = Result(
            theta_train=theta_tr,
            S_train=S_tr,
            posterior_flow=q_theta_s,
            x_obs=x_obs,
            s_obs=s_obs,
            S_mean=S_mean,
            S_std=S_std,
            th_mean_post=th_mean,
            th_std_post=th_std,
            posterior_samples_at_obs=theta_samps_robust,  # for metrics compatibility
            loss_history_theta=losses_list,
            denoised_s_samples=s_denoised_w,
            misspec_probs=misspec_probs,
        )
    else:
        raise ValueError(f"Unknown posterior.method: {run.posterior.method!r}")

    # Persist artefacts for metrics script
    if run.outdir:
        save_artifacts(
            outdir=run.outdir,
            spec={
                "name": getattr(spec, "name", "experiment"),
                "theta_dim": spec.theta_dim,
                "s_dim": spec.s_dim,
                "theta_labels": list(getattr(spec, "theta_labels", []) or []) or None,
                "summary_labels": list(getattr(spec, "summary_labels", []) or []) or None,
            },
            run_cfg=asdict(run),
            flow_cfg=asdict(flow_cfg),
            posterior_flow=res.posterior_flow,
            s_obs=res.s_obs,
            posterior_samples=(res.posterior_samples_at_obs if run.posterior.method in ("npe", "npe_rs") else None),
            robust_posterior_samples=(res.posterior_samples_at_obs if run.posterior.method == "rnpe" else None),
            theta_acc=res.theta_train,
            S_acc=res.S_train,
            S_mean=res.S_mean,
            S_std=res.S_std,
            th_mean=res.th_mean_post,
            th_std=res.th_std_post,
            loss_history=res.loss_history_theta,
            theta_labels=list(getattr(spec, "theta_labels", []) or []) or None,
            summary_labels=list(getattr(spec, "summary_labels", []) or []) or None,
        )
        ep = {
            "simulate": getattr(spec, "simulate_path", None),
            "summaries": getattr(spec, "summaries_path", None),
            "sim_kwargs": (run.sim_kwargs or {}),
            "summaries_kwargs": (run.summaries_kwargs or {}),
        }
        if ep["simulate"] and ep["summaries"] and run.outdir:
            Path(run.outdir, "entrypoints.json").write_text(json.dumps(ep, indent=2))

    return res

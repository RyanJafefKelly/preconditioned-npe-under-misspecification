# engine/run.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp

from precond_npe_misspec.utils.artifacts import save_artifacts

from .posterior import fit_posterior_flow, sample_posterior
from .preconditioning import run_preconditioning
from .robust import denoise_s, fit_s_flow, sample_robust_posterior


@dataclass(frozen=True)
class PrecondConfig:
    method: Literal["none", "rejection", "smc_abc"] = "none"
    n_sims: int = 200_000
    q_precond: float = 0.2
    # SMC‑ABC
    smc_n_particles: int = 1000
    smc_alpha: float = 0.5
    smc_epsilon0: float = 1e6
    smc_eps_min: float = 1e-3
    smc_acc_min: float = 0.10
    smc_max_iters: int = 5
    smc_initial_R: int = 1
    smc_c_tuning: float = 0.01
    smc_B_sim: int = 1


@dataclass(frozen=True)
class PosteriorConfig:
    method: Literal["npe", "rnpe"] = "npe"
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
class RunConfig:
    seed: int = 0
    obs_seed: int = 1234
    theta_true: tuple[float, ...] = (0.0,)  # set by pipeline
    sim_kwargs: dict[str, Any] | None = None
    outdir: str | None = None
    precond: PrecondConfig = field(default_factory=PrecondConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    batch_size: int = 512
    robust: RobustConfig = field(default_factory=RobustConfig)


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


def run_experiment(spec: Any, run: RunConfig, flow_cfg: Any) -> Result:
    rng = jax.random.key(run.seed)
    obs_rng = jax.random.key(run.obs_seed)

    # Observed data
    x_obs = spec.true_dgp(obs_rng, jnp.asarray(run.theta_true), **(run.sim_kwargs or {}))
    s_obs = spec.summaries(x_obs)

    # Preconditioning → training set
    rng, k_pre = jax.random.split(rng)
    theta_tr, S_tr = run_preconditioning(k_pre, spec, s_obs, run, flow_cfg)

    # Fit q(theta | s)
    rng, k_fit = jax.random.split(rng)
    q_theta_s, S_mean, S_std, th_mean, th_std, losses_theta = fit_posterior_flow(k_fit, spec, theta_tr, S_tr, flow_cfg)
    s_obs_w = (s_obs - S_mean) / (S_std + 1e-8)

    # NPE sampling
    if run.posterior.method == "npe":
        rng, k_post = jax.random.split(rng)
        theta_samps = sample_posterior(k_post, q_theta_s, s_obs_w, run.posterior.n_posterior_draws)

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
            loss_history_theta=losses_theta,
        )
    else:
        # RNPE: fit q(s), denoise, then mix q(theta|s)
        rng, k_sfit = jax.random.split(rng)
        q_s, _ = fit_s_flow(k_sfit, spec.s_dim, S_tr, flow_cfg)
        rng, k_mcmc = jax.random.split(rng)
        s_denoised_w, misspec_probs = denoise_s(k_mcmc, s_obs_w, q_s, run.robust)
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
            loss_history_theta=losses_theta,
        )

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
            posterior_samples=(res.posterior_samples_at_obs if run.posterior.method == "npe" else None),
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
        }
        if ep["simulate"] and ep["summaries"] and run.outdir:
            Path(run.outdir, "entrypoints.json").write_text(json.dumps(ep, indent=2))

    return res

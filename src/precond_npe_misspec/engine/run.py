# engine/run.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as _np

from precond_npe_misspec.utils.artifacts import save_artifacts

from .posterior import fit_posterior_flow, sample_posterior
from .preconditioning import run_preconditioning
from .robust import denoise_s, fit_s_flow, sample_robust_posterior


@dataclass(frozen=True)
class PrecondConfig:
    method: Literal["none", "rejection", "smc_abc"] = "smc_abc"
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
    method: Literal["npe", "rnpe"] = "rnpe"
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
    # Robust extras (RNPE/PRNPE)
    denoised_s_samples: jnp.ndarray | None = None
    misspec_probs: jnp.ndarray | None = None


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
    S_tr_w = (S_tr - S_mean) / (S_std + 1e-8)
    print("s_obs_w: ", s_obs_w)

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
        q_s_w, _ = fit_s_flow(k_sfit, spec.s_dim, S_tr_w, flow_cfg)
        rng, k_mcmc = jax.random.split(rng)
        print("q_s.log_prob(raw s_obs)   :", float(q_s_w.log_prob(s_obs)))
        print(
            "q_s.log_prob(whitened s_obs_w) (WRONG SCALE):",
            float(q_s_w.log_prob(s_obs_w)),
        )  # should be ~ -inf or very small

        s_denoised_w, misspec_probs = denoise_s(k_mcmc, s_obs_w, q_s_w, run.robust)
        print("s_denoised_w: ", s_denoised_w)
        print("misspec_probs: ", misspec_probs)
        resid = jnp.mean((s_denoised_w - s_obs_w) ** 2)
        print("MSE(denoised_w, obs_w):", float(resid))

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
            loss_history_theta=losses_theta,
            denoised_s_samples=s_denoised_w,
            misspec_probs=misspec_probs,
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

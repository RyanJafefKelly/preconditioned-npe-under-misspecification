# src/precond_npe_misspec/pipelines/svar_prnpe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from precond_npe_misspec.examples.svar import assumed_dgp as svar_assumed_dgp
from precond_npe_misspec.examples.svar import default_pairs
from precond_npe_misspec.examples.svar import prior_sample as svar_prior_sample
from precond_npe_misspec.examples.svar import summaries as svar_summaries
from precond_npe_misspec.examples.svar import true_dgp as svar_true_dgp
from precond_npe_misspec.pipelines.base_pnpe import (
    ExperimentSpec,
    FlowConfig,
    RunConfig,
    default_posterior_flow_builder,
    default_theta_flow_builder,
)
from precond_npe_misspec.pipelines.base_prnpe import run_experiment_prnpe


@dataclass
class Config:
    # Data-generating setup
    seed: int = 0
    theta_true: tuple[float, ...] = (0.579, -0.143, 0.836, 0.745, -0.660, -0.254, 0.1)
    k: int = 6
    T: int = 1000
    # Preconditioning (ABC)
    n_sims: int = 200_000
    q_precond: float = 0.1
    # Posterior draws
    n_posterior_draws: int = 5000
    # Flow hparams
    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 8.0
    learning_rate: float = 5e-4
    max_epochs: int = 500
    max_patience: int = 10
    batch_size: int = 512
    # Denoiser
    denoise_model: Literal["laplace", "laplace_adaptive", "student_t", "cauchy", "spike_slab"] = "laplace_adaptive"
    laplace_alpha: float = 0.3  # only for laplace_adaptive
    laplace_min_scale: float = 0.01
    student_t_scale: float = 0.05
    student_t_df: float = 1.0
    cauchy_scale: float = 0.05
    spike_std: float = 0.01
    slab_scale: float = 0.25
    misspecified_prob: float = 0.5
    learn_prob: bool = False
    # MCMC
    mcmc_warmup: int = 1000
    mcmc_samples: int = 2000
    mcmc_thin: int = 1


def main(cfg: Config) -> None:
    pairs = default_pairs(cfg.k)
    m = int(pairs.shape[0])
    theta_dim = 2 * m + 1
    s_dim = 2 * m + 1 + 1  # TODO: ADDED ONE FOR MEAN
    print("s_dim, ", s_dim)

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
        n_sims=cfg.n_sims,
        q_precond=cfg.q_precond,
        n_posterior_draws=cfg.n_posterior_draws,
        sim_kwargs={},
        batch_size=cfg.batch_size,
    )

    # NOTE: uncomment to test well-specified case
    # svar_true_dgp = svar_assumed_dgp

    spec = ExperimentSpec(
        name="svar_prnpe",
        theta_dim=theta_dim,
        s_dim=s_dim,
        prior_sample=lambda key: svar_prior_sample(key, pairs=pairs),
        true_dgp=lambda key, theta, **kw: svar_true_dgp(
            key,
            theta,
            k=cfg.k,
            T=cfg.T,
            pairs=pairs,
            eps=0.02,
            kappa=12.0,
            df=1.0,
            per_channel=True,
        ),
        simulate=lambda key, theta, **kw: svar_assumed_dgp(key, theta, k=cfg.k, T=cfg.T, pairs=pairs),
        summaries=lambda x: svar_summaries(x, pairs=pairs),
        build_theta_flow=default_theta_flow_builder(theta_dim),
        build_posterior_flow=default_posterior_flow_builder(theta_dim, s_dim),
    )

    denoise_kwargs = {
        "student_t_scale": cfg.student_t_scale,
        "student_t_df": cfg.student_t_df,
        "cauchy_scale": cfg.cauchy_scale,
        "spike_std": cfg.spike_std,
        "slab_scale": cfg.slab_scale,
        "misspecified_prob": cfg.misspecified_prob,
        "learn_prob": cfg.learn_prob,
        "laplace_alpha": cfg.laplace_alpha,
        "laplace_min_scale": cfg.laplace_min_scale,
    }

    res = run_experiment_prnpe(
        spec,
        run_cfg,
        flow_cfg,
        denoise_model=cfg.denoise_model,
        denoise_kwargs=denoise_kwargs,
        mcmc_num_warmup=cfg.mcmc_warmup,
        mcmc_num_samples=cfg.mcmc_samples,
        mcmc_thinning=cfg.mcmc_thin,
    )

    def qtiles(arr: jnp.ndarray) -> tuple[float, float, float]:
        q = jnp.quantile(arr, jnp.array([0.025, 0.5, 0.975]), axis=0)
        return float(q[0]), float(q[1]), float(q[2])

    print(f"Accepted in preconditioning: {int(res.theta_acc_precond.shape[0])}")
    post = res.posterior_samples_at_obs_robust
    for i in range(theta_dim):
        lo, med, hi = qtiles(post[:, i])
        print(f"θ[{i}]  median={med:.4f}  95% CI=({lo:.4f}, {hi:.4f})")
    if res.misspec_probs is not None:
        print("Posterior misspecification probabilities:", res.misspec_probs)
    print("Observed summaries:", res.s_obs)
    print("True parameters:", cfg.theta_true)
    print("Posterior mean error:", jnp.mean(post, axis=0) - jnp.array(cfg.theta_true))

    def _summ_label(idx: int) -> str:
        m = pairs.shape[0]
        if idx < m:
            i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
            return f"{j}→{i} (lag‑1)"
        elif idx < 2 * m:
            r = idx - m
            i, j = int(pairs[r, 1]), int(pairs[r, 0])
            return f"{j}→{i} (lag‑1)"
        else:
            return "pooled σ"

    # Whitened obs and denoised distribution
    xw = res.denoised_s_samples  # (M, s_dim) whitened
    xw_mean = xw.mean(axis=0)
    xw_q = jnp.quantile(xw, jnp.array([0.025, 0.5, 0.975]), axis=0)

    # Back to raw scale
    x_mean = xw_mean * res.S_std + res.S_mean
    x_lo = xw_q[0] * res.S_std + res.S_mean
    x_hi = xw_q[2] * res.S_std + res.S_mean
    adj = res.s_obs - x_mean  # adjustment applied by denoiser

    print("Denoising adjustments (obs − E[x|y]) and denoised 95% intervals:")
    for d in range(s_dim):
        lab = _summ_label(d)
        miss = None if res.misspec_probs is None else float(res.misspec_probs[d])
        line = (
            f"s[{d}] {lab}: "
            f"obs={float(res.s_obs[d]):+.4f}  "
            f"E[x|y]={float(x_mean[d]):+.4f}  "
            f"Δ={float(adj[d]):+.4f}  "
            f"denoised95%=({float(x_lo[d]):+.4f}, {float(x_hi[d]):+.4f})"
        )
        if miss is not None:
            line += f"  misspec_prob={miss:.2f}"
        print(line)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))

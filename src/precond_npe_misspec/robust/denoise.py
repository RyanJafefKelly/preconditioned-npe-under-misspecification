from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from precond_npe_misspec.robust.fj_numpyro_bridge import as_numpyro_dist

# --- denoiser models ---------------------------------------------------------


def student_t_denoiser(y_obs_w: jnp.ndarray, flow_s: Any, scale: float = 0.05, df: float = 1.0) -> None:
    x = numpyro.sample("x", as_numpyro_dist(flow_s))
    numpyro.sample("y", dist.StudentT(df, loc=x, scale=scale), obs=y_obs_w)


def cauchy_denoiser(y_obs_w: jnp.ndarray, flow_s: Any, scale: float = 0.05) -> None:
    x = numpyro.sample("x", as_numpyro_dist(flow_s))
    numpyro.sample("y", dist.Cauchy(x, scale), obs=y_obs_w)


def laplace_denoiser(y_obs_w: jnp.ndarray, flow_s: Any, scale: float = 0.05) -> None:
    x = numpyro.sample("x", as_numpyro_dist(flow_s))
    numpyro.sample("y", dist.Laplace(x, scale), obs=y_obs_w)


def laplace_obs_scaled_denoiser(y_obs_w: jnp.ndarray, flow_s: Any, alpha: float = 0.3, min_scale: float = 0.01) -> None:
    x = numpyro.sample("x", as_numpyro_dist(flow_s))
    b = jnp.maximum(alpha * jnp.abs(y_obs_w), min_scale)
    numpyro.sample("y", dist.Laplace(loc=x, scale=b), obs=y_obs_w)


def spike_and_slab_denoiser(
    y_obs_w: jnp.ndarray,
    flow_s: Any,
    spike_std: float = 0.01,
    slab_scale: float = 0.25,
    misspecified_prob: float = 0.5,
    learn_prob: bool = False,
) -> None:
    rho = numpyro.sample("misspecified_prob", dist.Uniform(0, 1)) if learn_prob else misspecified_prob
    x = numpyro.sample("x", as_numpyro_dist(flow_s))
    ll0 = dist.Normal(x, spike_std).log_prob(y_obs_w)  # spike
    ll1 = dist.Cauchy(x, slab_scale).log_prob(y_obs_w)  # slab
    log_mix = jnp.logaddexp(jnp.log1p(-rho) + ll0, jnp.log(rho) + ll1)
    numpyro.factor("mix_ll", jnp.sum(log_mix))


# --- driver ------------------------------------------------------------------


def run_denoising_mcmc(  # type: ignore
    key,
    y_obs_w: jnp.ndarray,
    flow_s: Any,
    model: Literal["laplace", "laplace_adaptive", "student_t", "cauchy", "spike_slab"] = "spike_slab",
    num_warmup: int = 1000,
    num_samples: int = 2000,
    thinning: int = 1,
    laplace_alpha: float = 0.3,
    laplace_min_scale: float = 0.01,
    student_t_scale: float = 0.05,
    student_t_df: float = 1.0,
    cauchy_scale: float = 0.05,
    spike_std: float = 0.01,
    slab_scale: float = 0.5,
    misspecified_prob: float = 0.5,
    learn_prob: bool = False,
) -> dict[str, jnp.ndarray]:
    if model == "laplace":
        kernel = NUTS(lambda y_obs_w, flow_s: laplace_denoiser(y_obs_w, flow_s))
    elif model == "laplace_adaptive":
        kernel = NUTS(
            lambda y_obs_w, flow_s: laplace_obs_scaled_denoiser(y_obs_w, flow_s, laplace_alpha, laplace_min_scale)
        )
    elif model == "student_t":
        kernel = NUTS(lambda y_obs_w, flow_s: student_t_denoiser(y_obs_w, flow_s, student_t_scale, student_t_df))
    elif model == "cauchy":
        kernel = NUTS(lambda y_obs_w, flow_s: cauchy_denoiser(y_obs_w, flow_s, cauchy_scale))
    elif model == "spike_slab":
        kernel = NUTS(
            lambda y_obs_w, flow_s: spike_and_slab_denoiser(
                y_obs_w, flow_s, spike_std, slab_scale, misspecified_prob, learn_prob
            ),
            target_accept_prob=0.9,
        )
    else:
        raise ValueError(f"Unknown model '{model}'")

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        thinning=thinning,
        progress_bar=True,
    )
    mcmc.run(key, y_obs_w=y_obs_w, flow_s=flow_s)
    out = mcmc.get_samples(group_by_chain=False)
    mcmc.print_summary()

    x_samples = out["x"]
    res = {"x_samples": x_samples}

    if model == "spike_slab":
        rho_samps = (
            out["misspecified_prob"]
            if "misspecified_prob" in out
            else jnp.full((x_samples.shape[0],), misspecified_prob)
        )
        ll0 = dist.Normal(x_samples, spike_std).log_prob(y_obs_w)
        ll1 = dist.Cauchy(x_samples, slab_scale).log_prob(y_obs_w)
        log_r0 = jnp.log1p(-rho_samps)[:, None] + ll0
        log_r1 = jnp.log(rho_samps)[:, None] + ll1
        pi = jnp.exp(log_r1 - jnp.logaddexp(log_r0, log_r1))
        res["misspec_probs"] = jnp.mean(pi, axis=0)
        if "misspecified_prob" in out:
            res["misspecified_prob"] = jnp.mean(rho_samps)
    return res

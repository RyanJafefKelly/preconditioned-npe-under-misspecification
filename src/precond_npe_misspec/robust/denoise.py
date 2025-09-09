from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from flowjax.experimental.numpyro import sample as fj_sample
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS


class FlowDist(dist.Distribution):  # type: ignore
    """Wrap a FlowJAX Distribution as a NumPyro Distribution (for log_prob in MCMC)."""

    support = constraints.real

    def __init__(self, flow: FlowDist):
        self.flow = flow
        event_shape = (flow.dim,)  # FlowJAX distributions expose .dim
        super().__init__(batch_shape=(), event_shape=event_shape)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        # value: (..., dim)
        batch = value.shape[:-1]
        val2d = value.reshape(-1, self.flow.dim)
        lp = self.flow.log_prob(val2d)
        return lp.reshape(batch)


def student_t_denoiser(
    y_obs_w: jnp.ndarray,
    flow_s: FlowDist,
    scale: float = 0.05,
    df: float = 1.0,
) -> None:
    x = fj_sample("x", flow_s)
    numpyro.sample("y", dist.StudentT(df, loc=x, scale=scale), obs=y_obs_w)


def cauchy_denoiser(
    y_obs_w: jnp.ndarray,
    flow_s: FlowDist,
    scale: float = 0.05,
) -> None:
    x = fj_sample("x", flow_s)
    numpyro.sample("y", dist.Cauchy(x, scale), obs=y_obs_w)


def laplace_denoiser(
    y_obs_w: jnp.ndarray,
    flow_s: FlowDist,
    scale: float = 0.05,
) -> None:
    x = fj_sample("x", flow_s)
    numpyro.sample("y", dist.Laplace(x, scale), obs=y_obs_w)


# def spike_and_slab_denoiser(  # type: ignore
#     y_obs_w: jnp.ndarray,
#     flow_s: FlowDist,
#     spike_std: float = 0.01,
#     slab_scale: float = 0.25,
#     misspecified_prob: float = 0.5,
#     learn_prob: bool = False,
# ):
#     """TODO: Docstring ... think on best MCMC approach... okay to avoid discrete step?."""
#     if learn_prob:
#         p = numpyro.sample("misspecified_prob", dist.Uniform(0, 1))
#     else:
#         p = misspecified_prob
#     with numpyro.plate("d", y_obs_w.shape[-1]):
#         misspecified = numpyro.sample("misspecified", dist.Bernoulli(probs=p))
#     x = fj_sample("x", flow_s)
#     with numpyro.handlers.mask(mask=(~misspecified.astype(bool))):
#         numpyro.sample("y_w", dist.Normal(x, spike_std), obs=y_obs_w)
#     with numpyro.handlers.mask(mask=misspecified.astype(bool)):
#         numpyro.sample("y_m", dist.Cauchy(x, slab_scale), obs=y_obs_w)


def spike_and_slab_denoiser(  # type: ignore
    y_obs_w: jnp.ndarray,
    flow_s: FlowDist,
    spike_std: float = 0.01,
    slab_scale: float = 0.25,
    misspecified_prob: float = 0.5,
    learn_prob: bool = False,
):
    """TODO: Docstring ... think on best MCMC approach... okay to avoid discrete step?."""
    rho = numpyro.sample("misspecified_prob", dist.Uniform(0, 1)) if learn_prob else misspecified_prob
    x = fj_sample("x", flow_s)  # latent denoised summaries
    ll0 = dist.Normal(x, spike_std).log_prob(y_obs_w)  # spike
    ll1 = dist.Cauchy(x, slab_scale).log_prob(y_obs_w)  # slab
    # log p(y|x,ρ) = Σ_j log[(1-ρ) exp(ll0_j) + ρ exp(ll1_j)]
    log_mix = jnp.logaddexp(jnp.log1p(-rho) + ll0, jnp.log(rho) + ll1)
    numpyro.factor("mix_ll", jnp.sum(log_mix))
    return None


def spike_and_slab_marginal_denoiser(
    y_obs_w: jnp.ndarray,
    flow_s: FlowDist,
    spike_std: float = 0.01,
    slab_scale: float = 0.25,
    misspecified_prob: float = 0.5,
    learn_prob: bool = False,
) -> None:
    rho = numpyro.sample("misspecified_prob", dist.Uniform(0, 1)) if learn_prob else misspecified_prob
    x = fj_sample("x", flow_s)  # latent denoised summaries
    ll0 = dist.Normal(x, spike_std).log_prob(y_obs_w)  # spike
    ll1 = dist.Cauchy(x, slab_scale).log_prob(y_obs_w)  # slab
    # log p(y|x) = sum_j log[(1-rho) exp(ll0) + rho exp(ll1)]
    log_mix = jnp.logaddexp(jnp.log1p(-rho) + ll0, jnp.log(rho) + ll1)
    numpyro.factor("mix_ll", jnp.sum(log_mix))
    return None


def laplace_obs_scaled_denoiser(
    y_obs_w: jnp.ndarray,
    flow_s: FlowDist,
    alpha: float = 0.3,
    min_scale: float = 0.01,
) -> None:
    x = fj_sample("x", flow_s)
    b = jnp.maximum(alpha * jnp.abs(y_obs_w), min_scale)  # per-dim scale on whitened summaries
    numpyro.sample("y", dist.Laplace(loc=x, scale=b), obs=y_obs_w)


def run_denoising_mcmc(  # type: ignore
    key,
    y_obs_w: jnp.ndarray,
    flow_s: FlowDist,
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
    slab_scale: float = 0.25,
    misspecified_prob: float = 0.5,
    learn_prob: bool = False,
) -> dict[str, jnp.ndarray]:
    """Returns dict with 'x_samples' and optionally 'misspec_probs' if spike-slab."""
    if model == "laplace":
        kernel = NUTS(lambda y_obs_w, flow_s: laplace_denoiser(y_obs_w, flow_s))
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            thinning=thinning,
            progress_bar=True,
        )
        mcmc.run(key, y_obs_w=y_obs_w, flow_s=flow_s)
    elif model == "laplace_adaptive":
        kernel = NUTS(
            lambda y_obs_w, flow_s: laplace_obs_scaled_denoiser(
                y_obs_w, flow_s, alpha=laplace_alpha, min_scale=laplace_min_scale
            )
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            thinning=thinning,
            progress_bar=True,
        )
        mcmc.run(key, y_obs_w=y_obs_w, flow_s=flow_s)
    elif model == "student_t":
        kernel = NUTS(lambda y_obs_w, flow_s: student_t_denoiser(y_obs_w, flow_s, student_t_scale, student_t_df))
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            thinning=thinning,
            progress_bar=True,
        )
        mcmc.run(key, y_obs_w=y_obs_w, flow_s=flow_s)
    elif model == "cauchy":
        kernel = NUTS(lambda y_obs_w, flow_s: cauchy_denoiser(y_obs_w, flow_s, cauchy_scale))
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            thinning=thinning,
            progress_bar=True,
        )
        mcmc.run(key, y_obs_w=y_obs_w, flow_s=flow_s)
    elif model == "spike_slab":
        kernel = NUTS(
            lambda y_obs_w, flow_s: spike_and_slab_denoiser(
                y_obs_w, flow_s, spike_std, slab_scale, misspecified_prob, learn_prob
            )
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            thinning=thinning,
            progress_bar=True,
        )
        mcmc.run(key, y_obs_w=y_obs_w, flow_s=flow_s)
    else:
        raise ValueError(f"Unknown model '{model}'")

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
        ll0 = dist.Normal(x_samples, spike_std).log_prob(y_obs_w)  # (M,d)
        ll1 = dist.Cauchy(x_samples, slab_scale).log_prob(y_obs_w)  # (M,d)
        log_r0 = jnp.log1p(-rho_samps)[:, None] + ll0
        log_r1 = jnp.log(rho_samps)[:, None] + ll1
        # π = P(z=1 | y, x, ρ) per dim
        pi = jnp.exp(log_r1 - jnp.logaddexp(log_r0, log_r1))  # (M,d)
        res["misspec_probs"] = jnp.mean(pi, axis=0)
        if "misspecified_prob" in out:
            res["misspecified_prob"] = jnp.mean(rho_samps)
    return res

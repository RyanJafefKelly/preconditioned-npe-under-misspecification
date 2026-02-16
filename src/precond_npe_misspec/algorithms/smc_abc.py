from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from blackjax.smc.abc import abc_step as smc_step
from blackjax.smc.abc import default_stopping
from blackjax.smc.abc import init as smc_init

from precond_npe_misspec.utils import distances as dist

type Array = jax.Array

# mypy: disable-error-code=no-untyped-def


def _build_distance(
    *,
    s_obs: Array,
    S_pilot: Array | None,
    make_distance: Callable[[Array], Callable[[Array, Array], Array]] | None,
) -> Callable[[Array], Array]:
    """Return ρ(s_sim) where input is a summary or a batch of summaries."""
    base: Callable[[Array, Array], Array]
    if make_distance is not None:
        if S_pilot is None:
            raise ValueError("S_pilot must be provided when make_distance is supplied.")
        base = make_distance(S_pilot)
    else:
        base = cast(Callable[[Array, Array], Array], dist.euclidean)

    def rho(s_sim: Array) -> Array:
        if s_sim.ndim == 1:
            return base(s_sim[None, :], s_obs)[0]
        return jnp.mean(base(s_sim, s_obs))

    return rho


def run_smc_abc(
    *,
    key: Array,
    n_particles: int,
    epsilon0: float,
    alpha: float,
    eps_min: float,
    acc_min: float,
    max_iters: int,
    initial_R: int,
    c_tuning: float,
    B_sim: int,
    spec,
    s_obs: Array,
    sim_kwargs: dict[str, Any] | None,
    S_pilot_for_distance: Array | None,
) -> tuple[Array, Array, Array]:
    """Return (theta_particles, S_particles, x_particles) at final ε without a second simulation pass."""
    sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)
    key, k_init, k_draw = jax.random.split(key, 3)

    # θ ~ prior
    th_keys = jax.random.split(k_draw, n_particles)
    theta0 = jax.vmap(spec.prior_sample)(th_keys)  # (N, d)

    # simulate raw x; handle B_sim > 1 internally
    def simulate_x_fn(k: Array, th: Array) -> Array:
        if B_sim > 1:
            keys = jax.random.split(k, B_sim)
            return jax.vmap(lambda kk: spec.simulate(kk, th, **sim_kwargs))(
                keys
            )  # (B, …)
        return cast(Array, spec.simulate(k, th, **sim_kwargs))  # (…,)

    # batched summaries
    def summary_fn(x: Array) -> Array:
        # For B_sim==1: x is a single trajectory array (e.g., (T,K)); do NOT vmap.
        # For B_sim>1 : x is (B, …); vmap over the leading batch only.
        return (
            jax.vmap(spec.summaries, in_axes=0)(x) if B_sim > 1 else spec.summaries(x)
        )

    # distance on summaries
    rho = _build_distance(
        s_obs=s_obs,
        S_pilot=S_pilot_for_distance,
        make_distance=getattr(spec, "make_distance", None),
    )

    # prior log pdf
    if getattr(spec, "prior_logpdf", None) is not None:
        prior_logpdf = spec.prior_logpdf
    elif (spec.theta_lo is not None) and (spec.theta_hi is not None):
        lo, hi = spec.theta_lo, spec.theta_hi

        def prior_logpdf(th: Array) -> Array:
            ok = jnp.all((th >= lo) & (th <= hi))
            return jnp.where(ok, 0.0, -jnp.inf)

    else:

        def prior_logpdf(_: Array) -> Array:
            return jnp.array(0.0)

    # init
    state = smc_init(
        k_init,
        theta0,
        epsilon0,
        distance_fn=rho,  # expects summaries
        simulate_fn=simulate_x_fn,  # returns raw x
        summary_fn=summary_fn,  # x -> summaries
        initial_R=initial_R,
    )

    # compiled step
    _core = jax.jit(
        lambda k, st, R: smc_step(
            k,
            st,
            R,
            simulate_fn=simulate_x_fn,
            summary_fn=summary_fn,
            distance_fn=rho,
            prior_logpdf=prior_logpdf,
            alpha=alpha,
            c=c_tuning,
            B_sim=B_sim,
        ),
        static_argnums=(2,),
    )

    # iterate
    t, key_loop = 0, key
    MIN_ITERS = 2
    eps_hist, acc_hist = [], []

    while t < max_iters:
        key_loop, k = jax.random.split(key_loop)
        state, info = _core(k, state, int(state.R))
        t += 1
        eps_hist.append(float(state.epsilon))
        acc_hist.append(float(info.acceptance_rate))
        print(
            f"[smc] t={t} eps={state.epsilon:.6g} acc={info.acceptance_rate:.3f} R={int(state.R)}"
        )
        if (t >= MIN_ITERS) and default_stopping(
            state, info, eps_min=eps_min, acc_min=acc_min
        ):
            break

    # directly return cached simulations and summaries
    return state.particles, state.summaries, state.sim_x

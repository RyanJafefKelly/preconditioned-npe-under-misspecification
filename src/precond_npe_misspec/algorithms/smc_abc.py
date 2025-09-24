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
    """
    Returns a function ρ(s_sim) -> scalar. Uses spec.make_distance if provided,
    else Euclidean on summaries.
    """
    base: Callable[[Array, Array], Array]
    if make_distance is not None:
        if S_pilot is None:
            raise ValueError("S_pilot must be provided when make_distance is supplied.")
        base = make_distance(S_pilot)
    else:
        base = cast(Callable[[Array, Array], Array], dist.euclidean)

    def rho(s_sim: Array) -> Array:
        # s_sim: (d,) or (B,d). Reduce to scalar via mean if batched.
        if s_sim.ndim == 1:
            d = base(s_sim[None, :], s_obs)[0]
        else:
            d = jnp.mean(base(s_sim, s_obs))
        return jnp.asarray(d)

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
    spec,  # ExperimentSpec
    s_obs: Array,
    sim_kwargs: dict[str, Any] | None,
    S_pilot_for_distance: Array | None,
) -> tuple[Array, Array, Array]:
    """Return (theta_particles, S_particles, x_particles) at final ε."""
    sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)
    key, k_init, k_draw = jax.random.split(key, 3)

    # initial particles θ ~ prior
    th_keys = jax.random.split(k_draw, n_particles)
    theta0 = jax.vmap(spec.prior_sample)(th_keys)  # (N,d)

    # simulate_fn: θ -> summaries (or (B,d) if B_sim>1)
    def _one_sim(k: Array, th: Array) -> Array:
        x = spec.simulate(k, th, **sim_kwargs)
        return cast(Array, spec.summaries(x))

    if B_sim > 1:

        def simulate_fn(k: Array, th: Array) -> Array:
            keys = jax.random.split(k, B_sim)
            return jax.vmap(lambda kk: _one_sim(kk, th))(keys)  # (B,d)

    else:

        def simulate_fn(k: Array, th: Array) -> Array:
            return _one_sim(k, th)  # (d,)

    # distance
    rho = _build_distance(
        s_obs=s_obs, S_pilot=S_pilot_for_distance, make_distance=spec.make_distance
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
        # last resort
        def prior_logpdf(_: Array) -> Array:
            return jnp.array(0.0)

    # init
    state = smc_init(
        k_init,
        theta0,
        epsilon0,
        distance_fn=rho,
        simulate_fn=simulate_fn,
        initial_R=initial_R,
    )

    # compiled SMC step (R is static)
    _core = jax.jit(
        lambda k, st, R: smc_step(
            k,
            st,
            R,
            simulate_fn=simulate_fn,
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
    while t < max_iters:
        key_loop, k = jax.random.split(key_loop)
        state, info = _core(k, state, int(state.R))
        t += 1
        print("state.epsilon: ", state.epsilon)
        if default_stopping(state, info, eps_min=eps_min, acc_min=acc_min):
            break

    # one pass to get simulated data and summaries for training
    kS = jax.random.split(key_loop, n_particles)
    def _simulate_and_summarize(kk: Array, th: Array) -> tuple[Array, Array]:
        x = spec.simulate(kk, th, **sim_kwargs)
        return x, spec.summaries(x)

    x_acc, S_acc = jax.vmap(_simulate_and_summarize)(kS, state.particles)
    return state.particles, S_acc, x_acc

"""Shared experiment spec and flow config for pipelines.

This module centralises the ExperimentSpec container used by example pipelines,
the FlowConfig used to build posterior flows, and a default posterior flow
builder compatible with FlowJAX/equinox.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.numpy as jnp
from flowjax.distributions import Normal
from flowjax.flows import coupling_flow

from precond_npe_misspec.examples.embeddings import EmbedBuilder

type Array = jax.Array

if TYPE_CHECKING:

    class _EqxModule:  # pragma: no cover - typing shim for eqx.Module
        def sample(self, key: Array, shape: tuple[int, ...], *, condition: Array) -> Array: ...

else:
    _EqxModule = eqx.Module


# Optional distance factory (if using ABC distances)
DistanceFn = Callable[[Array, Array], Array]
DistanceFactory = Callable[[Array], DistanceFn]


@dataclass(frozen=True)
class ExperimentSpec:
    """Container describing a simulation experiment.

    Pipelines fill this with problem‑specific callables. The engine only relies on
    these fields and does not require a specific subclass.
    """

    name: str
    theta_dim: int
    s_dim: int
    prior_sample: Callable[[Array], jnp.ndarray]
    prior_logpdf: Callable[[Array], Array] | None
    true_dgp: Callable[..., jnp.ndarray]  # true_dgp(key, theta, **sim_kwargs) -> x
    simulate: Callable[..., jnp.ndarray]  # simulate(key, theta, **sim_kwargs) -> x
    summaries: Callable[[jnp.ndarray], jnp.ndarray]  # s = summaries(x)
    build_posterior_flow: Callable[[Array, FlowConfig], eqx.Module]
    build_embedder: EmbedBuilder | None = None
    make_distance: DistanceFactory | None = None
    theta_labels: tuple[str, ...] | None = None
    summary_labels: tuple[str, ...] | None = None
    theta_lo: jnp.ndarray | None = None  # (theta_dim,)
    theta_hi: jnp.ndarray | None = None  # (theta_dim,)
    simulate_path: str | None = None
    summaries_path: str | None = None


@dataclass(frozen=True)
class FlowConfig:
    """Hyper‑parameters for FlowJAX conditional flows."""

    flow_layers: int = 8
    nn_width: int = 128
    knots: int = 10
    interval: float = 8.0
    learning_rate: float = 5e-4
    max_epochs: int = 50
    max_patience: int = 10
    batch_size: int = 512


def default_posterior_flow_builder(theta_dim: int, s_dim: int) -> Callable[[Array, FlowConfig], eqx.Module]:
    """Factory for a simple conditional coupling flow q(θ | s).

    Returns a builder that, given a PRNGKey and FlowConfig, constructs a FlowJAX
    coupling flow with an RQ‑spline transformer and an MLP conditioner over s.
    """

    def _builder(key: Array, cfg: FlowConfig) -> eqx.Module:
        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(theta_dim)),  # random variable is θ
            transformer=bij.RationalQuadraticSpline(knots=cfg.knots, interval=cfg.interval),
            cond_dim=s_dim,  # condition on s
            flow_layers=int(cfg.flow_layers),
            nn_width=int(cfg.nn_width),
        )

    return _builder

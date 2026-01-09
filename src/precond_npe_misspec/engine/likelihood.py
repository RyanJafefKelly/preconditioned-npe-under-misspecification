"""Placeholders for neural likelihood estimation (NLE) helpers.

These definitions provide the interface required by ``run_experiment`` when we
add an NLE branch. The concrete training and sampling logic should be filled in
later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

type Array = jax.Array


@dataclass
class LikelihoodFit:
    """Container for likelihood-model artefacts returned by ``fit_likelihood_flow``."""

    flow: Any | None = None
    s_mean: jnp.ndarray | None = None
    s_std: jnp.ndarray | None = None
    theta_transform: dict[str, jnp.ndarray] | None = None
    loss_history: Any | None = None


def fit_likelihood_flow(
    key: Array,
    spec: Any,
    theta_train: jnp.ndarray,
    S_train: jnp.ndarray,
    flow_cfg: Any,
) -> LikelihoodFit:
    """Train a conditional flow q(s | θ) for NLE.

    TODO: implement likelihood training, return a populated ``LikelihoodFit``.
    """
    raise NotImplementedError("fit_likelihood_flow is a placeholder; implement NLE training.")


def sample_posterior_via_importance(
    key: Array,
    spec: Any,
    likelihood: LikelihoodFit,
    s_obs: jnp.ndarray,
    n_draws: int,
) -> jnp.ndarray:
    """Draw posterior samples using importance resampling with the learned likelihood."""
    raise NotImplementedError("sample_posterior_via_importance is a placeholder; implement NLE sampling.")

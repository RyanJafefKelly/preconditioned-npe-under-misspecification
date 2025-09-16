from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as nd
from numpyro.distributions import constraints
from numpyro.distributions.constraints import Constraint


class FlowToNumPyro(nd.Distribution):  # type: ignore[misc]
    arg_constraints: dict[str, Constraint] = {}
    support = constraints.real_vector
    has_enumerate_support = False

    def __init__(self, flow):  # type: ignore
        self._flow = flow
        ex = flow.sample(jax.random.key(0), ())  # infer event shape
        super().__init__(batch_shape=(), event_shape=tuple(ex.shape), validate_args=False)

    def sample(self, key, sample_shape: tuple[int, ...] = ()):  # type: ignore
        return self._flow.sample(key, sample_shape)

    def log_prob(self, value: jnp.ndarray):  # type: ignore
        d = value.shape[-1]
        val2d = value.reshape(-1, d)
        lp = self._flow.log_prob(val2d)
        return lp.reshape(value.shape[:-1])


def as_numpyro_dist(flow) -> nd.Distribution:  # type: ignore
    return FlowToNumPyro(flow)  # type: ignore

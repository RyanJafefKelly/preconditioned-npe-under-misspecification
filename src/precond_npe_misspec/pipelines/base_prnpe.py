# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Callable, Any, Optional, Tuple
# from precond_npe_misspec.utils import distances as dist
# import equinox as eqx
# import jax
# import jax.numpy as jnp
# import flowjax.bijections as bij
# from flowjax.distributions import Normal, Transformed
# from flowjax.flows import coupling_flow
# from flowjax.train import fit_to_data


class ExperimentSpec:
    pass


class RunConfig:
    pass


class FlowConfig:
    pass


class RunResult:
    pass


def run_experiment(
    spec: ExperimentSpec, run: RunConfig, flow_cfg: FlowConfig
) -> RunResult:
    return RunResult()

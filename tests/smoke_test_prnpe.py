# quick_smoke_test_rnpe.py
import flowjax.bijections as bij
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from flowjax.distributions import Normal
from flowjax.experimental.numpyro import sample as fj_sample
from flowjax.flows import coupling_flow
from numpyro.infer import MCMC, NUTS


def student_t_denoiser(y_obs_w: jnp.ndarray, flow_s: object, scale: float = 0.05, df: float = 1.0) -> None:
    x = fj_sample("x", flow_s)

    numpyro.sample("y", dist.StudentT(df, loc=x, scale=scale), obs=y_obs_w)


def main() -> None:
    key = jax.random.key(0)
    s_dim = 5
    flow_s = coupling_flow(
        key,
        base_dist=Normal(jnp.zeros(s_dim)),
        transformer=bij.RationalQuadraticSpline(knots=8, interval=4.0),
        cond_dim=None,
        flow_layers=3,
        nn_width=64,
    )
    y_obs_w = jnp.array([0.5, -2.0, 0.0, 0.1, 3.0])

    mcmc = MCMC(NUTS(student_t_denoiser), num_warmup=200, num_samples=400)
    mcmc.run(key, y_obs_w=y_obs_w, flow_s=flow_s)
    print(mcmc.get_samples()["x"].shape)  # (400, 5)
    print(mcmc.print_summary())


if __name__ == "__main__":
    main()

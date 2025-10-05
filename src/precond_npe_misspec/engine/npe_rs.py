# src/precond_npe_misspec/engine/npe_rs.py
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import flowjax.bijections as bij
import jax
import jax.nn as jnn
import jax.numpy as jnp
from equinox.nn import MLP
from flowjax.bijections.utils import EmbedCondition
from flowjax.distributions import Normal, Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

from precond_npe_misspec.utils.mmd import median_heuristic_bandwidth, mmd2_vstat, rbf_kernel_matrix

# mypy: disallow-subclassing-any=False


type Array = jax.Array
EPS = 1e-8

Embedder = Callable[[Array], Array]
LossHistory = dict[str, list[float]]
Activation = Callable[[Array], Array]


def _relu(x: Array) -> Array:
    return jnn.relu(x)


def _tanh(x: Array) -> Array:
    return jnn.tanh(x)


_ACTIVATIONS: dict[str, Activation] = {
    "relu": _relu,
    "tanh": _tanh,
}


def _to_unconstrained(theta: Array, lo: Array, hi: Array) -> Array:
    p = (theta - lo) / (hi - lo)
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _from_unconstrained(u: Array, lo: Array, hi: Array) -> Array:
    return lo + (hi - lo) * jnn.sigmoid(u)


def _standardise(x: Array, m: Array, s: Array) -> Array:
    return (x - m) / (s + EPS)


class _BoundedPosterior(eqx.Module):
    base: Transformed
    u_mean: Array
    u_std: Array
    lo: Array
    hi: Array

    if TYPE_CHECKING:  # pragma: no cover - typing helper only

        def __init__(
            self,
            base: Transformed,
            u_mean: Array,
            u_std: Array,
            lo: Array,
            hi: Array,
        ) -> None: ...

    def sample(self, key: Array, shape: tuple[int, ...], *, condition: Array) -> Array:
        u_proc = self.base.sample(key, shape, condition=condition)
        u = u_proc * self.u_std + self.u_mean
        return _from_unconstrained(u, self.lo, self.hi)


def _whiten_stats(X: Array) -> tuple[Array, Array]:
    m = jnp.mean(X, axis=0)
    s = jnp.std(X, axis=0)
    return m, s


def _activation(name: str) -> Callable[[Array], Array]:
    try:
        return _ACTIVATIONS[name]
    except KeyError as err:
        raise ValueError(f"Unsupported activation: {name!r}") from err


def _build_posterior_with_embed(
    key: Array,
    *,
    theta_dim: int,
    # s_dim: int,
    # embed_dim: int,
    # embed_width: int,
    # embed_depth: int,
    # activation: str,
    raw_cond_shape: tuple[int, ...],
    npers_cfg: Any,
    spec: Any | None,
    flow_cfg: Any,
) -> Transformed:
    """Build q(theta | eta(s)) where eta is learned; accepts raw condition shape (s_dim,)."""
    k_flow, k_emb = jax.random.split(key)

    # Base flow with 'cond_dim=embed_dim'
    base = Normal(jnp.zeros(theta_dim))
    transformer = bij.RationalQuadraticSpline(knots=flow_cfg.knots, interval=flow_cfg.interval)
    flow = coupling_flow(
        key=k_flow,
        base_dist=base,
        transformer=transformer,
        cond_dim=int(npers_cfg.embed_dim),
        flow_layers=flow_cfg.flow_layers,
        nn_width=flow_cfg.nn_width,
    )

    # Embedder eta: R^{raw_cond_shape} -> R^{embed_dim}
    embedder: Embedder
    custom_builder = getattr(spec, "build_embedder", None) if spec is not None else None
    if callable(custom_builder):
        try:
            embedder = cast(
                Embedder,
                custom_builder(k_emb, int(npers_cfg.embed_dim), raw_cond_shape, npers_cfg),
            )
        except TypeError:
            # Fallback if builder does not accept npers_cfg
            embedder = cast(
                Embedder,
                custom_builder(k_emb, int(npers_cfg.embed_dim), raw_cond_shape),
            )
    else:
        # Default: simple MLP expecting 1D condition. Works for vector observations.
        in_size = int(raw_cond_shape[0]) if len(raw_cond_shape) == 1 else int(jnp.prod(jnp.array(raw_cond_shape)))

        def _maybe_flatten(z: Array) -> Array:
            return z if len(z.shape) == 1 else jnp.reshape(z, (in_size,))

        mlp = MLP(
            in_size=in_size,
            out_size=int(npers_cfg.embed_dim),
            width_size=int(npers_cfg.embed_width),
            depth=int(npers_cfg.embed_depth),
            activation=_activation(str(npers_cfg.activation)),
            key=k_emb,
        )

        class FlattenThenMLP(eqx.Module):
            mlp: MLP
            in_size: int

            def __init__(self, mlp: MLP, in_size: int):
                self.mlp = mlp
                self.in_size = in_size

            def __call__(self, z: Array) -> Array:
                z = z if len(z.shape) == 1 else jnp.reshape(z, (self.in_size,))
                return self.mlp(z)

        embedder = cast(Embedder, FlattenThenMLP(mlp, in_size))

    # Wrap bijection to accept raw condition shape (s_dim,) and embed internally.
    wrapped_bij = EmbedCondition(flow.bijection, embedding_net=embedder, raw_cond_shape=tuple(raw_cond_shape))
    # FlowJAX exposes Transformed as an eqx dataclass; cast keeps mypy happy.
    transformed_ctor = cast(Any, Transformed)
    return cast(Transformed, transformed_ctor(base_dist=base, bijection=wrapped_bij))


def _make_npers_loss(
    x_obs_w: Array,
    *,
    mmd_weight: float,
    bandwidth: float | str,
    mmd_subsample: int | None,
) -> Callable[[Any, Any, Array, Array, Array], Array]:
    """Return loss(params, static, theta_batch, s_batch_w, key) = NLL + λ·MMD²(η(x), η(x_obs))."""

    def loss(
        params: Any,
        static: Any,
        theta_b: Array,
        x_b_w: Array,
        key: Array,
    ) -> Array:
        dist: Transformed = eqx.combine(params, static)  # flow with EmbedCondition bijection
        # NLL term3
        nll = -dist.log_prob(theta_b, x_b_w).mean()

        # Embeddings
        embed = cast(Embedder, dist.bijection.embedding_net)

        # Batched embeddings: map embed over the leading batch axis of s_b_w.
        embed_batched = eqx.filter_vmap(embed)  # or: jax.vmap(lambda s: embed(s))
        Z = embed_batched(x_b_w)  # (B, d_emb)

        # Single observation embedding stays unbatched.
        z_star = embed(x_obs_w)  # (d_emb,)

        # Bandwidth per-batch
        # Bandwidth per-batch, computed on simulated embeddings ONLY (paper convention).
        if isinstance(bandwidth, str) and bandwidth == "median":
            ell = median_heuristic_bandwidth(Z, key=key)
        else:
            ell = jnp.asarray(bandwidth if not isinstance(bandwidth, str) else 1.0)

        # MMD²(Z, z*)

        def kmat(A: Array, B: Array) -> Array:
            return rbf_kernel_matrix(A, B, ell=ell)

        mmd2 = mmd2_vstat(
            Z,
            z_star,
            kernel_matrix=kmat,
            key=key,
            max_samples_X=mmd_subsample,
            max_samples_Y=None,
            drop_const_if_dirac=True,
        )
        return nll + float(mmd_weight) * mmd2

    return loss


def fit_posterior_flow_npe_rs(
    key: Array,
    spec: Any,
    theta_train: Array,  # (N, theta_dim)
    X_train: Array,  # (N, *x_shape), unwhitened
    x_obs: Array,
    flow_cfg: Any,
    npers_cfg: Any,
) -> tuple[eqx.Module, Array, Array, Array, Array, LossHistory]:
    """Train q(θ | η(s)) with NLL + λ·MMD²(η(x), η(x_obs))."""
    theta_dim = int(spec.theta_dim)
    has_bounds = (getattr(spec, "theta_lo", None) is not None) and (getattr(spec, "theta_hi", None) is not None)
    theta_train_proc = theta_train
    bounds_info: tuple[Array, Array, Array, Array] | None = None
    if has_bounds:
        lo = jnp.asarray(spec.theta_lo)
        hi = jnp.asarray(spec.theta_hi)
        u_train = _to_unconstrained(theta_train, lo, hi)
        u_mean = jnp.mean(u_train, axis=0)
        u_std = jnp.std(u_train, axis=0) + EPS
        theta_train_proc = _standardise(u_train, u_mean, u_std)
        bounds_info = (lo, hi, u_mean, u_std)
    # Whiten x
    if getattr(spec, "name", None) == "contaminated_weibull":
        # Use global stats so permutation-invariant encoders see exchangeable inputs
        X_mean = jnp.mean(X_train)
        X_std = jnp.std(X_train) + EPS
    else:
        X_mean, X_std = _whiten_stats(X_train)
        X_std = X_std + EPS
    X_train_w = (X_train - X_mean) / X_std
    x_obs_w = (x_obs - X_mean) / X_std
    print("X_train_w shape: ", X_train_w.shape)
    print("x_obs_w shape: ", x_obs_w.shape)
    print("max x: ", jnp.max(jnp.max(X_train_w)))
    print("max x obs: ", jnp.max(jnp.max(x_obs_w)))
    dtype = jnp.float32
    X_train_w = X_train_w.astype(dtype)
    x_obs_w = x_obs_w.astype(dtype)
    theta_train_proc = theta_train_proc.astype(dtype)
    if bounds_info is not None:
        lo, hi, u_mean, u_std = bounds_info
        bounds_info = (
            lo.astype(dtype),
            hi.astype(dtype),
            u_mean.astype(dtype),
            u_std.astype(dtype),
        )

    # Build q(theta | eta(x_w))
    dist0 = _build_posterior_with_embed(
        key,
        theta_dim=theta_dim,
        raw_cond_shape=tuple(X_train.shape[1:]),
        npers_cfg=npers_cfg,
        spec=spec,
        flow_cfg=flow_cfg,
    )

    # Optional warm-up with pure NLL
    dist_curr = dist0
    losses_acc: LossHistory | None = None
    warmup_epochs = int(getattr(npers_cfg, "warmup_epochs", 0) or 0)
    if warmup_epochs > 0:
        dist_curr, warm_losses = fit_to_data(
            key=key,
            dist=dist_curr,
            data=(theta_train_proc, X_train_w),
            learning_rate=flow_cfg.learning_rate,
            max_epochs=warmup_epochs,
            max_patience=0,
            batch_size=flow_cfg.batch_size,
            show_progress=True,
        )
        losses_acc = warm_losses

    # Full NPE-RS objective
    loss_fn = _make_npers_loss(
        x_obs_w,
        mmd_weight=float(npers_cfg.mmd_weight),
        bandwidth=getattr(npers_cfg, "bandwidth", "median"),
        mmd_subsample=getattr(npers_cfg, "mmd_subsample", None),
    )
    dist_fit, losses_main = fit_to_data(
        key=key,
        dist=dist_curr,
        data=(theta_train_proc, X_train_w),
        loss_fn=loss_fn,
        learning_rate=flow_cfg.learning_rate,
        max_epochs=flow_cfg.max_epochs,
        max_patience=flow_cfg.max_patience,
        batch_size=flow_cfg.batch_size,
        show_progress=True,
    )

    if losses_acc is not None:
        losses: LossHistory = {
            "train": list(losses_acc["train"]) + list(losses_main["train"]),
            "val": list(losses_acc["val"]) + list(losses_main["val"]),
        }
    else:
        losses = cast(LossHistory, losses_main)

    th_mean = jnp.mean(theta_train, axis=0)
    th_std = jnp.std(theta_train, axis=0)
    if bounds_info is not None:
        lo, hi, u_mean, u_std = bounds_info
        posterior_flow: eqx.Module = _BoundedPosterior(
            base=dist_fit,
            u_mean=u_mean,
            u_std=u_std,
            lo=lo,
            hi=hi,
        )
    else:
        posterior_flow = dist_fit
    return posterior_flow, X_mean, X_std, th_mean, th_std, losses

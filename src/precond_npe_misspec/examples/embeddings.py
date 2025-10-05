# embeddings.py
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from equinox.nn import MLP, Conv1d

type Array = jax.Array

if TYPE_CHECKING:

    class _EqxModule:  # pragma: no cover - typing alias for eqx.Module
        ...

else:
    _EqxModule = eqx.Module

EmbedBuilder = Callable[[Array, int, tuple[int, ...], Any], _EqxModule]

_REGISTRY: dict[str, EmbedBuilder] = {}


def register(name: str) -> Callable[[EmbedBuilder], EmbedBuilder]:
    def _wrap(fn: EmbedBuilder) -> EmbedBuilder:
        _REGISTRY[name] = fn
        return fn

    return _wrap


def build(name: str) -> EmbedBuilder:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown embedder '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


# ---- Helpers -----------------------------------------------------------------


def _ensure_bkt(x: Array, raw_shape: tuple[int, ...]) -> Array:
    """Accept (..., T) or (..., T, K) with any leading batch dims. Return (B, K, T)."""
    if len(raw_shape) == 1:
        T, K = int(raw_shape[0]), 1
    else:
        T, K = int(raw_shape[-2]), int(raw_shape[-1])
    B = int(x.size // (T * K))  # collapse all leading batch dims (incl. extra singleton)
    x_btk = jnp.reshape(x, (B, T, K))  # (B, T, K)
    return jnp.swapaxes(x_btk, -1, -2)  # (B, K, T)


def _global_avg_pool_time(x_bkt: Array) -> Array:
    return x_bkt.mean(axis=-1)  # (B,C)


def _causal_pad(x_bkt: Array, kernel: int, dilation: int) -> Array:
    pad = dilation * (kernel - 1)
    pad_width = [(0, 0), (0, 0), (pad, 0)]
    return jnp.pad(x_bkt, pad_width)


# ---- Embedders ---------------------------------------------------------------


@register("mlp_flat")
def mlp_flat(key: Array, embed_dim: int, raw_cond_shape: tuple[int, ...], cfg: Any) -> _EqxModule:
    in_size = int(jnp.prod(jnp.array(raw_cond_shape)))
    mlp = MLP(
        in_size=in_size,
        out_size=int(embed_dim),
        width_size=int(getattr(cfg, "embed_width", 128)),
        depth=int(getattr(cfg, "embed_depth", 2)),
        activation=(jnn.relu if getattr(cfg, "activation", "relu") == "relu" else jnn.tanh),
        key=key,
    )

    class _FlattenThenMLP(_EqxModule):
        mlp: MLP
        in_size: int

        def __init__(self, mlp: MLP, in_size: int):
            self.mlp = mlp
            self.in_size = in_size

        def __call__(self, x: Array) -> Array:
            # 1) force dtype to match MLP params (avoid float64/32 mismatch)
            x = jnp.asarray(x, dtype=jnp.float32)
            # 2) flatten ALL non-feature dims to a vector of length in_size
            x = jnp.reshape(x, (self.in_size,))
            # 3) run MLP and ensure rank‑1 (d_emb,)
            y = self.mlp(x)
            return jnp.reshape(y, (y.shape[-1],))

    return _FlattenThenMLP(mlp, in_size)


class _TCNBlock(_EqxModule):
    k: int
    d: int
    dil: Conv1d
    pw: Conv1d
    proj: Conv1d | None

    def __init__(self, key: Array, in_ch: int, ch: int, k: int, d: int):
        k1, k2, k3 = jax.random.split(key, 3)
        self.k, self.d = int(k), int(d)
        self.dil = Conv1d(in_ch, ch, kernel_size=k, dilation=d, use_bias=True, key=k1)
        self.pw = Conv1d(ch, ch, kernel_size=1, use_bias=True, key=k2)
        self.proj = None if in_ch == ch else Conv1d(in_ch, ch, kernel_size=1, use_bias=True, key=k3)

    def __call__(self, x_bkt: Array) -> Array:
        # x_bkt: (B, C_in, T)
        cdtype = self.dil.weight.dtype
        x_bkt = x_bkt.astype(cdtype)
        x_pad = _causal_pad(x_bkt, self.k, self.d)  # (B, C_in, Tpad)
        y = eqx.filter_vmap(self.dil)(x_pad)  # (B, C, T)
        y = jnn.relu(y)
        y = eqx.filter_vmap(self.pw)(y)  # (B, C, T)
        r = x_bkt if self.proj is None else eqx.filter_vmap(self.proj)(x_bkt)
        return jnn.relu(y + r)


class _SVAR_TCN(_EqxModule):
    blocks: tuple[_TCNBlock, ...]
    head: MLP
    raw_shape: tuple[int, ...]

    def __init__(
        self,
        key: Array,
        raw_shape: tuple[int, ...],
        embed_dim: int,
        ch: int,
        ksize: int,
        dilations: tuple[int, ...],
        head_width: int,
        head_depth: int,
    ):
        B = []
        k1, *ks = jax.random.split(key, 1 + len(dilations) * 3)
        in_ch = int(raw_shape[-1]) if len(raw_shape) == 2 else 1
        for i, d in enumerate(dilations):
            kblk = jax.random.fold_in(k1, i)
            B.append(_TCNBlock(kblk, in_ch if i == 0 else ch, ch, ksize, int(d)))
        self.blocks = tuple(B)
        self.head = MLP(
            in_size=ch,
            out_size=int(embed_dim),
            width_size=int(head_width),
            depth=int(head_depth),
            activation=jnn.relu,
            key=k1,
        )
        self.raw_shape = tuple(raw_shape)

    def __call__(self, x: Array) -> Array:
        is_single = x.ndim == len(self.raw_shape)  # e.g. (T,K)
        x_bkt = _ensure_bkt(x, self.raw_shape)  # (B, K, T)
        y = x_bkt
        for blk in self.blocks:
            y = blk(y)
        y = _global_avg_pool_time(y)  # (B, C)
        # Cast to head dtype and vmap MLP over batch.
        hdtype = self.head.layers[0].weight.dtype
        y = y.astype(hdtype)
        z = cast(Array, eqx.filter_vmap(self.head)(y))  # (B, d)
        return z[0] if is_single else z


@register("tcn_small")
def tcn_small(key: Array, embed_dim: int, raw_cond_shape: tuple[int, ...], cfg: Any) -> _EqxModule:
    ch = int(getattr(cfg, "tcn_channels", 32))
    ksize = int(getattr(cfg, "tcn_kernel", 5))
    dilations = tuple(getattr(cfg, "tcn_dilations", (1, 2, 4)))
    head_width = int(getattr(cfg, "embed_width", 128))
    head_depth = int(getattr(cfg, "embed_depth", 1))
    return _SVAR_TCN(
        key=key,
        raw_shape=tuple(raw_cond_shape),
        embed_dim=int(embed_dim),
        ch=ch,
        ksize=ksize,
        dilations=dilations,
        head_width=head_width,
        head_depth=head_depth,
    )


@register("asv_tcn")
def asv_tcn(key: Array, embed_dim: int, raw_cond_shape: tuple[int, ...], cfg: Any) -> _EqxModule:
    """Default embedder for alpha‑SV raw returns."""
    ch = int(getattr(cfg, "tcn_channels", 32))
    ksize = int(getattr(cfg, "tcn_kernel", 5))
    # Slightly deeper dilation stack for long‑memory in |z|
    dilations = tuple(getattr(cfg, "tcn_dilations", (1, 2, 4, 8)))
    head_width = int(getattr(cfg, "embed_width", 128))
    head_depth = int(getattr(cfg, "embed_depth", 1))
    return _SVAR_TCN(  # NOTE: turns out the network for SVAR pretty good for ASV...guess they both similar time-series
        key=key,
        raw_shape=tuple(raw_cond_shape),
        embed_dim=int(embed_dim),
        ch=ch,
        ksize=ksize,
        dilations=dilations,
        head_width=head_width,
        head_depth=head_depth,
    )


@register("iid_deepset")
def iid_deepset(key: Array, embed_dim: int, raw_cond_shape: tuple[int, ...], cfg: Any) -> _EqxModule:
    """
    DeepSets for iid 1D samples y[0:T):  φ: R->R^H, mean-pool over T,  ρ: R^H->R^{embed_dim}.
    Accepts x with shape (T,) or batched (..., T). Returns (embed_dim,) or batched (..., embed_dim).
    """
    # T = int(raw_cond_shape[-1])
    width = int(getattr(cfg, "embed_width", 128))
    depth = max(1, int(getattr(cfg, "embed_depth", 2)))
    k1, k2 = jax.random.split(key, 2)

    phi = MLP(
        in_size=1,
        out_size=width,
        width_size=width,
        depth=depth,
        activation=jnn.silu,
        key=k1,
    )
    rho = MLP(
        in_size=width,
        out_size=int(embed_dim),
        width_size=width,
        depth=depth,
        activation=jnn.silu,
        key=k2,
    )

    class _DeepSet1D(_EqxModule):
        phi: MLP
        rho: MLP
        raw_shape: tuple[int, ...]
        width: int

        def __init__(self, phi: MLP, rho: MLP, raw_shape: tuple[int, ...], width: int):
            self.phi = phi
            self.rho = rho
            self.raw_shape = tuple(raw_cond_shape)
            self.width = int(width)

        def __call__(self, x: Array) -> Array:
            x = jnp.asarray(x, dtype=jnp.float32)
            T = int(self.raw_shape[-1])
            B = int(x.size // T)
            x_bt = jnp.reshape(x, (B, T))

            # φ over elements
            xt = jnp.reshape(x_bt, (B * T, 1))  # (B*T, 1)
            Et = eqx.filter_vmap(self.phi)(xt)  # (B*T, H)
            E = jnp.reshape(Et, (B, T, self.width))  # (B, T, H)

            # mean pool over T
            Z = E.mean(axis=1)  # (B, H)

            # ρ head
            Zemb = eqx.filter_vmap(self.rho)(Z)  # (B, d_emb)
            return Zemb[0] if B == 1 else Zemb

    return _DeepSet1D(phi, rho, raw_cond_shape, width)


# @register("svar_ols_head")
# def svar_ols_head(
#     key: Any, embed_dim: int, raw_cond_shape: tuple[int, ...], cfg: Any
# ) -> _EqxModule:
#     import equinox as eqx
#     import jax
#     import jax.numpy as jnp
#     from equinox.nn import MLP

#     def _pairs(K: int) -> jax.Array:
#         # default SVAR pairs: (0,1),(2,3),(4,5) for K=6; generalizes if K even
#         return jnp.arange(K, dtype=jnp.int32).reshape(-1, 2)

#     width = int(getattr(cfg, "embed_width", 64))
#     depth = int(getattr(cfg, "embed_depth", 1))
#     head = MLP(
#         in_size=8,
#         out_size=int(embed_dim),
#         width_size=width,
#         depth=depth,
#         activation=jax.nn.relu,
#         key=key,
#     )

#     class _OLS(eqx.Module):
#         head: MLP
#         raw_shape: tuple[int, ...]

#         def __init__(self, head: MLP, raw_cond_shape: tuple[int, ...]) -> None:
#             self.head = head
#             self.raw_shape = tuple(raw_cond_shape)

#         def __call__(self, x: jax.Array) -> jax.Array:
#             # x ∈ ℝ^{T×K} or batched (...,T,K) → map to (B,d_emb)
#             x_bkt = _ensure_bkt(jnp.asarray(x, jnp.float32), self.raw_shape)  # (B,K,T)
#             Y_btk = jnp.swapaxes(x_bkt, -1, -2)  # (B,T,K)
#             K = int(self.raw_shape[-1] if len(self.raw_shape) == 2 else 1)

#             def _one(Y: jax.Array) -> jax.Array:
#                 T = Y.shape[0]
#                 Y0, Y1 = Y[:-1], Y[1:]  # (T-1,K)
#                 s00 = (Y0.T @ Y0) / (T - 1)  # (K,K)
#                 s10 = (Y1.T @ Y0) / (T - 1)  # (K,K)
#                 reg = 1e-4 * jnp.trace(s00) / K
#                 Xhat = s10 @ jnp.linalg.inv(s00 + reg * jnp.eye(K, dtype=Y.dtype))
#                 resid = Y1 - Y0 @ Xhat.T
#                 sigma = jnp.std(resid)

#                 pairs = _pairs(K)
#                 fwd = Xhat[pairs[:, 0], pairs[:, 1]]
#                 rev = Xhat[pairs[:, 1], pairs[:, 0]]
#                 ymean = jnp.mean(Y)
#                 stats = jnp.concatenate(
#                     [fwd, rev, jnp.array([sigma, ymean], Y.dtype)]
#                 )  # len=2*m+2=8 for K=6
#                 return stats

#             S = eqx.filter_vmap(_one)(Y_btk)  # (B, 8)
#             Z = eqx.filter_vmap(self.head)(S)  # (B, d_emb)
#             return Z[0] if (S.shape[0] == 1 and x.ndim == len(self.raw_shape)) else Z

#     return cast(_EqxModule, _OLS(head, raw_cond_shape))


# @register("svar_lagstats")
# def svar_lagstats(key, embed_dim, raw_cond_shape, cfg):
#     import equinox as eqx
#     import jax
#     import jax.numpy as jnp
#     from equinox.nn import MLP

#     def _pairs(K):
#         return jnp.arange(K, dtype=jnp.int32).reshape(-1, 2)

#     width = int(getattr(cfg, "embed_width", 64))
#     depth = int(getattr(cfg, "embed_depth", 1))
#     head = MLP(
#         in_size=8,
#         out_size=int(embed_dim),
#         width_size=width,
#         depth=depth,
#         activation=jax.nn.relu,
#         key=key,
#     )

#     class _LagStats(eqx.Module):
#         head: MLP
#         raw_shape: tuple[int, ...]

#         def __init__(self, head, raw_cond_shape):
#             self.head, self.raw_shape = head, tuple(raw_cond_shape)

#         def __call__(self, x):
#             x_bkt = _ensure_bkt(jnp.asarray(x, jnp.float32), self.raw_shape)  # (B,K,T)
#             Y_btk = jnp.swapaxes(x_bkt, -1, -2)  # (B,T,K)
#             K = int(self.raw_shape[-1] if len(self.raw_shape) == 2 else 1)
#             pairs = _pairs(K)

#             def _one(Y):
#                 C1 = (Y[1:].T @ Y[:-1]) / (Y.shape[0] - 1)  # (K,K)
#                 fwd = C1[pairs[:, 0], pairs[:, 1]]
#                 rev = C1[pairs[:, 1], pairs[:, 0]]
#                 s_sigma = jnp.std(Y)
#                 ymean = jnp.mean(Y)
#                 stats = jnp.concatenate(
#                     [fwd, rev, jnp.array([s_sigma, ymean], Y.dtype)]
#                 )  # len=8 for K=6
#                 return stats

#             S = eqx.filter_vmap(_one)(Y_btk)  # (B,8)
#             Z = eqx.filter_vmap(self.head)(S)  # (B,d_emb)
#             return Z[0] if (S.shape[0] == 1 and x.ndim == len(self.raw_shape)) else Z

#     return _LagStats(head, raw_cond_shape)


# class _FixedLinear(eqx.Module):
#     W: jax.Array = eqx.field(static=True)
#     b: jax.Array | None = eqx.field(static=True)

#     def __init__(
#         self, key: jax.Array, in_dim: int, out_dim: int, use_bias: bool = False
#     ):
#         k1, k2 = jax.random.split(key)
#         # Heuristically scale to keep magnitudes stable.
#         W = jax.random.normal(k1, (in_dim, out_dim), dtype=jnp.float32) / jnp.sqrt(
#             in_dim
#         )
#         self.W = W
#         self.b = jnp.zeros((out_dim,), dtype=jnp.float32) if use_bias else None

#     def __call__(self, x: jax.Array) -> jax.Array:
#         y = jnp.matmul(x, self.W)
#         return y if self.b is None else y + cast(jax.Array, self.b)


# def _lagstats_batched(x_bkt: jax.Array, pairs: jax.Array) -> jax.Array:
#     """x_bkt: (B,K,T) -> stats (B, 2m+2). Matches svar.summaries structure."""
#     B, K, T = x_bkt.shape
#     i, j = pairs[:, 0], pairs[:, 1]  # (m,)
#     A = (x_bkt[:, i, 1:] * x_bkt[:, j, :-1]).sum(axis=-1)  # (B,m)
#     Bdir = (x_bkt[:, j, 1:] * x_bkt[:, i, :-1]).sum(axis=-1)  # (B,m)
#     sdir = jnp.concatenate([A, Bdir], axis=-1) / jnp.asarray(T, jnp.float32)
#     pooled = x_bkt.reshape(B, -1)
#     s_sigma = jnp.std(pooled, axis=-1, ddof=0)  # (B,)
#     s_mu = jnp.mean(pooled, axis=-1)  # (B,)
#     return jnp.concatenate([sdir, s_sigma[:, None], s_mu[:, None]], axis=-1)  # (B,2m+2)


# @register("svar_lagstats_cpu")
# def svar_lagstats_cpu(
#     key: jax.Array, embed_dim: int, raw_cond_shape: tuple[int, ...], cfg: Any
# ) -> _EqxModule:
#     """
#     CPU‑cheap embedder. Computes lag‑1 directed cross‑products for default SVAR pairs,
#     plus pooled std and mean. Optional fixed linear projection to embed_dim.
#     """

#     # Accept (..., T, K) or (..., T) → (B,K,T)
#     def _to_bkt(x: jax.Array) -> jax.Array:
#         if len(raw_cond_shape) == 1:
#             T, K = int(raw_cond_shape[0]), 1
#         else:
#             T, K = int(raw_cond_shape[-2]), int(raw_cond_shape[-1])
#         B = int(x.size // (T * K))
#         return jnp.swapaxes(jnp.reshape(x, (B, T, K)), -1, -2).astype(jnp.float32)

#     pairs = default_pairs(
#         int(raw_cond_shape[-1]) if len(raw_cond_shape) == 2 else 6
#     )  # assumes k=6 if 1D

#     # Output size before projection
#     m = int(pairs.shape[0])
#     out0 = 2 * m + 2

#     proj = (
#         None
#         if embed_dim == out0
#         else _FixedLinear(key, out0, embed_dim, use_bias=False)
#     )

#     class _SVARLagStatsCPU(_EqxModule):
#         pairs: jax.Array
#         proj: _FixedLinear | None
#         raw_shape: tuple[int, ...]

#         def __init__(
#             self,
#             pairs: jax.Array,
#             proj: _FixedLinear | None,
#             raw_shape: tuple[int, ...],
#         ):
#             self.pairs = pairs
#             self.proj = proj
#             self.raw_shape = tuple(raw_shape)

#         def __call__(self, x: jax.Array) -> jax.Array:
#             xb = _to_bkt(x)  # (B,K,T)
#             z = _lagstats_batched(xb, self.pairs)  # (B, out0)
#             if self.proj is not None:
#                 z = jax.vmap(self.proj)(z)
#             return z[0] if z.shape[0] == 1 else z

#     return _SVARLagStatsCPU(pairs, proj, tuple(raw_cond_shape))


# @register("svar_lagstats_sketch")
# def svar_lagstats_sketch(
#     key: jax.Array, embed_dim: int, raw_cond_shape: tuple[int, ...], cfg: Any
# ) -> _EqxModule:
#     M = int(getattr(cfg, "sketch_M", 128))

#     def _to_bkt(x: jax.Array) -> jax.Array:
#         if len(raw_cond_shape) == 1:
#             T, K = int(raw_cond_shape[0]), 1
#         else:
#             T, K = int(raw_cond_shape[-2]), int(raw_cond_shape[-1])
#         B = int(x.size // (T * K))
#         return jnp.swapaxes(jnp.reshape(x, (B, T, K)), -1, -2).astype(jnp.float32)

#     # Shapes
#     if len(raw_cond_shape) == 1:
#         T, K = int(raw_cond_shape[0]), 1
#     else:
#         T, K = int(raw_cond_shape[-2]), int(raw_cond_shape[-1])

#     pairs = default_pairs(K if K else 6)

#     # Fixed subsample of time indices in [1, T-1] to respect lag‑1
#     key_idx = jax.random.fold_in(key, 17)
#     idx1 = jax.random.choice(
#         key_idx,
#         jnp.arange(1, T, dtype=jnp.int32),
#         shape=(min(M, max(T - 1, 1)),),
#         replace=False,
#     )
#     idx0 = idx1 - 1
#     M_eff = idx1.shape[0]
#     m = int(pairs.shape[0])
#     out0 = 2 * m + 2

#     proj = (
#         None
#         if embed_dim == out0
#         else _FixedLinear(key, out0, embed_dim, use_bias=False)
#     )

#     class _SVARLagStatsSketch(_EqxModule):
#         pairs: jax.Array
#         idx0: jax.Array
#         idx1: jax.Array
#         proj: _FixedLinear | None

#         def __init__(
#             self,
#             pairs: jax.Array,
#             idx0: jax.Array,
#             idx1: jax.Array,
#             proj: _FixedLinear | None,
#         ):
#             self.pairs = pairs
#             self.idx0 = idx0
#             self.idx1 = idx1
#             self.proj = proj

#         def __call__(self, x: jax.Array) -> jax.Array:
#             xb = _to_bkt(x)  # (B,K,T)
#             B, K, _ = xb.shape
#             i, j = self.pairs[:, 0], self.pairs[:, 1]  # (m,)

#             # Gather (vectorized) only at idx0/idx1
#             x_i_1 = xb[:, i[:, None], self.idx1[None, :]]  # (B,m,M)
#             x_j_0 = xb[:, j[:, None], self.idx0[None, :]]  # (B,m,M)
#             x_j_1 = xb[:, j[:, None], self.idx1[None, :]]
#             x_i_0 = xb[:, i[:, None], self.idx0[None, :]]

#             A = (x_i_1 * x_j_0).sum(axis=-1) / jnp.asarray(M_eff, jnp.float32)  # (B,m)
#             Bdir = (x_j_1 * x_i_0).sum(axis=-1) / jnp.asarray(
#                 M_eff, jnp.float32
#             )  # (B,m)

#             # Cheap pooled scale + mean on the SAME subsample for maximal speed
#             flat1 = xb[:, :, self.idx1].reshape(B, -1)
#             s_sigma = jnp.std(flat1, axis=-1, ddof=0)  # (B,)
#             s_mu = jnp.mean(flat1, axis=-1)  # (B,)

#             z = jnp.concatenate(
#                 [A, Bdir, s_sigma[:, None], s_mu[:, None]], axis=-1
#             )  # (B, out0)
#             if self.proj is not None:
#                 z = jax.vmap(self.proj)(z)
#             return z[0] if z.shape[0] == 1 else z

#     return _SVARLagStatsSketch(pairs, idx0, idx1, proj)

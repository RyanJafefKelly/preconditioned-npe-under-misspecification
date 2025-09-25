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

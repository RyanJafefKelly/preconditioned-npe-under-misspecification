from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

type Array = jax.Array
DistanceFn = Callable[[Array, Array], Array]  # (N,d),(d,) -> (N,)


def euclidean(Sw: Array, s_obs_w: Array) -> Array:
    return jnp.array(jnp.linalg.norm(Sw - s_obs_w, axis=-1))


def l1(Sw: Array, s_obs_w: Array) -> Array:
    return jnp.array(jnp.sum(jnp.abs(Sw - s_obs_w), axis=-1))


def rbf_kernel(x: Array, y: Array, ell: float) -> Array:
    """RBF(x,y) with broadcasting. x: (..., d), y: (..., d) or (d,)."""
    diff = x - y
    sq = jnp.sum(diff * diff, axis=-1)
    return jnp.exp(-sq / (2.0 * ell * ell))


def median_heuristic(X: Array, batch: int = 1000) -> float:
    """Median pairwise ℓ2 distance heuristic for bandwidth; X: (n,d)."""
    n = X.shape[0]
    nb = (n + batch - 1) // batch

    def block(i: int) -> Array:
        sl = slice(i * batch, min((i + 1) * batch, n))
        a = X[sl, None, :]  # (b,1,d)
        d2 = jnp.sum((a - X[None, :, :]) ** 2, axis=-1)  # (b,n)
        return jnp.sqrt(d2).ravel()

    dists = jnp.concatenate([block(i) for i in range(nb)])
    # For whitened data, a final / sqrt(2) is common; here use / sqrt(2) via /2 under sqrt.
    return float(jnp.sqrt(jnp.median(dists) / 2.0) + 1e-12)


def mmd_rbf_factory(
    ell: float, *, sqrt: bool = True, unbiased: bool = False
) -> DistanceFn:
    """
    Returns f(Sw, s_obs_w) -> (N,)
    - If Sw is (N,d): uses m=1 replicate per θ: MMD^2 = 2(1 - k(x,y)).
    - If Sw is (N,R,d): empirical P has m=R replicates per θ.
      Biased:  k_xx = mean_{r,r'} k(x_r,x_{r'}), k_yy=1, k_xy = mean_r k(x_r,y).
      Unbiased: remove r==r' from k_xx average.
    """
    ell = float(ell)

    def f(Sw: Array, s_obs_w: Array) -> Array:
        if Sw.ndim == 2:  # (N,d)
            kxy = rbf_kernel(Sw, s_obs_w, ell)  # (N,)
            mmd2 = 2.0 * (1.0 - kxy)
            return jnp.sqrt(jnp.maximum(mmd2, 0.0)) if sqrt else mmd2

        # (N,R,d)
        N, R, _ = Sw.shape
        y = s_obs_w
        # k_xy: (N,R)
        k_xy = rbf_kernel(Sw, y, ell)
        k_xy_mean = jnp.mean(k_xy, axis=1)  # (N,)

        # k_xx: (N,R,R)
        Sw1 = Sw[:, :, None, :]  # (N,R,1,d)
        Sw2 = Sw[:, None, :, :]  # (N,1,R,d)
        k_xx = rbf_kernel(Sw1, Sw2, ell)  # (N,R,R)

        if unbiased and R > 1:
            # remove diagonal, average over R*(R-1)
            s = jnp.sum(k_xx, axis=(1, 2)) - jnp.sum(
                jnp.diagonal(k_xx, axis1=1, axis2=2)
            )
            k_xx_mean = s / (R * (R - 1))
        else:
            k_xx_mean = jnp.mean(k_xx, axis=(1, 2))

        mmd2 = k_xx_mean + 1.0 - 2.0 * k_xy_mean  # (N,)
        return jnp.sqrt(jnp.maximum(mmd2, 0.0)) if sqrt else mmd2

    return f


def mmd_rbf_with_median(
    S_tr_w: Array, *, sqrt: bool = True, unbiased: bool = False
) -> DistanceFn:
    return mmd_rbf_factory(median_heuristic(S_tr_w), sqrt=sqrt, unbiased=unbiased)

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

type Array = jax.Array
KernelMatrixFn = Callable[[Array, Array], Array]  # (n,d),(m,d) -> (n,m)


# ------------------------
# Kernel helpers
# ------------------------
def _ensure_2d(a: Array) -> Array:
    """Ensure shape (n,d)."""
    return a[None, :] if a.ndim == 1 else a


def _pairwise_sq_dists(X: Array, Y: Array) -> Array:
    """Return ||x_i - y_j||^2 for X:(n,d), Y:(m,d)."""
    X = _ensure_2d(jnp.asarray(X))
    Y = _ensure_2d(jnp.asarray(Y))
    X2 = jnp.sum(X * X, axis=1, keepdims=True)  # (n,1)
    Y2 = jnp.sum(Y * Y, axis=1)[None, :]  # (1,m)
    d2 = X2 + Y2 - 2.0 * (X @ Y.T)  # (n,m)
    return jnp.maximum(d2, 0.0)


def rbf_kernel_matrix(X: Array, Y: Array, *, ell: Array) -> Array:
    """RBF kernel k(x,y)=exp(-||x-y||^2 / (2 ell^2)) as an (n,m) matrix)."""
    ell = jnp.asarray(ell, dtype=X.dtype)
    d2 = _pairwise_sq_dists(X, Y)
    return jnp.exp(-0.5 * d2 / (jnp.square(ell) + 1e-12))


def median_heuristic_bandwidth(
    X: Array,
    *,
    key: Array | None = None,
    max_samples: int | None = None,
    min_ell: float = 1e-6,
) -> Array:
    """
    Median heuristic for RBF lengthscale on X:(n,d).
    - Uses off‑diagonal pairwise distances via integer upper‑tri indices (JIT‑safe).
    - Our RBF is exp(-||x-y||^2 / (2 ell^2)); paper uses exp(-||s-s'||^2 / β^2).
      With β = sqrt(median/2), ell = β / sqrt(2) = sqrt(median/4).
    """
    X = _ensure_2d(jnp.asarray(X))
    n = X.shape[0]
    if n <= 1:
        return jnp.asarray(min_ell, dtype=X.dtype)

    # Optional point subsampling first (keeps shapes static in the step)
    if max_samples is not None and n > max_samples:
        if key is None:
            idx = jnp.arange(n)[:max_samples]
        else:
            idx = jr.choice(key, n, (max_samples,), replace=False)
        X = X[idx]
        n = X.shape[0]

    # Pairwise off‑diagonal distances (no boolean masking)
    d2 = _pairwise_sq_dists(X, X)  # (n,n)
    ii, jj = jnp.triu_indices(n, k=1)  # integer indices (size n(n-1)/2)
    vals = d2[ii, jj]  # (n(n-1)/2,)

    med_d2 = jnp.median(vals) if vals.size > 0 else jnp.asarray(1.0, dtype=X.dtype)
    ell = jnp.sqrt(0.25 * med_d2)  # paper convention
    return jnp.maximum(ell, min_ell)


# ------------------------
# MMD estimators
# ------------------------
def _u_statistic_offdiag(K: Array) -> Array:
    """Unbiased mean of off-diagonal kernel values for K:(n,n)."""
    n = K.shape[0]
    if n < 2:
        return jnp.array(0.0)
    sum_all = jnp.sum(K)
    sum_diag = jnp.trace(K)
    return (sum_all - sum_diag) / (n * (n - 1))


def mmd2_unbiased(
    X: Array,
    Y: Array,
    *,
    kernel_matrix: KernelMatrixFn,
    key: Array | None = None,
    max_samples_X: int | None = None,
    max_samples_Y: int | None = None,
) -> Array:
    """
    Unbiased MMD^2 between empirical samples X:(n,d) and Y:(m,d).
    Handles m=1 (term_yy=0 under unbiased estimator).
    Optional subsampling caps quadratic costs before forming kernel matrices.
    """
    X = _ensure_2d(jnp.asarray(X))
    Y = _ensure_2d(jnp.asarray(Y))

    if max_samples_X is not None and X.shape[0] > max_samples_X:
        idx = jnp.arange(X.shape[0]) if key is None else jr.choice(key, X.shape[0], (max_samples_X,), replace=False)
        X = X[idx if key is not None else idx[:max_samples_X]]

    if max_samples_Y is not None and Y.shape[0] > max_samples_Y:
        if key is None:
            Y = Y[:max_samples_Y]
        else:
            key, sub = jr.split(key)
            idx = jr.choice(sub, Y.shape[0], (max_samples_Y,), replace=False)
            Y = Y[idx]

    Kxx = kernel_matrix(X, X)  # (n,n)
    Kyy = kernel_matrix(Y, Y) if Y.shape[0] > 1 else None  # (m,m) or None
    Kxy = kernel_matrix(X, Y)  # (n,m)

    term_xx = _u_statistic_offdiag(Kxx)
    term_yy = _u_statistic_offdiag(Kyy) if Kyy is not None else jnp.array(0.0)
    term_xy = jnp.mean(Kxy)
    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return jnp.maximum(mmd2, 0.0)


def mmd2_unbiased_batched(
    Xs: Array,
    Ys: Array,
    *,
    kernel_matrix: KernelMatrixFn,
    key: Array | None = None,
    max_samples_X: int | None = None,
    max_samples_Y: int | None = None,
) -> Array:
    """
    Batched unbiased MMD^2.
    - Xs: (B,n,d)
    - Ys: (B,m,d) or (m,d) or (d,)
    Returns: (B,)
    """
    Xs = jnp.asarray(Xs)
    B = Xs.shape[0]

    if Ys.ndim == 1:  # (d,)
        Y_fixed = Ys

        def _single_with_key_1(key_i: Array, X_i: Array) -> Array:
            return mmd2_unbiased(
                X_i,
                Y_fixed,
                kernel_matrix=kernel_matrix,
                key=key_i,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
            )

        def _single_no_key_1(X_i: Array) -> Array:
            return mmd2_unbiased(
                X_i,
                Y_fixed,
                kernel_matrix=kernel_matrix,
                key=None,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
            )

        if key is None:
            return jax.vmap(_single_no_key_1)(Xs)
        keys = jr.split(key, B)
        return jax.vmap(_single_with_key_1)(keys, Xs)

    if Ys.ndim == 2:  # (m,d)
        Y_shared = Ys

        def _single_with_key_2(key_i: Array, X_i: Array) -> Array:
            return mmd2_unbiased(
                X_i,
                Y_shared,
                kernel_matrix=kernel_matrix,
                key=key_i,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
            )

        def _single_no_key_2(X_i: Array) -> Array:
            return mmd2_unbiased(
                X_i,
                Y_shared,
                kernel_matrix=kernel_matrix,
                key=None,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
            )

        if key is None:
            return jax.vmap(_single_no_key_2)(Xs)
        keys = jr.split(key, B)
        return jax.vmap(_single_with_key_2)(keys, Xs)

    if Ys.ndim == 3:  # (B,m,d)

        def _single_with_key_3(key_i: Array, X_i: Array, Y_i: Array) -> Array:
            return mmd2_unbiased(
                X_i,
                Y_i,
                kernel_matrix=kernel_matrix,
                key=key_i,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
            )

        def _single_no_key_3(X_i: Array, Y_i: Array) -> Array:
            return mmd2_unbiased(
                X_i,
                Y_i,
                kernel_matrix=kernel_matrix,
                key=None,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
            )

        if key is None:
            return jax.vmap(_single_no_key_3)(Xs, Ys)
        keys = jr.split(key, B)
        return jax.vmap(_single_with_key_3)(keys, Xs, Ys)

    raise ValueError(f"Unsupported Ys.ndim={Ys.ndim}; expected 1, 2, or 3.")


def mmd2_vstat(
    X: Array,
    Y: Array,
    *,
    kernel_matrix: KernelMatrixFn,
    key: Array | None = None,
    max_samples_X: int | None = None,
    max_samples_Y: int | None = None,
    drop_const_if_dirac: bool = True,
) -> Array:
    """
    V-statistic MMD^2:
      mean(Kxx) - 2 mean(Kxy) + mean(Kyy).
    If Y is a single point and drop_const_if_dirac=True, we drop mean(Kyy)=k(y,y)
    since it is constant w.r.t. parameters (useful in training).
    """
    X = _ensure_2d(jnp.asarray(X))
    Y = _ensure_2d(jnp.asarray(Y))

    if max_samples_X is not None and X.shape[0] > max_samples_X:
        if key is None:
            X = X[:max_samples_X]
        else:
            key, sub = jr.split(key)
            idx = jr.choice(sub, X.shape[0], (max_samples_X,), replace=False)
            X = X[idx]

    if max_samples_Y is not None and Y.shape[0] > max_samples_Y:
        if key is None:
            Y = Y[:max_samples_Y]
        else:
            key, sub = jr.split(key)
            idx = jr.choice(sub, Y.shape[0], (max_samples_Y,), replace=False)
            Y = Y[idx]

    Kxx = kernel_matrix(X, X)
    Kxy = kernel_matrix(X, Y)
    Kyy = kernel_matrix(Y, Y)

    term_xx = jnp.mean(Kxx)
    term_xy = jnp.mean(Kxy)
    if drop_const_if_dirac and Y.shape[0] == 1:
        term_yy = jnp.array(0.0)
    else:
        term_yy = jnp.mean(Kyy)
    mmd2 = term_xx - 2.0 * term_xy + term_yy
    return jnp.maximum(mmd2, 0.0)


def mmd2_vstat_batched(
    Xs: Array,
    Ys: Array,
    *,
    kernel_matrix: KernelMatrixFn,
    key: Array | None = None,
    max_samples_X: int | None = None,
    max_samples_Y: int | None = None,
    drop_const_if_dirac: bool = True,
) -> Array:
    """
    Batched V-statistic MMD^2.
      - Xs: (B,n,d)
      - Ys: (B,m,d) or (m,d) or (d,)
    Returns: (B,)
    """
    Xs = jnp.asarray(Xs)
    B = Xs.shape[0]

    if Ys.ndim == 1:  # (d,)
        y_fixed = Ys

        def _one_with_key_1(k: Array, X: Array) -> Array:
            return mmd2_vstat(
                X,
                y_fixed,
                kernel_matrix=kernel_matrix,
                key=k,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
                drop_const_if_dirac=drop_const_if_dirac,
            )

        def _one_no_key_1(X: Array) -> Array:
            return mmd2_vstat(
                X,
                y_fixed,
                kernel_matrix=kernel_matrix,
                key=None,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
                drop_const_if_dirac=drop_const_if_dirac,
            )

        if key is None:
            return jax.vmap(_one_no_key_1)(Xs)
        keys = jr.split(key, B)
        return jax.vmap(_one_with_key_1)(keys, Xs)

    if Ys.ndim == 2:  # (m,d), shared across batch
        Y_shared = Ys

        def _one_with_key_2(k: Array, X: Array) -> Array:
            return mmd2_vstat(
                X,
                Y_shared,
                kernel_matrix=kernel_matrix,
                key=k,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
                drop_const_if_dirac=drop_const_if_dirac,
            )

        def _one_no_key_2(X: Array) -> Array:
            return mmd2_vstat(
                X,
                Y_shared,
                kernel_matrix=kernel_matrix,
                key=None,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
                drop_const_if_dirac=drop_const_if_dirac,
            )

        if key is None:
            return jax.vmap(_one_no_key_2)(Xs)
        keys = jr.split(key, B)
        return jax.vmap(_one_with_key_2)(keys, Xs)

    if Ys.ndim == 3:  # (B,m,d)

        def _one_with_key_3(k: Array, X: Array, Y: Array) -> Array:
            return mmd2_vstat(
                X,
                Y,
                kernel_matrix=kernel_matrix,
                key=k,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
                drop_const_if_dirac=drop_const_if_dirac,
            )

        def _one_no_key_3(X: Array, Y: Array) -> Array:
            return mmd2_vstat(
                X,
                Y,
                kernel_matrix=kernel_matrix,
                key=None,
                max_samples_X=max_samples_X,
                max_samples_Y=max_samples_Y,
                drop_const_if_dirac=drop_const_if_dirac,
            )

        if key is None:
            return jax.vmap(_one_no_key_3)(Xs, Ys)
        keys = jr.split(key, B)
        return jax.vmap(_one_with_key_3)(keys, Xs, Ys)

    raise ValueError(f"Unsupported Ys.ndim={Ys.ndim}; expected 1, 2, or 3.")

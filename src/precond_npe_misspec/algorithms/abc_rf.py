# src/precond_npe_misspec/algorithms/abc_rf.py
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _fit_rf_multi(S_np: np.ndarray, th_np: np.ndarray, cfg: Any) -> tuple[Any, float]:
    rf = RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        min_samples_leaf=cfg.rf_min_leaf,
        max_depth=cfg.rf_max_depth,
        bootstrap=True,
        oob_score=True,
        max_features=1.0,
        n_jobs=cfg.rf_n_jobs,
        random_state=cfg.rf_random_state,
        min_samples_split=2 * cfg.rf_min_leaf,  # avoid splits that create leaves smaller than min_samples_leaf
        min_impurity_decrease=1e-6,
    )
    rf.fit(S_np, th_np)  # multi‑output
    return rf, float(getattr(rf, "oob_score_", np.nan))


def _fit_rf_per_param(S_np: np.ndarray, th_np: np.ndarray, cfg: Any) -> tuple[list[Any], float]:
    models, oobs = [], []
    for j in range(th_np.shape[1]):
        rf = RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            min_samples_leaf=cfg.rf_min_leaf,
            max_depth=cfg.rf_max_depth,
            bootstrap=True,
            oob_score=True,
            max_features=1.0,
            n_jobs=cfg.rf_n_jobs,
            random_state=cfg.rf_random_state + j,
            min_samples_split=2 * cfg.rf_min_leaf,  # avoid splits that create leaves smaller than min_samples_leaf
            min_impurity_decrease=1e-6,
        )
        rf.fit(S_np, th_np[:, j])
        models.append(rf)
        oobs.append(float(getattr(rf, "oob_score_", np.nan)))
    return models, float(np.nanmean(oobs))


def _qrf_weights_multi(rf: RandomForestRegressor, S_np: np.ndarray, s_obs_np: np.ndarray) -> np.ndarray:
    B = len(rf.estimators_)
    w = np.zeros(S_np.shape[0], dtype=np.float64)
    obs_leaf = rf.apply(s_obs_np[None, :]).ravel()  # (B,)
    for b, tree in enumerate(rf.estimators_):
        leaves = tree.apply(S_np)  # (N,)
        mask = leaves == obs_leaf[b]
        nb = int(mask.sum())
        if nb > 0:
            w[mask] += 1.0 / (B * nb)
    w /= w.sum() + 1e-12
    return w


def _qrf_weights_per_param(models: list[Any], S_np: np.ndarray, s_obs_np: np.ndarray) -> np.ndarray:
    B = sum(len(m.estimators_) for m in models)
    w = np.zeros(S_np.shape[0], dtype=np.float64)
    for rf in models:
        obs_leaf = rf.apply(s_obs_np[None, :]).ravel()
        for b, tree in enumerate(rf.estimators_):
            leaves = tree.apply(S_np)
            mask = leaves == obs_leaf[b]
            nb = int(mask.sum())
            if nb > 0:
                w[mask] += 1.0 / (B * nb)
    w /= w.sum() + 1e-12
    return w


def abc_rf_select(
    *,
    S: jnp.ndarray,
    theta: jnp.ndarray,
    s_obs: jnp.ndarray,
    cfg: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, float | np.ndarray]]:
    """Train RF(s) on (S→θ), compute QRF weights at s_obs, and subsample."""
    # to numpy (float32 is fine; trees handle it)
    S_np = np.asarray(S, dtype=np.float32)
    th_np = np.asarray(theta, dtype=np.float32)
    s_obs_np = np.asarray(s_obs, dtype=np.float32)

    # Optional subsample for fitting
    rng = np.random.default_rng(cfg.rf_random_state)
    n = S_np.shape[0]
    m_fit = int(np.ceil(cfg.rf_train_frac * n))
    idx_fit = rng.choice(n, size=m_fit, replace=True) if m_fit < n else np.arange(n)
    S_fit, th_fit = S_np[idx_fit], th_np[idx_fit]

    # Fit
    if cfg.abc_rf_mode == "per_param":
        models, oob = _fit_rf_per_param(S_fit, th_fit, cfg)
        w = _qrf_weights_per_param(models, S_np, s_obs_np)
    else:
        rf, oob = _fit_rf_multi(S_fit, th_fit, cfg)
        w = _qrf_weights_multi(rf, S_np, s_obs_np)

    return (
        theta,
        S,
        {
            "oob_r2": float(oob),
            "n_fit": float(len(idx_fit)),
            "n_total": float(n),
            "sum_w": float(w.sum()),
            "weights": w,
        },
    )

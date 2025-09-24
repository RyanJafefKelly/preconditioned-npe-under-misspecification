from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd


def load_sp500_returns_yahoo(
    start: str = "2013-01-02",
    end: str = "2017-02-07",
    field: str = "Close",
    log_returns: bool = True,
    standardise: bool = False,
) -> jnp.ndarray:
    """Load cached S&P 500 (^GSPC) returns and return them as a JAX array."""

    data_path = Path(__file__).with_name("sp500_daily_log_returns.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Cached S&P 500 data not found at {data_path}.")

    df = pd.read_csv(
        data_path,
        header=None,
        names=["idx", "log_return"],
        usecols=["log_return"],
    )
    log_r = df["log_return"].to_numpy(dtype=np.float64)

    if log_returns:
        r = log_r
    else:
        r = np.expm1(log_r)

    if standardise:
        mu, sd = r.mean(), r.std(ddof=0)
        if sd == 0:
            raise RuntimeError("Zero variance in returns.")
        r = (r - mu) / sd

    return jnp.asarray(r, dtype=jnp.float32)


def load_sp500_returns_yahoo_download(
    start: str = "2013-01-02",
    end: str = "2017-02-07",
    field: str = "Close",
    log_returns: bool = True,
    standardise: bool = False,
) -> jnp.ndarray:
    """Download S&P 500 (^GSPC) and return returns as a JAX array."""
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not available; cannot load ^GSPC.") from e

    df = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned for ^GSPC.")

    px = df[field].astype("float64").to_numpy().ravel()
    r = np.diff(np.log(px)) if log_returns else np.diff(px) / px[:-1]
    if standardise:
        mu, sd = r.mean(), r.std(ddof=0)
        if sd == 0:
            raise RuntimeError("Zero variance in returns.")
        r = (r - mu) / sd
    return jnp.asarray(r, dtype=jnp.float32)  # TODO? assume float32? Let be arg?

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def load_sp500_returns_yahoo(
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

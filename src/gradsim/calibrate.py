"""Estimate GBM parameters (mu, sigma) from historical prices.

If log returns r_k = ln(S_k / S_{k-1}) are i.i.d. N((mu - sigma^2/2) dt, sigma^2 dt), then
    sigma_hat = std(r) / sqrt(dt)
    mu_hat    = mean(r) / dt + sigma_hat^2 / 2
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_gbm_params(prices: pd.Series | np.ndarray, periods_per_year: float = 12) -> dict[str, float]:
    p = np.asarray(prices, dtype=float)
    r = np.diff(np.log(p))
    dt = 1.0 / periods_per_year
    sigma = r.std(ddof=1) / np.sqrt(dt)
    mu = r.mean() / dt + 0.5 * sigma**2
    # Standard error of mu_hat is ~ sigma / sqrt(total years): drift is estimated poorly.
    years = r.size * dt
    return {"mu": float(mu), "sigma": float(sigma), "mu_se": float(sigma / np.sqrt(years)),
            "n_obs": int(r.size), "years": float(years)}


def fetch_prices(ticker: str = "^SP500TR", start: str = "1990-01-01", interval: str = "1mo") -> pd.Series:
    """Download adjusted closes with yfinance (optional dependency).

    ^SP500TR is the S&P 500 *total return* index (dividends reinvested) — the
    right choice for an investor who reinvests. ^GSPC is price-only.
    """
    import yfinance as yf  # imported lazily so the rest of the package works offline

    df = yf.download(ticker, start=start, interval=interval, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()

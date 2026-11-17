"""Real S&P 500 data: loaders, GBM diagnostics, historical backtest, block bootstrap.

Two sources, both stored in data/ (committed; data/raw/ is gitignored) so results are reproducible offline:

1. Shiller (Yale) monthly S&P Composite, 1871-, via github.com/datasets/s-and-p-500.
   Price = *monthly average of daily closes*; Dividend = annualised (4-quarter total).
   Dividends end 2023-06 (later rows are price-only, Dividend = 0), so the total-return
   series stops there. Monthly total return:  (P_t + D_t / 12) / P_{t-1}.
   Caveat: averaging prices smooths returns. For a random walk, differences of monthly
   averages have ~2/3 of the true variance and lag-1 autocorrelation ~0.25 (Working, 1960),
   so sigma is biased LOW on this series.

2. Yahoo Finance ^SP500TR (S&P 500 total return index), month-end closes, 1988-2024.
   No averaging bias, but only ~37 years, so the drift is estimated very imprecisely.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .simulate import Scenario, cash_flows
from .strategies import Strategy

DATA = Path(__file__).resolve().parents[2] / "data"


# --------------------------------------------------------------------------- loaders
def load_shiller_total_return(path: Path | str = DATA / "sp500_shiller_monthly.csv") -> pd.Series:
    """Monthly gross total returns (index = month) from the Shiller data."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    df = df[df["Dividend"] > 0]  # drop price-only months after 2023-06
    gross = (df["SP500"] + df["Dividend"] / 12) / df["SP500"].shift(1)
    return gross.dropna().rename("gross_tr")


def load_yahoo_total_return(path: Path | str = DATA / "sp500tr_yahoo_monthly.csv") -> pd.Series:
    """Monthly gross total returns from ^SP500TR month-end closes (yfinance export format)."""
    df = pd.read_csv(path, skiprows=3, header=None, parse_dates=[0],
                     names=["Date", "Close", "High", "Low", "Open", "Volume"]).set_index("Date")
    return (df["Close"] / df["Close"].shift(1)).dropna().rename("gross_tr")


def to_price_index(gross: pd.Series, start: float = 1.0) -> pd.Series:
    return start * gross.cumprod()


# --------------------------------------------------------------------------- diagnostics
def gbm_diagnostics(gross: pd.Series | np.ndarray, periods_per_year: int = 12) -> dict[str, float]:
    """How well do monthly log returns fit the GBM assumption (i.i.d. normal)?

    Under GBM: skew = 0, excess kurtosis = 0, autocorrelation = 0, and a 3-sigma
    monthly drop happens with probability ~0.135%.
    """
    r = np.log(np.asarray(gross, dtype=float))
    n = r.size
    z = (r - r.mean()) / r.std(ddof=1)
    skew = float(np.mean(z**3))
    exkurt = float(np.mean(z**4) - 3)
    jb = n / 6 * (skew**2 + exkurt**2 / 4)          # Jarque-Bera, ~chi2(2) under normality
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    # annual (non-overlapping 12-month) log returns: does autocorrelation build up?
    k = n // periods_per_year
    ann = r[: k * periods_per_year].reshape(k, periods_per_year).sum(axis=1)
    var_ratio = float(ann.var(ddof=1) / (periods_per_year * r.var(ddof=1)))
    return {"n": n, "skew": skew, "excess_kurtosis": exkurt, "jarque_bera": float(jb),
            "lag1_autocorr": ac1, "ac1_se": 1 / np.sqrt(n), "variance_ratio_12m": var_ratio,
            "freq_below_-3sd": float(np.mean(z < -3)), "normal_below_-3sd": 0.00135,
            "worst_month": float(np.exp(r.min()) - 1)}


# --------------------------------------------------------------------------- backtest
def backtest(scn: Scenario, strategy: Strategy, gross_paths: np.ndarray) -> np.ndarray:
    """Terminal net worth when the portfolio earns *given* monthly gross returns.

    gross_paths: shape (n_windows, horizon). Uses exactly the same loan / contribution
    logic as simulate.run (cash_flows), so GBM and historical results are comparable.
    """
    debt, contrib, _ = cash_flows(scn, strategy)
    R = np.atleast_2d(gross_paths)
    if R.shape[1] != scn.horizon_months:
        raise ValueError("gross_paths must have horizon_months columns")
    port = np.full(R.shape[0], scn.initial_portfolio, dtype=float)
    for m in range(scn.horizon_months):
        port = (port + contrib[m]) * R[:, m]
    return port - debt[-1]


def rolling_windows(gross: pd.Series | np.ndarray, horizon: int, step: int = 1) -> np.ndarray:
    """Every historical window of `horizon` consecutive months (overlapping)."""
    g = np.asarray(gross, dtype=float)
    starts = range(0, g.size - horizon + 1, step)
    return np.stack([g[s:s + horizon] for s in starts])


def block_bootstrap(gross: pd.Series | np.ndarray, horizon: int, n_paths: int,
                    block: int = 12, seed=None) -> np.ndarray:
    """Moving-block bootstrap of historical monthly returns (keeps fat tails and
    short-range dependence that GBM ignores). Shape (n_paths, horizon)."""
    g = np.asarray(gross, dtype=float)
    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, g.size - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
    return g[idx]

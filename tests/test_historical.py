import numpy as np
import pytest

from gradsim import INVEST_FIRST, PAYOFF_FIRST, Loan, Scenario, run
from gradsim.historical import (backtest, block_bootstrap, gbm_diagnostics, load_shiller_total_return,
                                load_yahoo_total_return, rolling_windows)


def test_yahoo_matches_published_annual_total_returns():
    y = load_yahoo_total_return()
    for year, published in {2008: -0.3700, 2022: -0.1811, 2023: 0.2629, 2024: 0.2502}.items():
        assert y[str(year)].prod() - 1 == pytest.approx(published, abs=5e-4)


def test_shiller_series_is_clean():
    s = load_shiller_total_return()
    assert s.index.min().year == 1871 and s.index.max() <= np.datetime64("2023-06-30")
    assert np.all(s > 0.5) and np.all(s < 1.6) and not s.isna().any()


def test_backtest_matches_gbm_engine_with_constant_returns():
    """A flat historical return R per month must equal the GBM engine with sigma=0, mu=12 ln R."""
    R = 1.006
    scn = Scenario(Loan(30_000, 0.065), monthly_budget=800, horizon_months=120, mu=12 * np.log(R), sigma=0.0)
    for s in (INVEST_FIRST, PAYOFF_FIRST):
        hist = backtest(scn, s, np.full((1, 120), R))[0]
        gbm = run(scn, s, n_paths=1, seed=0).terminal_net_worth[0]
        assert hist == pytest.approx(gbm, rel=1e-10)


def test_rolling_windows_and_bootstrap_shapes():
    g = np.arange(1, 301) / 1000 + 1
    w = rolling_windows(g, 240)
    assert w.shape == (61, 240) and w[1, 0] == g[1]
    b = block_bootstrap(g, 240, 50, block=12, seed=0)
    assert b.shape == (50, 240) and np.isin(b, g).all()
    # within a block, consecutive months are consecutive in history
    assert np.allclose(np.diff(b[:, :12], axis=1), 0.001)


def test_diagnostics_on_normal_data():
    r = np.random.default_rng(0).normal(0.007, 0.045, 50_000)
    d = gbm_diagnostics(np.exp(r))
    assert abs(d["skew"]) < 0.05 and abs(d["excess_kurtosis"]) < 0.1 and abs(d["lag1_autocorr"]) < 0.02

"""Monte Carlo engine: combine the loan, the strategy and GBM market returns.

Monthly timeline for each month m = 1..horizon:
    1. Loan accrues interest; required payment = min(min_payment, balance).
    2. Discretionary cash = budget - required payment.
       extra prepayment = alpha * discretionary (capped at remaining balance);
       everything not used on the loan is invested.
    3. Portfolio (plus this month's contribution) grows by the GBM gross return.

The loan path is deterministic given alpha, so it is computed once; only the
portfolio is simulated across paths (fully vectorised over paths).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .debt import Loan
from .gbm import gross_returns
from .strategies import Strategy


@dataclass
class Scenario:
    loan: Loan
    monthly_budget: float          # $ per month available for loan + investing
    horizon_months: int = 240      # evaluation horizon (e.g. 20 years)
    mu: float = 0.08               # annual drift of the market
    sigma: float = 0.16            # annual volatility
    initial_portfolio: float = 0.0

    def __post_init__(self):
        if self.monthly_budget < self.loan.min_payment:
            raise ValueError("monthly_budget must cover the loan minimum payment")


@dataclass
class SimResult:
    strategy: Strategy
    terminal_net_worth: np.ndarray   # (n_paths,)
    portfolio_paths: np.ndarray      # (n_paths, horizon+1)
    debt_path: np.ndarray            # (horizon+1,)
    contributions: np.ndarray        # (horizon,) invested each month
    payoff_month: int | None         # month the loan hit zero (None if not within horizon)


def cash_flows(scn: Scenario, strategy: Strategy) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Deterministic debt balances and monthly investment contributions."""
    H, loan = scn.horizon_months, scn.loan
    i = loan.annual_rate / 12
    debt = np.empty(H + 1)
    contrib = np.empty(H)
    debt[0] = bal = loan.principal
    payoff = None
    for m in range(H):
        bal *= 1 + i
        required = min(loan.min_payment, bal)
        bal -= required
        discretionary = scn.monthly_budget - required
        extra = min(strategy.alpha * discretionary, bal)
        bal -= extra
        if bal < 1e-9:
            bal = 0.0
            if payoff is None:
                payoff = m + 1
        contrib[m] = discretionary - extra
        debt[m + 1] = bal
    return debt, contrib, payoff


def run(scn: Scenario, strategy: Strategy, n_paths: int = 10_000, seed=None,
        z: np.ndarray | None = None) -> SimResult:
    """Simulate one strategy. Pass the same `z` (or seed) across strategies for
    common random numbers, which makes strategy comparisons far less noisy."""
    debt, contrib, payoff = cash_flows(scn, strategy)
    H = scn.horizon_months
    R = gross_returns(scn.mu, scn.sigma, 1 / 12, n_paths, H, seed=seed, z=z)

    port = np.empty((R.shape[0], H + 1))
    port[:, 0] = scn.initial_portfolio
    for m in range(H):  # loop over time only; vectorised over paths
        port[:, m + 1] = (port[:, m] + contrib[m]) * R[:, m]

    return SimResult(strategy, port[:, -1] - debt[-1], port, debt, contrib, payoff)


def run_all(scn: Scenario, strategies, n_paths: int = 10_000, seed=0) -> dict[str, SimResult]:
    """Run several strategies on identical market paths (common random numbers)."""
    z = np.random.default_rng(seed).standard_normal((n_paths, scn.horizon_months))
    return {s.name: run(scn, s, z=z) for s in strategies}

"""Allocation strategies.

A strategy decides, each month, how the *discretionary* budget
(monthly_budget - loan minimum payment) is split between extra loan
prepayment and investing. The minimum payment is always made first.

alpha = fraction of the discretionary budget sent to extra prepayment.
    alpha = 0.0 -> "minimum payments, invest the rest"
    alpha = 1.0 -> "avalanche: kill the debt, then invest"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    name: str
    alpha: float  # in [0, 1]

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")


INVEST_FIRST = Strategy("Invest first (min payments)", 0.0)
BALANCED = Strategy("50/50 split", 0.5)
PAYOFF_FIRST = Strategy("Pay off first", 1.0)

DEFAULT_STRATEGIES = [INVEST_FIRST, BALANCED, PAYOFF_FIRST]

# TODO (extension): "match first" strategy — contribute enough to capture an
# employer 401(k) match, then prepay. Requires modelling the match in simulate.py.
# TODO (extension): rate-threshold rule — prepay only if loan APR > some hurdle.

"""Deterministic student-loan amortisation (monthly)."""
from __future__ import annotations

from dataclasses import dataclass


def minimum_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """Standard fixed amortising payment: P * i / (1 - (1+i)^-n)."""
    i = annual_rate / 12
    if i == 0:
        return principal / term_months
    return principal * i / (1 - (1 + i) ** -term_months)


@dataclass
class Loan:
    principal: float           # starting balance ($)
    annual_rate: float         # APR as a decimal, e.g. 0.065
    term_months: int = 120     # standard 10-year plan

    @property
    def min_payment(self) -> float:
        return minimum_payment(self.principal, self.annual_rate, self.term_months)


def step(balance: float, annual_rate: float, payment: float) -> tuple[float, float]:
    """Advance one month: accrue interest, then pay.

    Returns (new_balance, amount_actually_paid). Never overpays past zero.
    """
    accrued = balance * (1 + annual_rate / 12)
    paid = min(payment, accrued)
    return accrued - paid, paid


def amortisation_schedule(loan: Loan, extra: float = 0.0) -> list[float]:
    """Month-end balances paying min_payment + extra each month until the balance is 0."""
    bal, out = loan.principal, []
    for _ in range(loan.term_months * 2):  # safety cap
        if bal <= 1e-9:
            break
        bal, _ = step(bal, loan.annual_rate, loan.min_payment + extra)
        out.append(bal)
    return out

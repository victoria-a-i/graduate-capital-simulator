import numpy as np
import pytest

from gradsim import BALANCED, INVEST_FIRST, PAYOFF_FIRST, Loan, Scenario, run, run_all
from gradsim.debt import amortisation_schedule
from gradsim.metrics import crra_certainty_equivalent, cvar, prob_a_beats_b
from gradsim.simulate import cash_flows


def test_min_payment_retires_loan_on_schedule():
    loan = Loan(30_000, 0.065, 120)
    sched = amortisation_schedule(loan)
    assert len(sched) == 120
    assert sched[-1] == pytest.approx(0, abs=1e-6)


def test_zero_rate_loan():
    loan = Loan(12_000, 0.0, 120)
    assert loan.min_payment == pytest.approx(100)


def test_budget_must_cover_minimum():
    with pytest.raises(ValueError):
        Scenario(Loan(30_000, 0.065), monthly_budget=100)


def test_cash_is_conserved():
    scn = Scenario(Loan(30_000, 0.065), monthly_budget=800, horizon_months=120)
    for s in (INVEST_FIRST, BALANCED, PAYOFF_FIRST):
        debt, contrib, _ = cash_flows(scn, s)
        # total paid to loan + total invested == total budget
        i = scn.loan.annual_rate / 12
        paid = sum(debt[m] * (1 + i) - debt[m + 1] for m in range(120))
        assert paid + contrib.sum() == pytest.approx(800 * 120)


def test_payoff_first_retires_debt_sooner():
    scn = Scenario(Loan(30_000, 0.065), monthly_budget=800)
    _, _, p_fast = cash_flows(scn, PAYOFF_FIRST)
    _, _, p_slow = cash_flows(scn, INVEST_FIRST)
    assert p_fast < p_slow == 120


def test_zero_vol_equal_rates_are_equivalent():
    """Sanity check: if the market returns exactly the loan rate with no risk,
    the allocation should not matter (up to discretisation)."""
    r = 0.06
    mu = 12 * np.log(1 + r / 12)  # monthly compounding match
    scn = Scenario(Loan(30_000, r), monthly_budget=800, horizon_months=240, mu=mu, sigma=0.0)
    res = run_all(scn, [INVEST_FIRST, PAYOFF_FIRST], n_paths=4)
    a, b = (res[k].terminal_net_worth[0] for k in res)
    assert a == pytest.approx(b, rel=1e-6)


def test_common_random_numbers_and_metrics():
    scn = Scenario(Loan(30_000, 0.065), monthly_budget=800)
    res = run_all(scn, [INVEST_FIRST, PAYOFF_FIRST], n_paths=2000, seed=1)
    w_inv = res[INVEST_FIRST.name].terminal_net_worth
    w_pay = res[PAYOFF_FIRST.name].terminal_net_worth
    p = prob_a_beats_b(w_inv, w_pay)
    assert 0 <= p <= 1
    assert cvar(w_inv) <= np.median(w_inv)
    assert crra_certainty_equivalent(w_inv, 3) <= w_inv.mean()


def test_single_run_reproducible():
    scn = Scenario(Loan(30_000, 0.065), monthly_budget=800)
    a = run(scn, BALANCED, n_paths=100, seed=7).terminal_net_worth
    b = run(scn, BALANCED, n_paths=100, seed=7).terminal_net_worth
    assert np.array_equal(a, b)

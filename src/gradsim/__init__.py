"""gradsim: Monte Carlo study of student-debt payoff vs. early investing."""
from .debt import Loan, minimum_payment
from .simulate import Scenario, run, run_all
from .strategies import BALANCED, DEFAULT_STRATEGIES, INVEST_FIRST, PAYOFF_FIRST, Strategy

__all__ = ["Loan", "minimum_payment", "Scenario", "run", "run_all", "Strategy",
           "INVEST_FIRST", "BALANCED", "PAYOFF_FIRST", "DEFAULT_STRATEGIES"]

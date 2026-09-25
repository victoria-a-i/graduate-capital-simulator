"""Risk-aware comparison metrics.

Comparing only mean terminal wealth is biased toward investing: debt
prepayment earns a *guaranteed* return, the market a *risky* one. These
metrics put risk back in the comparison.
"""
from __future__ import annotations

import numpy as np


def summary(w: np.ndarray) -> dict[str, float]:
    p5, p50, p95 = np.percentile(w, [5, 50, 95])
    return {"mean": float(w.mean()), "std": float(w.std(ddof=1)),
            "p5": float(p5), "median": float(p50), "p95": float(p95)}


def prob_below(w: np.ndarray, threshold: float) -> float:
    """Shortfall probability P(W < threshold)."""
    return float(np.mean(w < threshold))


def prob_a_beats_b(w_a: np.ndarray, w_b: np.ndarray) -> float:
    """P(W_A > W_B) path-by-path. Requires both runs to share market paths."""
    return float(np.mean(w_a > w_b))


def cvar(w: np.ndarray, level: float = 0.05) -> float:
    """Expected shortfall: mean of the worst `level` fraction of outcomes."""
    k = max(1, int(np.ceil(level * w.size)))
    return float(np.sort(w)[:k].mean())


def crra_certainty_equivalent(w: np.ndarray, gamma: float, floor: float = 1.0) -> float:
    """Certainty-equivalent wealth under CRRA utility u(w) = w^(1-g)/(1-g).

    Wealth is floored at `floor` because CRRA is undefined for w <= 0.
    TODO: discuss in the report how this floor (and negative net worth) biases results.
    """
    w = np.maximum(w, floor)
    if np.isclose(gamma, 1.0):
        return float(np.exp(np.mean(np.log(w))))
    eu = np.mean(w ** (1 - gamma)) / (1 - gamma)
    return float(((1 - gamma) * eu) ** (1 / (1 - gamma)))


def mc_standard_error(w: np.ndarray) -> float:
    """Standard error of the Monte Carlo mean — report it alongside every mean."""
    return float(w.std(ddof=1) / np.sqrt(w.size))

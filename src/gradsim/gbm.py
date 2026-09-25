"""Geometric Brownian Motion, implemented from scratch with NumPy.

Model:  dS_t = mu * S_t dt + sigma * S_t dW_t

Exact solution (Ito's lemma on log S):
    S_{t+dt} = S_t * exp((mu - sigma^2 / 2) dt + sigma * sqrt(dt) * Z),   Z ~ N(0, 1)

Euler-Maruyama discretisation (approximate, kept for the verification study):
    S_{t+dt} = S_t * (1 + mu dt + sigma * sqrt(dt) * Z)

All functions are vectorised over paths: arrays have shape (n_paths, n_steps[+1]).
"""
from __future__ import annotations

import numpy as np


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def standard_normals(n_paths: int, n_steps: int, seed=None, antithetic: bool = False) -> np.ndarray:
    """Draw Z ~ N(0,1) of shape (n_paths, n_steps).

    antithetic=True returns [Z; -Z] (variance reduction); n_paths must then be even.
    """
    rng = _rng(seed)
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic sampling")
        z = rng.standard_normal((n_paths // 2, n_steps))
        return np.vstack([z, -z])
    return rng.standard_normal((n_paths, n_steps))


def gross_returns(mu: float, sigma: float, dt: float, n_paths: int, n_steps: int,
                  seed=None, z: np.ndarray | None = None, antithetic: bool = False) -> np.ndarray:
    """Per-step gross returns S_{t+dt}/S_t under exact GBM. Shape (n_paths, n_steps)."""
    if z is None:
        z = standard_normals(n_paths, n_steps, seed, antithetic)
    return np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)


def simulate_exact(s0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int,
                   seed=None, z: np.ndarray | None = None, antithetic: bool = False) -> np.ndarray:
    """Price paths under the exact GBM solution. Shape (n_paths, n_steps + 1), column 0 = s0."""
    dt = T / n_steps
    if z is None:
        z = standard_normals(n_paths, n_steps, seed, antithetic)
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return s0 * np.exp(log_path)


def simulate_euler(s0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int,
                   seed=None, z: np.ndarray | None = None) -> np.ndarray:
    """Price paths under Euler-Maruyama. Same shape/convention as simulate_exact.

    Pass the same `z` as simulate_exact to compare the schemes path-by-path (strong error).
    """
    dt = T / n_steps
    if z is None:
        z = standard_normals(n_paths, n_steps, seed)
    factors = 1.0 + mu * dt + sigma * np.sqrt(dt) * z
    return s0 * np.concatenate([np.ones((z.shape[0], 1)), np.cumprod(factors, axis=1)], axis=1)


def theoretical_moments(s0: float, mu: float, sigma: float, t: float) -> tuple[float, float]:
    """E[S_t] and Var[S_t] for GBM (used to verify the simulator)."""
    mean = s0 * np.exp(mu * t)
    var = s0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1.0)
    return mean, var

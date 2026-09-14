"""Vectorized, seed-controlled geometric Brownian motion path simulator.

Every path is drawn from a single ``rng.standard_normal`` call shaped
``(n_paths, n_steps)`` - there is no per-path Python loop, so cost scales
with numpy's vectorized RNG rather than the interpreter.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def simulate_gbm_paths(
    spot: float,
    rate: float,
    vol: float,
    days: int,
    n_paths: int,
    steps_per_day: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths under the risk-neutral (or given drift) rate.

    Uses the exact lognormal update

        S_{t+dt} = S_t * exp((rate - 0.5*vol**2)*dt + vol*sqrt(dt)*Z)

    rather than an Euler discretization, so there is no discretization bias
    to separate from Monte Carlo sampling error.

    Parameters
    ----------
    spot : initial price, must be > 0.
    rate : annualized drift (risk-free rate for risk-neutral pricing).
    vol : annualized volatility, must be > 0.
    days : horizon in trading days.
    n_paths : number of simulated paths.
    steps_per_day : sub-day time steps, 1 by default.
    seed : RNG seed. Same seed always reproduces identical paths.

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps + 1), column 0 is ``spot`` on
    every path. n_steps = days * steps_per_day.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if vol <= 0:
        raise ValueError(f"vol must be positive, got {vol}")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if steps_per_day <= 0:
        raise ValueError(f"steps_per_day must be positive, got {steps_per_day}")

    n_steps = days * steps_per_day
    dt = 1.0 / TRADING_DAYS_PER_YEAR / steps_per_day

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))

    drift = (rate - 0.5 * vol**2) * dt
    diffusion = vol * np.sqrt(dt) * z
    log_increments = drift + diffusion

    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(log_paths)
    return paths


def terminal_prices(paths: np.ndarray) -> np.ndarray:
    """The last column of a path matrix - the horizon-T price per path."""
    return paths[:, -1]

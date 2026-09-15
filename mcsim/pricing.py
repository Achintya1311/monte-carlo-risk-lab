"""Monte Carlo European option pricing on top of the Day 1 GBM simulator.

Discounts simulated terminal payoffs back to today under the same
risk-neutral drift used to generate the paths, so the estimator is
unbiased for the Black-Scholes price by construction - any gap left at
large N is sampling error, not model mismatch.
"""
from __future__ import annotations

import numpy as np

from mcsim.blackscholes import VALID_OPTION_TYPES
from mcsim.gbm import TRADING_DAYS_PER_YEAR, simulate_gbm_paths, terminal_prices


def mc_option_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    n_paths: int,
    option_type: str = "call",
    seed: int | None = None,
) -> dict:
    """Price a European option by Monte Carlo.

    Returns a dict with the discounted-payoff mean (``price``), the
    standard error of that mean (``std_error``, from the sample std of
    the discounted payoffs / sqrt(n_paths)), and ``n_paths``.
    """
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type!r}")

    paths = simulate_gbm_paths(spot=spot, rate=rate, vol=vol, days=days, n_paths=n_paths, seed=seed)
    terminals = terminal_prices(paths)

    if option_type == "call":
        payoffs = np.maximum(terminals - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminals, 0.0)

    T = days / TRADING_DAYS_PER_YEAR
    discounted = np.exp(-rate * T) * payoffs

    price = float(discounted.mean())
    # ddof=1: sample std, since n_paths is a finite draw, not the population.
    std_error = float(discounted.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan")

    return {"price": price, "std_error": std_error, "n_paths": n_paths}

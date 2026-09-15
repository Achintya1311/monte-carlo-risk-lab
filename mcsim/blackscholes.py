"""Black-Scholes closed-form European option pricing.

The correctness oracle for the Monte Carlo pricer: same risk-neutral GBM
assumptions (constant vol, constant rate, no dividends) as ``mcsim.gbm``, so
the two are directly comparable rather than pricing under different models.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from mcsim.gbm import TRADING_DAYS_PER_YEAR

VALID_OPTION_TYPES = ("call", "put")


def bs_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    option_type: str = "call",
) -> float:
    """European option price under Black-Scholes-Merton, no dividends.

    Parameters
    ----------
    spot : underlying price, must be > 0.
    strike : strike price, must be > 0.
    rate : annualized risk-free rate (continuously compounded drift).
    vol : annualized volatility, must be > 0.
    days : time to expiry in trading days, must be > 0.
    option_type : "call" or "put".
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if vol <= 0:
        raise ValueError(f"vol must be positive, got {vol}")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type!r}")

    T = days / TRADING_DAYS_PER_YEAR
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if option_type == "call":
        return float(spot * norm.cdf(d1) - strike * np.exp(-rate * T) * norm.cdf(d2))
    return float(strike * np.exp(-rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1))

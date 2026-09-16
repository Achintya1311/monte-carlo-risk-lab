"""Asian (average-price) option pricing - the first case in this repo with
no closed form to check against directly.

Arithmetic-average Asian options have no known closed form under GBM (a sum
of lognormals is not lognormal). The correctness check here is indirect but
still verifiable: the *geometric*-average Asian option under GBM does have a
closed form (the discrete-monitoring Kemna-Vorst formula below), so
``mc_asian_option_price`` is first validated on the geometric average against
that formula, and the arithmetic price is then checked against the geometric
one via Jensen's inequality - the arithmetic mean of positive numbers is
never less than their geometric mean, and a call payoff is a nondecreasing
function of the average, so the arithmetic call must price at or above the
geometric call (and symmetrically, at or below for puts).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from mcsim.blackscholes import VALID_OPTION_TYPES
from mcsim.gbm import TRADING_DAYS_PER_YEAR, simulate_gbm_paths

VALID_AVERAGE_TYPES = ("arithmetic", "geometric")


def _check_inputs(strike: float, option_type: str, average_type: str) -> None:
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type!r}")
    if average_type not in VALID_AVERAGE_TYPES:
        raise ValueError(f"average_type must be one of {VALID_AVERAGE_TYPES}, got {average_type!r}")


def mc_asian_option_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    n_paths: int,
    option_type: str = "call",
    average_type: str = "arithmetic",
    seed: int | None = None,
) -> dict:
    """Price a discretely-monitored average-price Asian option by Monte Carlo.

    Fixings are the simulator's daily closes (one per trading day, excluding
    the initial spot), so ``n_fixings == days`` - this needs the full path,
    unlike the Day 2/3 European pricers which only need the terminal price.

    Returns a dict with ``price``, ``std_error``, and ``n_paths``, matching
    the shape of ``mcsim.pricing.mc_option_price``.
    """
    _check_inputs(strike, option_type, average_type)

    paths = simulate_gbm_paths(spot=spot, rate=rate, vol=vol, days=days, n_paths=n_paths, seed=seed)
    fixings = paths[:, 1:]  # exclude the deterministic S0 column - only realized prices are fixings

    if average_type == "arithmetic":
        average = fixings.mean(axis=1)
    else:
        average = np.exp(np.log(fixings).mean(axis=1))

    if option_type == "call":
        payoffs = np.maximum(average - strike, 0.0)
    else:
        payoffs = np.maximum(strike - average, 0.0)

    T = days / TRADING_DAYS_PER_YEAR
    discounted = np.exp(-rate * T) * payoffs

    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan")
    return {"price": price, "std_error": std_error, "n_paths": n_paths}


def geometric_asian_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    n_fixings: int,
    option_type: str = "call",
) -> float:
    """Closed form for a discretely-monitored geometric-average Asian option.

    Kemna-Vorst (1990): the geometric average of n equally-spaced GBM fixings
    is itself lognormal, so the option prices with a Black-Scholes-shaped
    formula under an adjusted volatility and drift:

        vol_adj = vol * sqrt((n+1)(2n+1) / (6*n**2))
        rate_adj = 0.5*vol_adj**2 + (rate - 0.5*vol**2) * (n+1) / (2*n)

    which comes from ``Var[(1/n) sum_i W_{t_i}] = dt*(n+1)(2n+1)/(6n)`` for
    fixings at ``t_i = i*dt``. At n=1 this reduces exactly to Black-Scholes
    (vol_adj = vol, rate_adj = rate), which is a useful sanity check on the
    formula itself.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if vol <= 0:
        raise ValueError(f"vol must be positive, got {vol}")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if n_fixings <= 0:
        raise ValueError(f"n_fixings must be positive, got {n_fixings}")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type!r}")

    n = n_fixings
    T = days / TRADING_DAYS_PER_YEAR
    vol_adj = vol * np.sqrt((n + 1) * (2 * n + 1) / (6.0 * n**2))
    rate_adj = 0.5 * vol_adj**2 + (rate - 0.5 * vol**2) * (n + 1) / (2.0 * n)

    d1 = (np.log(spot / strike) + (rate_adj + 0.5 * vol_adj**2) * T) / (vol_adj * np.sqrt(T))
    d2 = d1 - vol_adj * np.sqrt(T)

    forward = spot * np.exp(rate_adj * T)
    if option_type == "call":
        undiscounted = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        undiscounted = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    return float(np.exp(-rate * T) * undiscounted)

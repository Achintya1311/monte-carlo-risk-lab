"""Barrier option pricing - the second path-dependent case with no closed
form exercised here (closed forms exist for continuously-monitored GBM
barriers, e.g. Reiner-Rubinstein, but a discretely-monitored barrier checked
against daily closes, as simulated here, does not match that formula and is
not implemented).

The correctness check is an exact identity rather than a tolerance band:
for the same barrier, ``knock-in + knock-out == vanilla`` on every single
path (a path either breaches the barrier or it does not - there is no third
case), so summing the two Monte Carlo prices computed from the *same* paths
must reproduce the vanilla Monte Carlo price to floating-point precision,
not just within sampling error. That vanilla MC price is then, separately,
checked against the Black-Scholes closed form within the usual std-error
tolerance - the same correctness gate Day 2 established.
"""
from __future__ import annotations

import numpy as np

from mcsim.blackscholes import VALID_OPTION_TYPES
from mcsim.gbm import TRADING_DAYS_PER_YEAR, simulate_gbm_paths

VALID_DIRECTIONS = ("up", "down")


def _check_inputs(spot: float, strike: float, barrier: float, option_type: str, direction: str) -> None:
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if barrier <= 0:
        raise ValueError(f"barrier must be positive, got {barrier}")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type!r}")
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {VALID_DIRECTIONS}, got {direction!r}")
    if direction == "up" and barrier <= spot:
        raise ValueError(f"an 'up' barrier must be above spot, got barrier={barrier} <= spot={spot}")
    if direction == "down" and barrier >= spot:
        raise ValueError(f"a 'down' barrier must be below spot, got barrier={barrier} >= spot={spot}")


def mc_barrier_pair_price(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    vol: float,
    days: int,
    n_paths: int,
    option_type: str = "call",
    direction: str = "up",
    seed: int | None = None,
) -> dict:
    """Price knock-in and knock-out together on one set of simulated paths.

    Monitoring is discrete: the barrier is checked at every simulated daily
    close (including the starting price, which never itself breaches since
    inputs are validated to have the barrier on the correct side of spot).

    Returns a dict with ``knock_in``, ``knock_out``, and ``vanilla_mc``
    sub-dicts (each ``{price, std_error, n_paths}``), ``breach_fraction``
    (share of paths that ever touch the barrier), and ``identity_gap``
    (``knock_in.price + knock_out.price - vanilla_mc.price``, which should
    be ~0 to floating-point precision by construction).
    """
    _check_inputs(spot, strike, barrier, option_type, direction)

    paths = simulate_gbm_paths(spot=spot, rate=rate, vol=vol, days=days, n_paths=n_paths, seed=seed)
    terminals = paths[:, -1]

    if direction == "up":
        breached = paths.max(axis=1) >= barrier
    else:
        breached = paths.min(axis=1) <= barrier

    if option_type == "call":
        vanilla_payoffs = np.maximum(terminals - strike, 0.0)
    else:
        vanilla_payoffs = np.maximum(strike - terminals, 0.0)

    T = days / TRADING_DAYS_PER_YEAR
    discount = np.exp(-rate * T)

    in_payoffs = discount * vanilla_payoffs * breached
    out_payoffs = discount * vanilla_payoffs * ~breached
    vanilla_discounted = discount * vanilla_payoffs

    def _summary(discounted: np.ndarray) -> dict:
        price = float(discounted.mean())
        std_error = float(discounted.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan")
        return {"price": price, "std_error": std_error, "n_paths": n_paths}

    knock_in = _summary(in_payoffs)
    knock_out = _summary(out_payoffs)
    vanilla_mc = _summary(vanilla_discounted)

    return {
        "knock_in": knock_in,
        "knock_out": knock_out,
        "vanilla_mc": vanilla_mc,
        "breach_fraction": float(breached.mean()),
        "identity_gap": knock_in["price"] + knock_out["price"] - vanilla_mc["price"],
    }

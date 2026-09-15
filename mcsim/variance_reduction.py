"""Variance reduction for European option MC pricing: antithetic and control variates.

Both are measured against the plain estimator (``mcsim.pricing.mc_option_price``)
at the *same* path budget ``n_paths``, so the payoff is a standard-error
reduction factor at fixed simulation cost, not just a smaller number that
could come from quietly using more paths.

Both techniques here only need the terminal price, not the full path, so
they draw it directly from the exact one-step lognormal update rather than
going through ``mcsim.gbm.simulate_gbm_paths``. That is mathematically the
same distribution as summing ``days`` daily increments (a sum of independent
Gaussian increments is itself Gaussian with the summed variance), so it does
not change what is being priced - it only means these functions consume the
RNG differently from ``mc_option_price``, and the two are not driven by
common random numbers.
"""
from __future__ import annotations

import numpy as np

from mcsim.blackscholes import VALID_OPTION_TYPES
from mcsim.gbm import TRADING_DAYS_PER_YEAR


def _terminal_prices(spot: float, rate: float, vol: float, days: int, z: np.ndarray) -> np.ndarray:
    T = days / TRADING_DAYS_PER_YEAR
    return spot * np.exp((rate - 0.5 * vol**2) * T + vol * np.sqrt(T) * z)


def _payoffs(terminals: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(terminals - strike, 0.0)
    return np.maximum(strike - terminals, 0.0)


def _check_inputs(strike: float, option_type: str) -> None:
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type!r}")


def mc_option_price_antithetic(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    n_paths: int,
    option_type: str = "call",
    seed: int | None = None,
) -> dict:
    """Antithetic variates: pair each draw Z with -Z.

    A vanilla call/put payoff is a monotone function of the terminal price,
    so Z and -Z produce negatively correlated payoffs; averaging each pair
    before computing the sample variance is what buys the reduction - it is
    not simply averaging more draws.
    """
    _check_inputs(strike, option_type)
    n_pairs = n_paths // 2
    if n_pairs < 1:
        raise ValueError(f"n_paths must be >= 2 for antithetic pairing, got {n_paths}")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_pairs)

    T = days / TRADING_DAYS_PER_YEAR
    discount = np.exp(-rate * T)

    payoff_plus = discount * _payoffs(_terminal_prices(spot, rate, vol, days, z), strike, option_type)
    payoff_minus = discount * _payoffs(_terminal_prices(spot, rate, vol, days, -z), strike, option_type)
    pair_avg = 0.5 * (payoff_plus + payoff_minus)

    price = float(pair_avg.mean())
    std_error = float(pair_avg.std(ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else float("nan")
    return {"price": price, "std_error": std_error, "n_paths": n_pairs * 2}


def mc_option_price_control_variate(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    n_paths: int,
    option_type: str = "call",
    seed: int | None = None,
) -> dict:
    """Control variate: the discounted terminal price itself.

    Under the risk-neutral drift used everywhere in this repo,
    ``exp(-rate*T) * S_T`` is a martingale with known mean exactly ``spot`` -
    no simulation needed for that mean, so its own sampling error is removed
    outright rather than merely estimated down. The optimal coefficient
    ``c* = Cov(X, Y) / Var(X)`` is estimated from the same sample used to
    price the option, at no extra simulation cost.
    """
    _check_inputs(strike, option_type)
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2 to estimate the control coefficient, got {n_paths}")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    terminals = _terminal_prices(spot, rate, vol, days, z)

    T = days / TRADING_DAYS_PER_YEAR
    discount = np.exp(-rate * T)
    payoffs = discount * _payoffs(terminals, strike, option_type)
    control = discount * terminals
    control_mean = spot  # E[exp(-rate*T) * S_T] under this drift, exactly.

    cov = np.cov(control, payoffs, ddof=1)
    c_star = float(cov[0, 1] / cov[0, 0]) if cov[0, 0] > 0 else 0.0

    adjusted = payoffs - c_star * (control - control_mean)

    price = float(adjusted.mean())
    std_error = float(adjusted.std(ddof=1) / np.sqrt(n_paths))
    return {"price": price, "std_error": std_error, "n_paths": n_paths, "c_star": c_star}


def variance_reduction_table(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    n_paths: int,
    option_type: str = "call",
    seed: int | None = None,
) -> list[dict]:
    """Plain vs antithetic vs control-variate MC at the same path budget.

    Returns one row per method with ``price``, ``std_error``, ``n_paths``,
    and ``reduction_factor`` (plain std error / this method's std error -
    greater than 1 means the technique bought a tighter estimate without
    spending more paths).
    """
    from mcsim.pricing import mc_option_price

    plain = mc_option_price(
        spot=spot, strike=strike, rate=rate, vol=vol, days=days,
        n_paths=n_paths, option_type=option_type, seed=seed,
    )
    antithetic = mc_option_price_antithetic(
        spot=spot, strike=strike, rate=rate, vol=vol, days=days,
        n_paths=n_paths, option_type=option_type, seed=seed,
    )
    control = mc_option_price_control_variate(
        spot=spot, strike=strike, rate=rate, vol=vol, days=days,
        n_paths=n_paths, option_type=option_type, seed=seed,
    )

    rows = [
        {"method": "plain", **plain},
        {"method": "antithetic", **antithetic},
        {"method": "control_variate", **control},
    ]
    plain_se = plain["std_error"]
    for row in rows:
        row["reduction_factor"] = plain_se / row["std_error"] if row["std_error"] > 0 else float("nan")
    return rows

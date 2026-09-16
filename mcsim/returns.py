"""Historical daily log returns loaded from a committed OHLCV fixture.

The risk-measurement side of this repo (Day 5+) needs a real-world return
sample, not a risk-neutral pricing assumption - Days 1-4 draw GBM paths
under a risk-neutral drift ``rate`` because they price hedgeable payoffs
under Q; VaR/CVaR estimate the physical (P-measure) chance of a loss, so
the drift used here is the sample mean of the historical series itself.
"""
from __future__ import annotations

import csv

import numpy as np

from mcsim.gbm import TRADING_DAYS_PER_YEAR


def load_closes(csv_path: str) -> np.ndarray:
    """Read a ``date,close,...`` fixture and return the close column.

    Rows are assumed to already be in ascending date order, which is how
    every fixture in this pipeline is written.
    """
    closes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            closes.append(float(row["close"]))

    if len(closes) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 rows of close prices, cannot compute returns")

    return np.asarray(closes, dtype=np.float64)


def load_daily_log_returns(csv_path: str) -> np.ndarray:
    """Read a ``date,close,...`` fixture and return daily log returns."""
    return np.diff(np.log(load_closes(csv_path)))


def annualize(mu_daily: float, sigma_daily: float) -> tuple[float, float]:
    """Scale a daily log-return mean/std to annualized figures (iid assumption)."""
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    return mu_annual, sigma_annual

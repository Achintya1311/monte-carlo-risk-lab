"""Commodity term structure: contango and backwardation, handled explicitly.

Days 1-5 only ever price or risk a single series (an equity spot/terminal
price). A commodity futures curve is a second axis the rest of this repo
never has to look at: at any date there is a near-month contract and
several further-out contracts, and whether the far contract is pricier
(contango, cost-of-carry dominates) or cheaper (backwardation, a scarcity
convenience yield dominates) changes the economics of holding the position
- a continuously-rolled long that ignores curve shape quietly pays away
the contango roll cost or quietly earns the backwardation roll yield,
either way without saying so. This module makes that shape a first-class,
measured quantity rather than assuming a single cost-of-carry rate the way
Days 1-4's risk-neutral GBM drift does for equities.

The correctness check is an exact identity, not a tolerance band, in the
same spirit as ``mcsim.barrier``'s knock-in + knock-out = vanilla: the
curve's log-carry from contract 1 to contract 4 must equal the sum of the
three one-month segment log-carries (1->2, 2->3, 3->4), because both sides
are the same telescoping sum of log-prices - any nonzero gap is a data or
parsing bug, not sampling noise.

VaR/CVaR by curve regime reuses ``mcsim.risk.historical_var_cvar``'s exact
method (empirical quantile of overlapping horizon-day return windows) - the
only change is conditioning each window on the regime observed on the day
the window starts, instead of pooling every day into one unconditional
number the way Day 5 does for the equity fixture.
"""
from __future__ import annotations

import csv

import numpy as np

MIN_REGIME_WINDOWS = 30


def load_curve(csv_path: str) -> dict:
    """Read a ``date,c1,c2,c3,c4`` futures-curve fixture."""
    dates, c1, c2, c3, c4 = [], [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row["date"])
            c1.append(float(row["c1"]))
            c2.append(float(row["c2"]))
            c3.append(float(row["c3"]))
            c4.append(float(row["c4"]))

    if len(dates) == 0:
        raise ValueError(f"{csv_path} has no rows")

    return {
        "dates": dates,
        "c1": np.asarray(c1, dtype=np.float64),
        "c2": np.asarray(c2, dtype=np.float64),
        "c3": np.asarray(c3, dtype=np.float64),
        "c4": np.asarray(c4, dtype=np.float64),
    }


def classify_regime(c_near: np.ndarray, c_far: np.ndarray) -> np.ndarray:
    """'contango' if the far contract is pricier, 'backwardation' if cheaper, else 'flat'."""
    return np.where(c_far > c_near, "contango", np.where(c_far < c_near, "backwardation", "flat"))


def annualized_basis(c_near: np.ndarray, c_far: np.ndarray, months_apart: float) -> np.ndarray:
    """Annualized cost-of-carry the curve implies between two contract months.

    ``ln(far/near) * 12/months_apart``: positive means the curve slopes up
    (contango), negative means it slopes down (backwardation). This is the
    same log-return-annualization idiom ``mcsim.returns.annualize`` uses
    for daily equity returns, applied across contract months instead of
    across trading days.
    """
    if months_apart <= 0:
        raise ValueError(f"months_apart must be positive, got {months_apart}")
    return np.log(c_far / c_near) * (12.0 / months_apart)


def curve_identity_gap(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, c4: np.ndarray) -> float:
    """Max absolute gap between the summed segment log-carries and the direct c1->c4 log-carry.

    ``log(c4/c1) == log(c2/c1) + log(c3/c2) + log(c4/c3)`` is a telescoping
    log-arithmetic identity, true for any positive prices - it should hold
    to floating-point precision, not just approximately.
    """
    segment_sum = np.log(c2 / c1) + np.log(c3 / c2) + np.log(c4 / c3)
    direct = np.log(c4 / c1)
    return float(np.max(np.abs(segment_sum - direct)))


def regime_summary(regime: np.ndarray) -> dict:
    """Counts, shares, and the current label of a per-day regime array."""
    total = len(regime)
    counts = {label: int((regime == label).sum()) for label in ("contango", "backwardation", "flat")}
    return {
        "n_days": total,
        "counts": counts,
        "pct": {k: v / total for k, v in counts.items()},
        "current": str(regime[-1]),
    }


def regime_conditioned_var_cvar(
    daily_log_returns: np.ndarray, window_regime: np.ndarray, confidence: float, horizon_days: int
) -> dict:
    """Historical VaR/CVaR (mcsim.risk's overlapping-window method) split by curve regime.

    ``window_regime[i]`` is the regime observed on the day the horizon
    window starting at day ``i`` begins - risk is conditioned on the curve
    shape *today*, not on the shape realized somewhere inside the window.
    A regime with fewer than ``MIN_REGIME_WINDOWS`` windows is reported
    (so the thin sample is visible) but its VaR/CVaR is withheld rather
    than printed as a number nobody should trust.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")

    cumulative = np.convolve(daily_log_returns, np.ones(horizon_days), mode="valid")
    n_windows = len(cumulative)
    if len(window_regime) < n_windows:
        raise ValueError(
            f"window_regime has {len(window_regime)} entries, need at least {n_windows} "
            f"to label every {horizon_days}-day window"
        )
    starts = window_regime[:n_windows]

    def _var_cvar(sample: np.ndarray) -> tuple[float, float]:
        var = float(np.quantile(sample, 1.0 - confidence))
        tail = sample[sample <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        return var, cvar

    out = {}
    for label in ("contango", "backwardation", "flat"):
        mask = starts == label
        n = int(mask.sum())
        if n < MIN_REGIME_WINDOWS:
            out[label] = {"n_windows": n, "var": None, "cvar": None}
            continue
        var, cvar = _var_cvar(cumulative[mask])
        out[label] = {"n_windows": n, "var": var, "cvar": cvar}

    var, cvar = _var_cvar(cumulative)
    out["unconditional"] = {"n_windows": n_windows, "var": var, "cvar": cvar}
    return out

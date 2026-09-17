"""Pipeline audit tests.

The point of this audit is to catch look-ahead bias and fixture
misalignment mechanically. That is only worth anything if the checkers
actually fire when fed something broken, so the negative-control tests
below feed them a deliberately non-causal function and a deliberately
misaligned fixture pair alongside the real ones.
"""
import csv

import numpy as np

from mcsim.audit import alignment_check, audit_curve, causality_violations, run_audit
from mcsim.commodity import load_curve

CURVE = "fixtures/commodity/WTI_futures_curve.csv"
PRICES = "fixtures/commodity/WTI_C1.csv"


# ---------------------------------------------------------------------------
# causality
# ---------------------------------------------------------------------------


def test_real_curve_functions_are_causal():
    assert audit_curve(CURVE) == []


def test_centered_smoothing_is_flagged():
    """A centered window looks ahead by definition - the audit must catch it."""
    dates = [str(i) for i in range(40)]
    c1 = np.linspace(90, 110, 40)
    c4 = np.linspace(95, 105, 40)

    def leaky(near, far):
        ratio = far / near
        return np.convolve(ratio, np.ones(5) / 5.0, mode="same")

    violations = causality_violations("leaky_centered", leaky, (c1, c4), dates, min_index=3)
    assert violations, "a centered convolution uses future bars and must be flagged"


def test_negative_shift_is_flagged():
    """Shifting a series backwards pulls tomorrow's value into today's row."""
    dates = [str(i) for i in range(40)]
    c1 = np.linspace(90, 110, 40)
    c4 = np.linspace(95, 105, 40)

    def leaky(near, far):
        return np.roll(far, -1)

    violations = causality_violations("leaky_shift", leaky, (c1, c4), dates)
    assert violations, "shift(-1) is future data by construction and must be flagged"


# ---------------------------------------------------------------------------
# fixture alignment
# ---------------------------------------------------------------------------


def test_committed_fixtures_are_aligned():
    result = alignment_check(CURVE, PRICES)
    assert result["aligned"] is True
    assert result["mismatches"] == []
    assert result["n_curve_days"] == result["n_price_days"]


def test_reordered_dates_are_flagged(tmp_path):
    # A prices file covering the same calendar as the curve fixture but
    # rotated by one row - the exact silent failure mode mcsim.curve's
    # positional window_regime alignment has no defense against, since
    # load_daily_log_returns discards dates entirely.
    curve_dates = load_curve(CURVE)["dates"]
    rotated = curve_dates[1:] + curve_dates[:1]
    rotated_path = tmp_path / "rotated_prices.csv"
    with open(rotated_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close"])
        for i, d in enumerate(rotated):
            writer.writerow([d, 100.0 + i])

    result = alignment_check(CURVE, str(rotated_path))
    assert result["aligned"] is False
    assert result["mismatches"]


def test_length_mismatch_is_flagged(tmp_path):
    curve_dates = load_curve(CURVE)["dates"]
    short_path = tmp_path / "short_prices.csv"
    with open(short_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close"])
        for i, d in enumerate(curve_dates[:-5]):
            writer.writerow([d, 100.0 + i])

    result = alignment_check(CURVE, str(short_path))
    assert result["aligned"] is False


# ---------------------------------------------------------------------------
# run_audit
# ---------------------------------------------------------------------------


def test_run_audit_clean_on_committed_fixtures():
    report = run_audit(CURVE, PRICES)
    assert report["clean"] is True
    assert report["causality_findings"] == []
    assert report["alignment"]["aligned"] is True

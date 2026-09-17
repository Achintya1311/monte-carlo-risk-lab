import numpy as np
import pytest

from mcsim.commodity import (
    MIN_REGIME_WINDOWS,
    annualized_basis,
    classify_regime,
    curve_identity_gap,
    load_curve,
    regime_conditioned_var_cvar,
    regime_summary,
)
from mcsim.risk import historical_var_cvar

FIXTURE_CURVE = "fixtures/commodity/WTI_futures_curve.csv"
FIXTURE_PRICES = "fixtures/commodity/WTI_C1.csv"


# ---------------------------------------------------------------------------
# load_curve
# ---------------------------------------------------------------------------


def test_load_curve_reads_committed_fixture():
    curve = load_curve(FIXTURE_CURVE)
    n = curve["c1"].shape[0]
    assert n > 400
    for key in ("c2", "c3", "c4"):
        assert curve[key].shape[0] == n
    assert len(curve["dates"]) == n
    assert (curve["c1"] > 0).all()


def test_load_curve_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("date,c1,c2,c3,c4\n")
    with pytest.raises(ValueError, match="no rows"):
        load_curve(str(empty))


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------


def test_classify_regime_labels_contango_backwardation_flat():
    near = np.array([100.0, 100.0, 100.0])
    far = np.array([105.0, 95.0, 100.0])
    regime = classify_regime(near, far)
    assert list(regime) == ["contango", "backwardation", "flat"]


def test_classify_regime_on_fixture_matches_price_comparison():
    curve = load_curve(FIXTURE_CURVE)
    regime = classify_regime(curve["c1"], curve["c4"])
    assert set(regime) <= {"contango", "backwardation", "flat"}
    assert ((regime == "contango") == (curve["c4"] > curve["c1"])).all()
    assert ((regime == "backwardation") == (curve["c4"] < curve["c1"])).all()
    # The fixture genuinely contains both regimes - not cherry-picked to one.
    assert (regime == "contango").sum() > 0
    assert (regime == "backwardation").sum() > 0


# ---------------------------------------------------------------------------
# annualized_basis
# ---------------------------------------------------------------------------


def test_annualized_basis_known_value():
    near = np.array([100.0])
    far = near * np.exp(0.03)  # 3% log-carry over the stated window
    basis = annualized_basis(near, far, months_apart=3.0)
    assert basis[0] == pytest.approx(0.03 * 4, abs=1e-10)


def test_annualized_basis_sign_matches_regime():
    near = np.array([100.0, 100.0])
    far = np.array([110.0, 90.0])
    basis = annualized_basis(near, far, months_apart=3.0)
    assert basis[0] > 0  # contango
    assert basis[1] < 0  # backwardation


def test_annualized_basis_rejects_nonpositive_months():
    with pytest.raises(ValueError, match="months_apart"):
        annualized_basis(np.array([100.0]), np.array([101.0]), months_apart=0.0)


# ---------------------------------------------------------------------------
# curve_identity_gap - exact telescoping identity, not a tolerance band
# ---------------------------------------------------------------------------


def test_curve_identity_gap_is_zero_by_construction():
    rng = np.random.default_rng(3)
    c1 = rng.uniform(50, 150, size=200)
    c2 = c1 * rng.uniform(0.9, 1.1, size=200)
    c3 = c2 * rng.uniform(0.9, 1.1, size=200)
    c4 = c3 * rng.uniform(0.9, 1.1, size=200)
    gap = curve_identity_gap(c1, c2, c3, c4)
    assert gap < 1e-9


def test_curve_identity_gap_on_fixture_is_floating_point_zero():
    curve = load_curve(FIXTURE_CURVE)
    gap = curve_identity_gap(curve["c1"], curve["c2"], curve["c3"], curve["c4"])
    assert gap < 1e-9


# ---------------------------------------------------------------------------
# regime_summary
# ---------------------------------------------------------------------------


def test_regime_summary_counts_and_current():
    regime = np.array(["contango", "contango", "backwardation", "flat"])
    summary = regime_summary(regime)
    assert summary["n_days"] == 4
    assert summary["counts"] == {"contango": 2, "backwardation": 1, "flat": 1}
    assert summary["pct"]["contango"] == pytest.approx(0.5)
    assert summary["current"] == "flat"


# ---------------------------------------------------------------------------
# regime_conditioned_var_cvar
# ---------------------------------------------------------------------------


def test_regime_conditioned_var_cvar_matches_historical_when_single_regime():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0003, 0.015, size=200)
    window_regime = np.full(200, "contango")

    conditioned = regime_conditioned_var_cvar(returns, window_regime, confidence=0.95, horizon_days=10)
    baseline = historical_var_cvar(returns, confidence=0.95, horizon_days=10)

    assert conditioned["contango"]["var"] == pytest.approx(baseline["var"])
    assert conditioned["contango"]["cvar"] == pytest.approx(baseline["cvar"])
    assert conditioned["unconditional"]["var"] == pytest.approx(baseline["var"])
    assert conditioned["backwardation"]["var"] is None
    assert conditioned["backwardation"]["n_windows"] == 0


def test_regime_conditioned_var_cvar_withholds_thin_samples():
    rng = np.random.default_rng(5)
    n = 200
    returns = rng.normal(0.0, 0.02, size=n)
    labels = np.array(["backwardation"] * (n - 5) + ["contango"] * 5)

    result = regime_conditioned_var_cvar(returns, labels, confidence=0.95, horizon_days=10)
    assert result["contango"]["n_windows"] < MIN_REGIME_WINDOWS
    assert result["contango"]["var"] is None
    assert result["backwardation"]["var"] is not None


def test_regime_conditioned_var_cvar_cvar_at_or_below_var_when_reported():
    curve = load_curve(FIXTURE_CURVE)
    regime = classify_regime(curve["c1"], curve["c4"])
    log_returns = np.diff(np.log(curve["c1"]))
    result = regime_conditioned_var_cvar(log_returns, regime[:-1], confidence=0.95, horizon_days=20)
    for label, row in result.items():
        if row["var"] is not None:
            assert row["cvar"] <= row["var"], label


def test_regime_conditioned_var_cvar_rejects_short_regime_array():
    returns = np.zeros(50)
    with pytest.raises(ValueError, match="window_regime"):
        regime_conditioned_var_cvar(returns, np.full(10, "contango"), confidence=0.95, horizon_days=20)


def test_regime_conditioned_var_cvar_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        regime_conditioned_var_cvar(np.zeros(50), np.full(50, "contango"), confidence=1.5, horizon_days=10)

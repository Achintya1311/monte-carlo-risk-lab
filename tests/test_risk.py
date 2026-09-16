import numpy as np
import pytest

from mcsim.risk import (
    drawdown_distribution_stats,
    historical_var_cvar,
    kupiec_pof_test,
    max_drawdowns,
    mc_var_cvar_from_paths,
    parametric_var_cvar,
    simulate_risk_paths,
)

# ---------------------------------------------------------------------------
# historical_var_cvar
# ---------------------------------------------------------------------------


def test_historical_var_cvar_cvar_at_or_below_var():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0003, 0.015, size=500)
    result = historical_var_cvar(returns, confidence=0.95, horizon_days=10)
    assert result["cvar"] <= result["var"]
    assert result["n_windows"] == 500 - 10 + 1


def test_historical_var_cvar_higher_confidence_is_more_extreme():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.02, size=500)
    var_95 = historical_var_cvar(returns, confidence=0.95, horizon_days=5)["var"]
    var_99 = historical_var_cvar(returns, confidence=0.99, horizon_days=5)["var"]
    assert var_99 <= var_95


def test_historical_var_cvar_rejects_horizon_longer_than_sample():
    with pytest.raises(ValueError, match="need at least"):
        historical_var_cvar(np.zeros(5), confidence=0.95, horizon_days=10)


def test_historical_var_cvar_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        historical_var_cvar(np.zeros(20), confidence=1.5, horizon_days=5)


# ---------------------------------------------------------------------------
# parametric_var_cvar - checked against known standard-normal tail constants
# ---------------------------------------------------------------------------


def test_parametric_var_cvar_standard_normal_95():
    result = parametric_var_cvar(mu_daily=0.0, sigma_daily=1.0, confidence=0.95, horizon_days=1)
    assert result["var"] == pytest.approx(-1.644854, abs=1e-5)
    assert result["cvar"] == pytest.approx(-2.062713, abs=1e-4)


def test_parametric_var_cvar_standard_normal_99():
    result = parametric_var_cvar(mu_daily=0.0, sigma_daily=1.0, confidence=0.99, horizon_days=1)
    assert result["var"] == pytest.approx(-2.326348, abs=1e-5)
    assert result["cvar"] == pytest.approx(-2.665214, abs=1e-4)


def test_parametric_var_cvar_horizon_scaling():
    one_day = parametric_var_cvar(mu_daily=0.001, sigma_daily=0.02, confidence=0.95, horizon_days=1)
    ten_day = parametric_var_cvar(mu_daily=0.001, sigma_daily=0.02, confidence=0.95, horizon_days=10)
    assert ten_day["mu_h"] == pytest.approx(one_day["mu_h"] * 10)
    assert ten_day["sigma_h"] == pytest.approx(one_day["sigma_h"] * np.sqrt(10))


def test_parametric_var_cvar_cvar_at_or_below_var():
    result = parametric_var_cvar(mu_daily=0.0005, sigma_daily=0.018, confidence=0.99, horizon_days=20)
    assert result["cvar"] <= result["var"]


def test_parametric_var_cvar_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma_daily"):
        parametric_var_cvar(mu_daily=0.0, sigma_daily=0.0, confidence=0.95, horizon_days=1)


# ---------------------------------------------------------------------------
# Monte Carlo VaR/CVaR
# ---------------------------------------------------------------------------


def test_mc_var_cvar_from_paths_matches_manual_quantile():
    spot = 100.0
    terminals = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    paths = np.column_stack([np.full(10, spot), terminals])
    result = mc_var_cvar_from_paths(paths, spot=spot, confidence=0.9)

    returns = np.log(terminals / spot)
    expected_var = np.quantile(returns, 0.1)
    expected_cvar = returns[returns <= expected_var].mean()
    assert result["var"] == pytest.approx(expected_var)
    assert result["cvar"] == pytest.approx(expected_cvar)
    assert result["cvar"] <= result["var"]


def test_mc_var_cvar_converges_toward_parametric_at_matching_moments():
    mu_daily, sigma_daily = 0.0, 0.02
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)
    paths = simulate_risk_paths(spot=100.0, mu_annual=mu_annual, sigma_annual=sigma_annual, horizon_days=1, n_paths=500_000, seed=7)
    mc = mc_var_cvar_from_paths(paths, spot=100.0, confidence=0.95)
    param = parametric_var_cvar(mu_daily, sigma_daily, confidence=0.95, horizon_days=1)
    assert mc["var"] == pytest.approx(param["var"], abs=0.001)
    assert mc["cvar"] == pytest.approx(param["cvar"], abs=0.001)


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------


def test_max_drawdowns_zero_on_monotonic_increase():
    paths = np.array([[100.0, 105.0, 110.0, 120.0]])
    assert max_drawdowns(paths)[0] == pytest.approx(0.0)


def test_max_drawdowns_known_v_shape():
    paths = np.array([[100.0, 80.0, 90.0]])
    assert max_drawdowns(paths)[0] == pytest.approx(-0.2)


def test_max_drawdowns_vectorized_over_multiple_paths():
    paths = np.array([[100.0, 105.0, 110.0, 120.0], [100.0, 80.0, 90.0, 85.0]])
    dd = max_drawdowns(paths)
    assert dd[0] == pytest.approx(0.0)
    assert dd[1] == pytest.approx(-0.2)


def test_drawdown_distribution_stats_worst_is_minimum():
    drawdowns = np.array([-0.1, -0.2, -0.3, -0.05, -0.15])
    stats = drawdown_distribution_stats(drawdowns)
    assert stats["n_paths"] == 5
    assert stats["worst"] == pytest.approx(-0.3)
    assert stats["mean"] == pytest.approx(drawdowns.mean())
    assert stats["percentiles"][50] == pytest.approx(np.median(drawdowns))


# ---------------------------------------------------------------------------
# Kupiec proportion-of-failures backtest
# ---------------------------------------------------------------------------


def test_kupiec_exact_expected_rate_gives_zero_lr():
    n = 1000
    n_exceptions = 50  # exactly p=0.05 of n at confidence=0.95
    returns = np.concatenate([np.full(n_exceptions, -10.0), np.full(n - n_exceptions, 10.0)])
    result = kupiec_pof_test(returns, var_threshold=-1.0, confidence=0.95)
    assert result["n_exceptions"] == n_exceptions
    assert result["lr_stat"] == pytest.approx(0.0, abs=1e-8)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)
    assert result["reject_at_5pct"] is False


def test_kupiec_too_many_exceptions_is_rejected():
    n = 500
    returns = np.concatenate([np.full(100, -10.0), np.full(n - 100, 10.0)])  # 20% breach vs 5% expected
    result = kupiec_pof_test(returns, var_threshold=-1.0, confidence=0.95)
    assert result["reject_at_5pct"] is True
    assert result["lr_stat"] > 0


def test_kupiec_zero_exceptions_does_not_crash():
    returns = np.full(200, 10.0)
    result = kupiec_pof_test(returns, var_threshold=-1.0, confidence=0.95)
    assert result["n_exceptions"] == 0
    assert np.isfinite(result["lr_stat"])
    assert 0.0 <= result["p_value"] <= 1.0


def test_kupiec_all_exceptions_does_not_crash():
    returns = np.full(50, -10.0)
    result = kupiec_pof_test(returns, var_threshold=-1.0, confidence=0.95)
    assert result["n_exceptions"] == 50
    assert np.isfinite(result["lr_stat"])
    assert result["reject_at_5pct"] is True


def test_kupiec_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        kupiec_pof_test(np.zeros(10), var_threshold=-1.0, confidence=0.0)

import numpy as np
import pytest

from mcsim.gbm import TRADING_DAYS_PER_YEAR, simulate_gbm_paths, terminal_prices


def test_shape_is_paths_by_steps_plus_one():
    paths = simulate_gbm_paths(spot=100, rate=0.05, vol=0.2, days=10, n_paths=50, seed=1)
    assert paths.shape == (50, 11)


def test_steps_per_day_multiplies_columns():
    paths = simulate_gbm_paths(spot=100, rate=0.05, vol=0.2, days=10, n_paths=5, steps_per_day=4, seed=1)
    assert paths.shape == (5, 41)


def test_first_column_is_spot_on_every_path():
    paths = simulate_gbm_paths(spot=123.45, rate=0.05, vol=0.2, days=5, n_paths=20, seed=1)
    assert np.all(paths[:, 0] == 123.45)


def test_paths_stay_strictly_positive():
    paths = simulate_gbm_paths(spot=100, rate=0.05, vol=0.6, days=252, n_paths=1000, seed=7)
    assert np.all(paths > 0)


def test_same_seed_reproduces_identical_paths():
    a = simulate_gbm_paths(spot=100, rate=0.05, vol=0.2, days=30, n_paths=100, seed=42)
    b = simulate_gbm_paths(spot=100, rate=0.05, vol=0.2, days=30, n_paths=100, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge():
    a = simulate_gbm_paths(spot=100, rate=0.05, vol=0.2, days=30, n_paths=100, seed=1)
    b = simulate_gbm_paths(spot=100, rate=0.05, vol=0.2, days=30, n_paths=100, seed=2)
    assert not np.array_equal(a, b)


def test_zero_volatility_paths_are_deterministic_growth():
    # vol=0 collapses the diffusion term; every path should follow exp(rate*t)
    # exactly. vol must stay > 0 in simulate_gbm_paths (division-by-noise-less
    # cases aren't the point), so approximate with a tiny vol and a tight bound.
    paths = simulate_gbm_paths(spot=100, rate=0.05, vol=1e-8, days=252, n_paths=3, seed=1)
    expected = 100 * np.exp(0.05 * 1.0)
    np.testing.assert_allclose(paths[:, -1], expected, rtol=1e-5)


def test_terminal_prices_is_last_column():
    paths = simulate_gbm_paths(spot=100, rate=0.05, vol=0.2, days=10, n_paths=5, seed=1)
    np.testing.assert_array_equal(terminal_prices(paths), paths[:, -1])


def test_large_sample_mean_converges_to_lognormal_theory():
    # E[S_T] = S0 * exp(rate * T) under the risk-neutral / given drift.
    spot, rate, vol, days = 100.0, 0.07, 0.25, 30
    paths = simulate_gbm_paths(spot=spot, rate=rate, vol=vol, days=days, n_paths=500_000, seed=123)
    terminals = terminal_prices(paths)
    T = days / TRADING_DAYS_PER_YEAR
    theoretical_mean = spot * np.exp(rate * T)
    sample_mean = terminals.mean()
    # Monte Carlo std error at N=500k, vol=0.25, T=30/252 is small; 1% is generous.
    assert abs(sample_mean - theoretical_mean) / theoretical_mean < 0.01


def test_large_sample_log_return_std_matches_theory():
    spot, rate, vol, days = 100.0, 0.07, 0.25, 30
    paths = simulate_gbm_paths(spot=spot, rate=rate, vol=vol, days=days, n_paths=500_000, seed=123)
    terminals = terminal_prices(paths)
    T = days / TRADING_DAYS_PER_YEAR
    theoretical_log_std = vol * np.sqrt(T)
    sample_log_std = np.log(terminals / spot).std(ddof=1)
    assert abs(sample_log_std - theoretical_log_std) / theoretical_log_std < 0.02


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spot=0, rate=0.05, vol=0.2, days=10, n_paths=10),
        dict(spot=-5, rate=0.05, vol=0.2, days=10, n_paths=10),
        dict(spot=100, rate=0.05, vol=0, days=10, n_paths=10),
        dict(spot=100, rate=0.05, vol=0.2, days=0, n_paths=10),
        dict(spot=100, rate=0.05, vol=0.2, days=10, n_paths=0),
        dict(spot=100, rate=0.05, vol=0.2, days=10, n_paths=10, steps_per_day=0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_gbm_paths(**kwargs, seed=1)

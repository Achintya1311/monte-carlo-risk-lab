import pytest

from mcsim.blackscholes import bs_price
from mcsim.pricing import mc_option_price
from mcsim.variance_reduction import (
    mc_option_price_antithetic,
    mc_option_price_control_variate,
    variance_reduction_table,
)

SPOT, STRIKE, RATE, VOL, DAYS = 100.0, 105.0, 0.07, 0.25, 30


def test_antithetic_converges_to_black_scholes_at_large_n():
    theoretical = bs_price(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, option_type="call")
    mc = mc_option_price_antithetic(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=500_000, seed=42
    )
    assert abs(mc["price"] - theoretical) < 4 * mc["std_error"]


def test_antithetic_converges_for_puts_too():
    theoretical = bs_price(spot=SPOT, strike=95.0, rate=0.05, vol=0.3, days=60, option_type="put")
    mc = mc_option_price_antithetic(
        spot=SPOT, strike=95.0, rate=0.05, vol=0.3, days=60, n_paths=500_000, option_type="put", seed=7
    )
    assert abs(mc["price"] - theoretical) < 4 * mc["std_error"]


def test_antithetic_reduces_std_error_vs_plain_at_same_path_budget():
    kwargs = dict(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, option_type="call", seed=42)
    plain = mc_option_price(n_paths=100_000, **kwargs)
    antithetic = mc_option_price_antithetic(n_paths=100_000, **kwargs)
    assert antithetic["n_paths"] == plain["n_paths"]
    assert antithetic["std_error"] < plain["std_error"]


def test_antithetic_odd_path_count_uses_floor_pairs():
    mc = mc_option_price_antithetic(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=1_001, seed=1
    )
    assert mc["n_paths"] == 1_000


def test_antithetic_too_few_paths_raises():
    with pytest.raises(ValueError):
        mc_option_price_antithetic(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=1, seed=1)


def test_antithetic_reproducible_with_same_seed():
    kwargs = dict(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=1_000, seed=99)
    a = mc_option_price_antithetic(**kwargs)
    b = mc_option_price_antithetic(**kwargs)
    assert a == b


def test_antithetic_invalid_strike_raises():
    with pytest.raises(ValueError):
        mc_option_price_antithetic(spot=SPOT, strike=0, rate=RATE, vol=VOL, days=DAYS, n_paths=100, seed=1)


def test_antithetic_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_option_price_antithetic(
            spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=100, option_type="bad", seed=1
        )


def test_control_variate_converges_to_black_scholes_at_large_n():
    theoretical = bs_price(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, option_type="call")
    mc = mc_option_price_control_variate(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=500_000, seed=42
    )
    assert abs(mc["price"] - theoretical) < 4 * mc["std_error"]


def test_control_variate_converges_for_puts_too():
    theoretical = bs_price(spot=SPOT, strike=95.0, rate=0.05, vol=0.3, days=60, option_type="put")
    mc = mc_option_price_control_variate(
        spot=SPOT, strike=95.0, rate=0.05, vol=0.3, days=60, n_paths=500_000, option_type="put", seed=7
    )
    assert abs(mc["price"] - theoretical) < 4 * mc["std_error"]


def test_control_variate_reduces_std_error_vs_plain_at_same_path_budget():
    kwargs = dict(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, option_type="call", seed=42)
    plain = mc_option_price(n_paths=100_000, **kwargs)
    control = mc_option_price_control_variate(n_paths=100_000, **kwargs)
    assert control["n_paths"] == plain["n_paths"]
    assert control["std_error"] < plain["std_error"]


def test_control_variate_coefficient_is_positive_for_itm_call():
    # A call payoff is positively related to the terminal price, so the
    # optimal control coefficient should come out positive.
    mc = mc_option_price_control_variate(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=50_000, seed=42
    )
    assert mc["c_star"] > 0


def test_control_variate_too_few_paths_raises():
    with pytest.raises(ValueError):
        mc_option_price_control_variate(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=1, seed=1)


def test_control_variate_reproducible_with_same_seed():
    kwargs = dict(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=1_000, seed=99)
    a = mc_option_price_control_variate(**kwargs)
    b = mc_option_price_control_variate(**kwargs)
    assert a == b


def test_control_variate_invalid_strike_raises():
    with pytest.raises(ValueError):
        mc_option_price_control_variate(spot=SPOT, strike=0, rate=RATE, vol=VOL, days=DAYS, n_paths=100, seed=1)


def test_control_variate_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_option_price_control_variate(
            spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=100, option_type="bad", seed=1
        )


def test_variance_reduction_table_has_one_row_per_method():
    rows = variance_reduction_table(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=50_000, seed=42
    )
    methods = [r["method"] for r in rows]
    assert methods == ["plain", "antithetic", "control_variate"]


def test_variance_reduction_table_reduction_factor_above_one_for_variance_reduction_methods():
    rows = variance_reduction_table(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=50_000, seed=42
    )
    by_method = {r["method"]: r for r in rows}
    assert by_method["plain"]["reduction_factor"] == pytest.approx(1.0)
    assert by_method["antithetic"]["reduction_factor"] > 1.0
    assert by_method["control_variate"]["reduction_factor"] > 1.0

import numpy as np
import pytest

from mcsim.blackscholes import bs_price
from mcsim.pricing import mc_option_price


def test_mc_converges_to_black_scholes_at_large_n():
    spot, strike, rate, vol, days = 100.0, 105.0, 0.07, 0.25, 30
    theoretical = bs_price(spot=spot, strike=strike, rate=rate, vol=vol, days=days, option_type="call")
    mc = mc_option_price(spot=spot, strike=strike, rate=rate, vol=vol, days=days, n_paths=500_000, seed=42)
    # within 4 standard errors of the BS price - the same tolerance mcsim.price uses.
    assert abs(mc["price"] - theoretical) < 4 * mc["std_error"]


def test_mc_converges_for_puts_too():
    spot, strike, rate, vol, days = 100.0, 95.0, 0.05, 0.3, 60
    theoretical = bs_price(spot=spot, strike=strike, rate=rate, vol=vol, days=days, option_type="put")
    mc = mc_option_price(
        spot=spot, strike=strike, rate=rate, vol=vol, days=days, n_paths=500_000, option_type="put", seed=7,
    )
    assert abs(mc["price"] - theoretical) < 4 * mc["std_error"]


def test_std_error_shrinks_roughly_as_inverse_sqrt_n():
    kwargs = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.2, days=30, seed=1)
    small = mc_option_price(n_paths=1_000, **kwargs)
    large = mc_option_price(n_paths=100_000, **kwargs)
    ratio = small["std_error"] / large["std_error"]
    # 100x more paths -> std error should shrink by ~sqrt(100)=10x.
    assert ratio == pytest.approx(10.0, rel=0.3)


def test_deep_otm_call_has_near_zero_price():
    # Strike far above any plausible terminal price at this vol/horizon:
    # almost every path pays off zero.
    mc = mc_option_price(spot=100, strike=500, rate=0.05, vol=0.2, days=10, n_paths=50_000, seed=1)
    assert mc["price"] < 0.01


def test_call_payoffs_are_never_negative():
    mc = mc_option_price(spot=100, strike=100, rate=0.05, vol=0.4, days=30, n_paths=10_000, seed=3)
    assert mc["price"] >= 0


def test_same_seed_reproducible_price():
    kwargs = dict(spot=100, strike=100, rate=0.05, vol=0.2, days=30, n_paths=1_000, seed=99)
    a = mc_option_price(**kwargs)
    b = mc_option_price(**kwargs)
    assert a["price"] == b["price"]
    assert a["std_error"] == b["std_error"]


def test_invalid_strike_raises():
    with pytest.raises(ValueError):
        mc_option_price(spot=100, strike=0, rate=0.05, vol=0.2, days=10, n_paths=100, seed=1)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_option_price(spot=100, strike=100, rate=0.05, vol=0.2, days=10, n_paths=100, option_type="bad", seed=1)

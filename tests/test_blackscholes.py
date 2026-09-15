import numpy as np
import pytest

from mcsim.blackscholes import bs_price


def test_known_textbook_value():
    # Hull, "Options, Futures, and Other Derivatives": S=42, K=40, r=0.10,
    # vol=0.20, T=0.5y (call price ~ 4.76). Convert T=0.5y to trading days
    # at TRADING_DAYS_PER_YEAR=252 so it matches this module's day-count.
    price = bs_price(spot=42, strike=40, rate=0.10, vol=0.20, days=126, option_type="call")
    assert price == pytest.approx(4.76, abs=0.02)


def test_call_price_increases_with_spot():
    low = bs_price(spot=90, strike=100, rate=0.05, vol=0.2, days=30, option_type="call")
    high = bs_price(spot=110, strike=100, rate=0.05, vol=0.2, days=30, option_type="call")
    assert high > low


def test_put_price_decreases_with_spot():
    low = bs_price(spot=90, strike=100, rate=0.05, vol=0.2, days=30, option_type="put")
    high = bs_price(spot=110, strike=100, rate=0.05, vol=0.2, days=30, option_type="put")
    assert high < low


def test_put_call_parity():
    # C - P = S - K*exp(-rT), independent of vol - a model-free identity
    # any correct BS implementation must satisfy exactly (to float precision).
    spot, strike, rate, days = 100.0, 95.0, 0.06, 60
    call = bs_price(spot=spot, strike=strike, rate=rate, vol=0.3, days=days, option_type="call")
    put = bs_price(spot=spot, strike=strike, rate=rate, vol=0.3, days=days, option_type="put")
    T = days / 252
    assert (call - put) == pytest.approx(spot - strike * np.exp(-rate * T), abs=1e-9)


def test_price_is_never_negative():
    for option_type in ("call", "put"):
        price = bs_price(spot=50, strike=150, rate=0.05, vol=0.15, days=5, option_type=option_type)
        assert price >= 0


def test_deep_itm_call_approaches_intrinsic_value():
    # Deep ITM, short-dated, low vol: time value is small, price ~ S - K*exp(-rT).
    spot, strike, rate, days = 200.0, 50.0, 0.05, 1
    price = bs_price(spot=spot, strike=strike, rate=rate, vol=0.1, days=days, option_type="call")
    intrinsic = spot - strike * np.exp(-rate * days / 252)
    assert price == pytest.approx(intrinsic, abs=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spot=0, strike=100, rate=0.05, vol=0.2, days=10),
        dict(spot=-5, strike=100, rate=0.05, vol=0.2, days=10),
        dict(spot=100, strike=0, rate=0.05, vol=0.2, days=10),
        dict(spot=100, strike=-1, rate=0.05, vol=0.2, days=10),
        dict(spot=100, strike=100, rate=0.05, vol=0, days=10),
        dict(spot=100, strike=100, rate=0.05, vol=0.2, days=0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        bs_price(**kwargs)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(spot=100, strike=100, rate=0.05, vol=0.2, days=10, option_type="straddle")

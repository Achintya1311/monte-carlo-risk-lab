import pytest

from mcsim.asian import geometric_asian_price, mc_asian_option_price
from mcsim.blackscholes import bs_price

SPOT, STRIKE, RATE, VOL, DAYS = 100.0, 100.0, 0.07, 0.25, 30


def test_geometric_mc_converges_to_closed_form():
    closed_form = geometric_asian_price(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_fixings=DAYS)
    mc = mc_asian_option_price(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS,
        n_paths=500_000, average_type="geometric", seed=42,
    )
    assert abs(mc["price"] - closed_form) < 4 * mc["std_error"]


def test_geometric_mc_converges_for_puts_too():
    closed_form = geometric_asian_price(
        spot=SPOT, strike=105.0, rate=0.05, vol=0.3, days=60, n_fixings=60, option_type="put"
    )
    mc = mc_asian_option_price(
        spot=SPOT, strike=105.0, rate=0.05, vol=0.3, days=60,
        n_paths=500_000, option_type="put", average_type="geometric", seed=7,
    )
    assert abs(mc["price"] - closed_form) < 4 * mc["std_error"]


def test_geometric_closed_form_reduces_to_black_scholes_at_one_fixing():
    # With a single fixing, the "average" is just the terminal price - the
    # Kemna-Vorst formula's adjusted vol/rate should collapse to plain BS.
    bs = bs_price(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, option_type="call")
    geo = geometric_asian_price(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_fixings=1)
    assert geo == pytest.approx(bs, rel=1e-9)


def test_arithmetic_call_prices_at_or_above_geometric_call():
    # Jensen's inequality: arithmetic mean >= geometric mean, and a call
    # payoff is nondecreasing in the average, so this must hold up to
    # sampling noise even though there's no closed form for arithmetic.
    arithmetic = mc_asian_option_price(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS,
        n_paths=500_000, average_type="arithmetic", seed=42,
    )
    geometric = mc_asian_option_price(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS,
        n_paths=500_000, average_type="geometric", seed=42,
    )
    combined_se = (arithmetic["std_error"] ** 2 + geometric["std_error"] ** 2) ** 0.5
    assert arithmetic["price"] >= geometric["price"] - 4 * combined_se


def test_arithmetic_put_prices_at_or_below_geometric_put():
    arithmetic = mc_asian_option_price(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS,
        n_paths=500_000, option_type="put", average_type="arithmetic", seed=42,
    )
    geometric = mc_asian_option_price(
        spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS,
        n_paths=500_000, option_type="put", average_type="geometric", seed=42,
    )
    combined_se = (arithmetic["std_error"] ** 2 + geometric["std_error"] ** 2) ** 0.5
    assert arithmetic["price"] <= geometric["price"] + 4 * combined_se


def test_reproducible_with_same_seed():
    kwargs = dict(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_paths=1_000, seed=99)
    a = mc_asian_option_price(**kwargs)
    b = mc_asian_option_price(**kwargs)
    assert a == b


def test_invalid_strike_raises():
    with pytest.raises(ValueError):
        mc_asian_option_price(spot=SPOT, strike=0, rate=RATE, vol=VOL, days=DAYS, n_paths=100, seed=1)


def test_invalid_average_type_raises():
    with pytest.raises(ValueError):
        mc_asian_option_price(
            spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS,
            n_paths=100, average_type="harmonic", seed=1,
        )


def test_geometric_closed_form_invalid_n_fixings_raises():
    with pytest.raises(ValueError):
        geometric_asian_price(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, n_fixings=0)

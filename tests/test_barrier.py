import pytest

from mcsim.barrier import mc_barrier_pair_price
from mcsim.blackscholes import bs_price

SPOT, STRIKE, RATE, VOL, DAYS = 100.0, 100.0, 0.07, 0.25, 30


def test_knock_in_plus_knock_out_equals_vanilla_mc_exactly():
    # Exact identity, not a tolerance: every path either breaches or doesn't.
    result = mc_barrier_pair_price(
        spot=SPOT, strike=STRIKE, barrier=120.0, rate=RATE, vol=VOL, days=DAYS,
        n_paths=200_000, direction="up", seed=42,
    )
    assert abs(result["identity_gap"]) < 1e-9


def test_vanilla_mc_converges_to_black_scholes():
    theoretical = bs_price(spot=SPOT, strike=STRIKE, rate=RATE, vol=VOL, days=DAYS, option_type="call")
    result = mc_barrier_pair_price(
        spot=SPOT, strike=STRIKE, barrier=120.0, rate=RATE, vol=VOL, days=DAYS,
        n_paths=500_000, direction="up", seed=42,
    )
    vanilla = result["vanilla_mc"]
    assert abs(vanilla["price"] - theoretical) < 4 * vanilla["std_error"]


def test_knock_out_cheaper_than_vanilla_for_up_barrier():
    # An up-and-out call can only pay less than the vanilla call, never more.
    result = mc_barrier_pair_price(
        spot=SPOT, strike=STRIKE, barrier=110.0, rate=RATE, vol=VOL, days=DAYS,
        n_paths=200_000, direction="up", seed=42,
    )
    assert result["knock_out"]["price"] <= result["vanilla_mc"]["price"]
    assert result["knock_in"]["price"] <= result["vanilla_mc"]["price"]


def test_very_high_barrier_is_never_breached_so_knock_out_equals_vanilla():
    result = mc_barrier_pair_price(
        spot=SPOT, strike=STRIKE, barrier=1_000_000.0, rate=RATE, vol=VOL, days=DAYS,
        n_paths=50_000, direction="up", seed=1,
    )
    assert result["breach_fraction"] == 0.0
    assert result["knock_out"]["price"] == pytest.approx(result["vanilla_mc"]["price"])
    assert result["knock_in"]["price"] == 0.0


def test_down_direction_with_low_barrier():
    result = mc_barrier_pair_price(
        spot=SPOT, strike=STRIKE, barrier=80.0, rate=RATE, vol=VOL, days=DAYS,
        n_paths=200_000, direction="down", seed=42,
    )
    assert abs(result["identity_gap"]) < 1e-9
    assert 0.0 <= result["breach_fraction"] <= 1.0


def test_up_barrier_below_spot_raises():
    with pytest.raises(ValueError):
        mc_barrier_pair_price(
            spot=SPOT, strike=STRIKE, barrier=90.0, rate=RATE, vol=VOL, days=DAYS,
            n_paths=1_000, direction="up", seed=1,
        )


def test_down_barrier_above_spot_raises():
    with pytest.raises(ValueError):
        mc_barrier_pair_price(
            spot=SPOT, strike=STRIKE, barrier=110.0, rate=RATE, vol=VOL, days=DAYS,
            n_paths=1_000, direction="down", seed=1,
        )


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        mc_barrier_pair_price(
            spot=SPOT, strike=STRIKE, barrier=120.0, rate=RATE, vol=VOL, days=DAYS,
            n_paths=1_000, direction="sideways", seed=1,
        )


def test_reproducible_with_same_seed():
    kwargs = dict(spot=SPOT, strike=STRIKE, barrier=120.0, rate=RATE, vol=VOL, days=DAYS, n_paths=1_000, seed=99)
    a = mc_barrier_pair_price(**kwargs)
    b = mc_barrier_pair_price(**kwargs)
    assert a == b

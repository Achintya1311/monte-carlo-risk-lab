"""P151: Holm step-down adjustment for the in-sample Kupiec family."""

from pathlib import Path

import pytest

from mcsim.returns import load_daily_log_returns
from mcsim.risk import holm_adjust, kupiec_pof_test, parametric_var_cvar

ROOT = Path(__file__).resolve().parents[1]


def test_holm_textbook_example():
    # Holm (1979) step-down: sort, multiply by (m - rank), enforce monotonicity, cap at 1.
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_input_order_and_caps_at_one():
    assert holm_adjust([0.9, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []


def test_holm_rejects_out_of_range():
    with pytest.raises(ValueError):
        holm_adjust([0.5, 1.2])


def _kupiec_p(fixture, confidence):
    r = load_daily_log_returns(ROOT / "fixtures" / "ohlcv" / fixture)
    var_1day = parametric_var_cvar(float(r.mean()), float(r.std(ddof=1)), confidence, 1)["var"]
    return kupiec_pof_test(r, var_1day, confidence)["p_value"]


def test_four_test_family_matches_the_readme():
    names = [("RELIANCE_NS.csv", 0.95), ("RELIANCE_NS.csv", 0.99), ("GULFOILLUB_NS.csv", 0.95), ("GULFOILLUB_NS.csv", 0.99)]
    adjusted = holm_adjust([_kupiec_p(f, c) for f, c in names])
    rel95, rel99, g95, g99 = adjusted
    assert g95 < 0.05  # the only rejection that survives
    assert g99 > 0.05 and rel99 > 0.05 and rel95 > 0.05
    readme = (ROOT / "README.md").read_text()
    for value in (g95, g99, rel99):
        assert f"{value:.3f}" in readme

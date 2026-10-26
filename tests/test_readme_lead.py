"""R22: the README's lead Finding names its stock and says the Kupiec test is in-sample.

The numbers are recomputed here the way ``mcsim.portfolio`` computes them
(1-day parametric VaR from the fitted sample, tested on the same days).
"""

from pathlib import Path

from mcsim.returns import load_daily_log_returns
from mcsim.risk import kupiec_pof_test, parametric_var_cvar

ROOT = Path(__file__).resolve().parents[1]


def _kupiec(fixture: str, confidence: float) -> dict:
    r = load_daily_log_returns(ROOT / fixture)
    var_1day = parametric_var_cvar(float(r.mean()), float(r.std(ddof=1)), confidence, 1)["var"]
    return kupiec_pof_test(r, var_1day, confidence)


def _lead() -> str:
    text = (ROOT / "README.md").read_text()
    return next(line for line in text.splitlines() if line.startswith("**Finding"))


def test_lead_names_reliance_and_says_in_sample():
    lead = _lead()
    assert "RELIANCE" in lead
    assert "in-sample" in lead
    k = _kupiec("fixtures/ohlcv/RELIANCE_NS.csv", 0.99)
    assert f"{k['n_exceptions']}/{k['n_obs']} exceptions vs. {k['expected_exceptions']:.1f} expected" in lead
    assert f"p={k['p_value']:.3f}" in lead


def test_lead_gives_gulfoillub_in_sample_both_levels():
    lead = _lead()
    for c in (0.95, 0.99):
        k = _kupiec("fixtures/ohlcv/GULFOILLUB_NS.csv", c)
        assert k["reject_at_5pct"]
        assert f"{k['n_exceptions']} vs {k['expected_exceptions']:.1f}" in lead

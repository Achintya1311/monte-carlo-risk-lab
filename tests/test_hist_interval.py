"""P152: a moving-block bootstrap interval for historical VaR/CVaR."""
import numpy as np
import pytest
from scipy.stats import norm

from mcsim.risk import historical_var_cvar, historical_var_cvar_interval


def test_interval_brackets_the_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.015, 500)
    a = historical_var_cvar_interval(r, 0.95, 20, n_boot=300, seed=7)
    b = historical_var_cvar_interval(r, 0.95, 20, n_boot=300, seed=7)
    assert a == b
    point = historical_var_cvar(r, 0.95, 20)
    assert a["var_low"] <= point["var"] <= a["var_high"]
    assert a["cvar_low"] <= point["cvar"] <= a["cvar_high"]
    assert a["cvar_low"] <= a["var_high"]
    assert a["block"] == 20 and a["level"] == 0.90 and a["n_boot"] == 300


def test_interval_covers_the_true_var_on_iid_normal_returns():
    # 20-day sum of iid N(0, s) is N(0, s*sqrt(20)): the true 95% VaR is known.
    s, h, c = 0.015, 20, 0.95
    true_var = s * np.sqrt(h) * norm.ppf(1 - c)
    hits = 0
    reps = 60
    for k in range(reps):
        r = np.random.default_rng(100 + k).normal(0.0, s, 500)
        iv = historical_var_cvar_interval(r, c, h, n_boot=200, seed=k)
        hits += iv["var_low"] <= true_var <= iv["var_high"]
    # nominal 90%; a block bootstrap of a tail quantile from ~25 independent
    # windows can under-cover somewhat, so accept 75-100%.
    assert hits / reps >= 0.75


def test_interval_rejects_bad_inputs():
    r = np.zeros(100)
    with pytest.raises(ValueError):
        historical_var_cvar_interval(r, 0.95, 20, level=1.0)
    with pytest.raises(ValueError):
        historical_var_cvar_interval(r, 0.95, 20, n_boot=0)
    with pytest.raises(ValueError):
        historical_var_cvar_interval(r[:10], 0.95, 20)


@pytest.mark.parametrize("fixture,expected", [
    ("fixtures/ohlcv/RELIANCE_NS.csv", ["[-11.97%, -7.10%]", "[-14.03%, -8.65%]", "[-15.84%, -9.54%]", "[-17.39%, -10.39%]"]),
    ("fixtures/ohlcv/GULFOILLUB_NS.csv", ["[-14.77%, -10.70%]", "[-21.51%, -14.06%]", "[-27.36%, -15.91%]", "[-34.29%, -18.32%]"]),
])
def test_readme_intervals_match_the_command(fixture, expected):
    from pathlib import Path
    from mcsim.returns import load_daily_log_returns

    root = Path(__file__).resolve().parents[1]
    r = load_daily_log_returns(root / fixture)
    got = []
    for c in (0.95, 0.99):
        iv = historical_var_cvar_interval(r, c, 20, seed=42)
        got += [f"[{iv['var_low']:.2%}, {iv['var_high']:.2%}]", f"[{iv['cvar_low']:.2%}, {iv['cvar_high']:.2%}]"]
    assert got == expected
    readme = (root / "README.md").read_text()
    for text in expected:
        assert text in readme

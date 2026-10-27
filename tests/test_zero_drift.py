"""P125 (CRITIQUES.md): show VaR and the simulated drawdown at zero drift
beside the fitted-drift numbers, since the fitted mean is mostly noise."""

import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import norm

from mcsim.portfolio import build_parser, run

GULFOILLUB_FIXTURE = "fixtures/ohlcv/GULFOILLUB_NS.csv"


def _run(fixture, *extra):
    args = build_parser().parse_args(
        ["--fixture", fixture, "--horizon-days", "20", "--confidence", "0.95", "0.99", "--paths", "50000", "--seed", "42", *extra]
    )
    return run(args)


def _synthetic_fixture(tmp_path, mu_daily):
    # a strong known drift: log returns alternate +-1% around mu_daily
    n = 500
    r = mu_daily + 0.01 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    path = tmp_path / "SYN.csv"
    lines = ["date,close,high,low,open,volume"]
    for i, c in enumerate(closes):
        lines.append(f"2020-01-{(i % 28) + 1:02d},{c},{c},{c},{c},1")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_zero_drift_var_is_sigma_times_z_only(tmp_path):
    result = _run(str(_synthetic_fixture(tmp_path, 0.002)))
    sigma = result["sigma_daily"]
    for row in result["zero_drift"]["by_confidence"]:
        z = norm.ppf(1 - row["confidence"])
        assert row["parametric"]["var"] == pytest.approx(sigma * np.sqrt(20) * z)


def test_positive_drift_makes_the_fitted_var_less_severe(tmp_path):
    result = _run(str(_synthetic_fixture(tmp_path, 0.002)))
    fitted = {r["confidence"]: r["parametric"]["var"] for r in result["by_confidence"]}
    zero = {r["confidence"]: r["parametric"]["var"] for r in result["zero_drift"]["by_confidence"]}
    for c in (0.95, 0.99):
        assert fitted[c] - zero[c] == pytest.approx(0.002 * 20, rel=1e-9)
    assert result["zero_drift"]["drawdown_mean"] < result["drawdown_stats"]["mean"]


def test_gulfoillub_zero_drift_numbers():
    result = _run(GULFOILLUB_FIXTURE)
    zero = {r["confidence"]: r["parametric"]["var"] for r in result["zero_drift"]["by_confidence"]}
    assert zero[0.95] == pytest.approx(-0.1565, abs=5e-5)
    assert zero[0.99] == pytest.approx(-0.2214, abs=5e-5)


def test_cli_prints_the_zero_drift_block():
    out = subprocess.run(
        [sys.executable, "-m", "mcsim.portfolio", "--fixture", GULFOILLUB_FIXTURE, "--paths", "50000"],
        capture_output=True, text=True,
    ).stdout
    block = out.split("Zero-drift check", 1)[1]
    assert "-0.1565" in block
    assert "drawdown mean" in block

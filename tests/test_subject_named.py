"""S31, P169, P170 (hub CRITIQUES.md): every public page carries rule 1, and the
chart and the out-of-sample result name the stock they describe."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from mcsim.risk import plot_drawdown_distribution

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = (
    "Educational research on historical data. Not investment advice. "
    "Not a SEBI-registered research analyst."
)


def test_every_docs_page_carries_the_disclaimer():
    pages = sorted((ROOT / "docs").glob("*.md"))
    assert pages
    for page in pages:
        assert DISCLAIMER in page.read_text(), page.name


def test_drawdown_chart_title_names_stock_horizon_and_paths(tmp_path):
    drawdowns = -np.abs(np.random.default_rng(0).normal(0.05, 0.02, 1000))
    title = plot_drawdown_distribution(
        drawdowns, str(tmp_path / "dd.png"), ticker="RELIANCE", horizon_days=20
    )
    assert (tmp_path / "dd.png").exists()
    assert "RELIANCE" in title
    assert "20-day" in title
    assert "1,000 paths" in title


def test_portfolio_cli_titles_the_chart_from_the_fixture(tmp_path):
    out = tmp_path / "dd.png"
    res = subprocess.run(
        [sys.executable, "-m", "mcsim.portfolio", "--fixture", "fixtures/ohlcv/GULFOILLUB_NS.csv",
         "--paths", "2000", "--drawdown-plot", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert res.returncode in (0, 1), res.stderr
    assert "GULFOILLUB, 20-day horizon, 2,000 paths" in res.stdout


def test_oosvar_names_the_fixture_first(tmp_path):
    from mcsim import oosvar

    import io
    from contextlib import redirect_stdout
    from unittest import mock

    fake = {"n_returns": 520, "fixture": "fixtures/ohlcv/GULFOILLUB_NS.csv"}
    for name in ("garch_t", "gaussian_control"):
        fake[name] = {"by_confidence": {}, "fits": {"n_fits": 0, "not_converged": 0},
                      "verdict": {"all_pass": False}}
    buf = io.StringIO()
    with mock.patch.object(oosvar, "run", return_value=fake), \
         mock.patch.object(oosvar, "verdict_line", return_value="criteria: -"), redirect_stdout(buf):
        oosvar.main(["--fixture", "fixtures/ohlcv/GULFOILLUB_NS.csv"])
    lines = buf.getvalue().splitlines()
    assert lines[0] == "fixture=fixtures/ohlcv/GULFOILLUB_NS.csv"

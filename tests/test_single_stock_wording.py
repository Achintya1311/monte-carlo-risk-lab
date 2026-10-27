"""P126 (CRITIQUES.md): mcsim.portfolio measures one return series, not a portfolio."""

from pathlib import Path

import mcsim.portfolio as portfolio
from mcsim.portfolio import build_parser

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def test_readme_does_not_call_it_portfolio_risk():
    assert "models portfolio risk" not in README
    assert "Day 5 - portfolio risk" not in README
    assert "single-stock" in README


def test_cli_docstring_and_description_say_single_stock():
    assert portfolio.__doc__.startswith("CLI: single-stock risk")
    assert "module name" in portfolio.__doc__
    assert build_parser().description.startswith("Single-stock VaR/CVaR")

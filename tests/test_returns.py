import numpy as np
import pytest

from mcsim.returns import annualize, load_closes, load_daily_log_returns
from mcsim.gbm import TRADING_DAYS_PER_YEAR

FIXTURE = "fixtures/ohlcv/RELIANCE_NS.csv"


def test_load_closes_matches_row_count():
    closes = load_closes(FIXTURE)
    with open(FIXTURE) as f:
        n_rows = sum(1 for _ in f) - 1  # minus header
    assert len(closes) == n_rows
    assert (closes > 0).all()


def test_load_daily_log_returns_is_one_shorter_than_closes():
    closes = load_closes(FIXTURE)
    returns = load_daily_log_returns(FIXTURE)
    assert len(returns) == len(closes) - 1


def test_load_daily_log_returns_matches_manual_diff_log():
    closes = load_closes(FIXTURE)
    expected = np.diff(np.log(closes))
    returns = load_daily_log_returns(FIXTURE)
    np.testing.assert_allclose(returns, expected)


def test_load_closes_rejects_too_short_file(tmp_path):
    p = tmp_path / "one_row.csv"
    p.write_text("date,close\n2024-01-01,100\n")
    with pytest.raises(ValueError, match="fewer than 2 rows"):
        load_closes(str(p))


def test_annualize_scales_by_trading_days():
    mu_annual, sigma_annual = annualize(mu_daily=0.001, sigma_daily=0.02)
    assert mu_annual == pytest.approx(0.001 * TRADING_DAYS_PER_YEAR)
    assert sigma_annual == pytest.approx(0.02 * np.sqrt(TRADING_DAYS_PER_YEAR))

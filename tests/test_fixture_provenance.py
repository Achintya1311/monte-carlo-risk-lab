"""D53: the equity fixtures are copies of STOCKSTALKER's yfinance bars.

Each copy is pinned by hash here, and compared with the sibling checkout
when one exists (the pattern STOCKSTALKER's own fixture provenance test
uses), so a refresh on either side fails a test instead of drifting.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "STOCKSTALKER" / "fixtures" / "ohlcv"
PINNED = {
    "RELIANCE_NS.csv": ("81d022cb3dff50dc316d94dc1a58b64947a991850c6595480165a2723b623369", "d9fe2d8"),
    "GULFOILLUB_NS.csv": ("91f0f915352e5f06bd118484fb4cec7cd8dce964d72d5c4ffc14d782d1f8d7b3", "2065467"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_sources() -> str:
    text = (ROOT / "README.md").read_text()
    return text.split("## Data sources", 1)[1].split("\n## ", 1)[0]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_equity_fixture_matches_the_pinned_hash(name):
    assert _sha256(ROOT / "fixtures" / "ohlcv" / name) == PINNED[name][0], (
        f"{name} changed: re-copy it from STOCKSTALKER and update PINNED and the README together"
    )


@pytest.mark.parametrize("name", sorted(PINNED))
def test_equity_fixture_matches_stockstalker_when_checked_out(name):
    sibling = SIBLING / name
    if not sibling.is_file():
        pytest.skip("STOCKSTALKER is not checked out next to this repo")
    assert (ROOT / "fixtures" / "ohlcv" / name).read_bytes() == sibling.read_bytes(), (
        f"{name} differs from STOCKSTALKER's copy: one side was refreshed without the other"
    )


def test_data_sources_record_commit_hash_and_adjustment():
    sources = _data_sources()
    for sha, commit in PINNED.values():
        assert sha in sources
        assert commit in sources
    assert "auto_adjust=True" in sources
    assert "back-adjusted" in sources
    assert "NSE daily closes" not in sources
    assert "- yfinance - equity and index history" not in sources


def test_gulfoillub_counts_match_the_file():
    rows = (ROOT / "fixtures" / "ohlcv" / "GULFOILLUB_NS.csv").read_text().strip().splitlines()[1:]
    assert len(rows) == 3007
    text = (ROOT / "README.md").read_text()
    assert "3,007 daily closes" in _data_sources()
    assert "~11 years" not in text
    assert "3,006 daily closes" not in text


# D54: the commodity fixtures. WTI_C1.csv is derived from the curve file's
# c1 column, so the two cannot disagree; the hashes pin the vintage.
COMMODITY = ROOT / "fixtures" / "commodity"
COMMODITY_PINNED = {
    "WTI_futures_curve.csv": "0629f3a7785617d333a1bf282fd308fc10dd07bc12ad64c90ff97b9e05e48142",
    "WTI_C1.csv": "b0c368c7b51247090c8632fc37c7b21e8e7a9cbed289fa98634cf2e717200f15",
}


@pytest.mark.parametrize("name", sorted(COMMODITY_PINNED))
def test_commodity_fixture_matches_the_pinned_hash(name):
    assert _sha256(COMMODITY / name) == COMMODITY_PINNED[name]


def test_front_month_file_is_the_curve_c1_column():
    import csv

    with open(COMMODITY / "WTI_C1.csv") as f:
        front = [(r["date"], r["close"]) for r in csv.DictReader(f)]
    with open(COMMODITY / "WTI_futures_curve.csv") as f:
        curve = [(r["date"], r["c1"]) for r in csv.DictReader(f)]
    assert front == curve


def test_readme_says_the_front_month_file_is_derived():
    sources = _data_sources()
    for sha in COMMODITY_PINNED.values():
        assert sha in sources
    assert "the `c1` column of `WTI_futures_curve.csv`" in sources
    assert "not committed" in sources
    text = (ROOT / "README.md").read_text()
    assert "holds by construction for the committed files" in text

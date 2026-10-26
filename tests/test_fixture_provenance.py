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

"""P102, P103, P104: README statements that must match the repo and its git record."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _limitations() -> str:
    return README.split("## Limitations and what would make me wrong", 1)[1].split("\n## ", 1)[0]


def test_limitations_do_not_deny_the_garch_t_model():
    lim = _limitations()
    assert "this repo doesn't build one" not in lim
    assert "Nothing here adapts the distributional assumption" not in lim
    assert "GARCH(1,1)-t" in lim
    assert "only for the 1-day out-of-sample backtest" in lim


def test_out_of_sample_pointer_points_up():
    assert "see the out-of-sample section below" not in README
    assert "see the out-of-sample section above" in README


def test_status_line_names_the_october_work():
    status = next(line for line in README.splitlines() if line.startswith("**Status:**"))
    assert "Oct 2026" in status


def test_preregistration_claim_matches_the_git_record():
    assert "committed before any number was computed" not in README
    for token in ("2a259fc", "f3a3a67", "16:29:56", "16:32:18", "order"):
        assert token in README


def test_checkpoint_rows_carry_their_commits():
    log = README.split("<!-- CHECKPOINTS:START -->", 1)[1].split("<!-- CHECKPOINTS:END -->", 1)[0]
    assert "(pending)" not in log
    assert "| 2026-10-17 | `0cfb5d9` |" in log
    assert "| 2026-10-12 | `66962ff` |" in log

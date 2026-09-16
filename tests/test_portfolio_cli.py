import subprocess
import sys

FIXTURE = "fixtures/ohlcv/RELIANCE_NS.csv"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mcsim.portfolio", *args],
        capture_output=True,
        text=True,
    )


def test_cli_reports_all_three_methods_at_each_confidence():
    result = run_cli(
        "--fixture", FIXTURE, "--horizon-days", "20",
        "--confidence", "0.95", "0.99", "--paths", "50000", "--seed", "42",
    )
    assert result.returncode in (0, 1), result.stderr
    for label in ("historical", "parametric", "mc"):
        assert label in result.stdout
    assert "95%" in result.stdout
    assert "99%" in result.stdout
    assert "Kupiec" in result.stdout
    assert "drawdown" in result.stdout.lower()


def test_cli_same_seed_is_reproducible():
    args = ("--fixture", FIXTURE, "--horizon-days", "10", "--confidence", "0.95", "--paths", "20000", "--seed", "7")
    a = run_cli(*args)
    b = run_cli(*args)
    assert a.stdout == b.stdout
    assert a.returncode == b.returncode


def test_cli_different_seed_changes_mc_output_only():
    common = ("--fixture", FIXTURE, "--horizon-days", "10", "--confidence", "0.95", "--paths", "20000")
    a = run_cli(*common, "--seed", "1")
    b = run_cli(*common, "--seed", "2")
    assert a.stdout != b.stdout


def test_cli_writes_drawdown_plot(tmp_path):
    out = tmp_path / "drawdown.png"
    result = run_cli(
        "--fixture", FIXTURE, "--horizon-days", "20", "--confidence", "0.95",
        "--paths", "50000", "--seed", "42", "--drawdown-plot", str(out),
    )
    assert result.returncode in (0, 1), result.stderr
    assert out.exists()
    assert out.stat().st_size > 0
    assert "wrote drawdown distribution plot" in result.stdout


def test_cli_rejects_invalid_confidence():
    result = run_cli(
        "--fixture", FIXTURE, "--horizon-days", "20", "--confidence", "1.5",
        "--paths", "1000", "--seed", "1",
    )
    assert result.returncode != 0
    assert "confidence" in (result.stdout + result.stderr).lower()


def test_cli_default_reliance_fixture_kupiec_99pct_is_rejected_and_flagged():
    # Deterministic against the committed fixture: the GBM/Gaussian model's
    # thin tails underestimate real exceptions at the 99% level (see README
    # limitations) - this is the "honest negative result" the repo commits to
    # surfacing rather than hiding.
    result = run_cli(
        "--fixture", FIXTURE, "--horizon-days", "20",
        "--confidence", "0.95", "0.99", "--paths", "50000", "--seed", "42",
    )
    assert "REJECTED" in result.stdout
    assert result.returncode == 1
    assert "NOTE" in result.stderr

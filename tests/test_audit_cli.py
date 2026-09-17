import subprocess
import sys

CURVE = "fixtures/commodity/WTI_futures_curve.csv"
PRICES = "fixtures/commodity/WTI_C1.csv"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mcsim.audit", *args],
        capture_output=True,
        text=True,
    )


def test_cli_clean_on_committed_fixtures(tmp_path):
    result = run_cli(
        "--curve", CURVE, "--prices", PRICES, "--output-dir", str(tmp_path), "--date", "2026-09-17",
    )
    assert result.returncode == 0, result.stderr
    assert "causality" in result.stdout
    assert "alignment" in result.stdout
    assert "no look-ahead detected" in result.stdout

    report_path = tmp_path / "audit_2026-09-17.md"
    assert report_path.exists()
    text = report_path.read_text()
    assert "clean" in text
    assert "holds" in text


def test_cli_same_inputs_are_reproducible(tmp_path):
    args = ("--curve", CURVE, "--prices", PRICES, "--output-dir", str(tmp_path), "--date", "2026-09-17")
    a = run_cli(*args)
    b = run_cli(*args)
    assert a.stdout == b.stdout
    assert a.returncode == b.returncode == 0

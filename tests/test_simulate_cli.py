import csv
import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mcsim.simulate", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_and_reports_convergence():
    result = run_cli(
        "--spot", "100", "--vol", "0.25", "--rate", "0.07",
        "--days", "30", "--paths", "200000", "--seed", "42",
    )
    assert result.returncode == 0, result.stderr
    assert "sample mean terminal price" in result.stdout
    assert "relative error" in result.stdout
    assert result.stderr == ""  # no convergence warning at 200k paths


def test_cli_same_seed_is_reproducible():
    a = run_cli("--spot", "100", "--vol", "0.25", "--rate", "0.07", "--days", "10", "--paths", "1000", "--seed", "7")
    b = run_cli("--spot", "100", "--vol", "0.25", "--rate", "0.07", "--days", "10", "--paths", "1000", "--seed", "7")
    assert a.stdout == b.stdout


def test_cli_writes_terminal_prices_csv(tmp_path):
    out = tmp_path / "terminals.csv"
    result = run_cli(
        "--spot", "100", "--vol", "0.2", "--rate", "0.05",
        "--days", "5", "--paths", "500", "--seed", "1", "--out", str(out),
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["terminal_price"]
    assert len(rows) == 501  # header + 500 paths
    assert all(float(r[0]) > 0 for r in rows[1:])

import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mcsim.reduce", *args],
        capture_output=True,
        text=True,
    )


def test_cli_reports_all_three_methods():
    result = run_cli(
        "--spot", "100", "--strike", "105", "--vol", "0.25", "--rate", "0.07",
        "--days", "30", "--paths", "50000", "--seed", "42",
    )
    assert result.returncode == 0, result.stderr
    assert "Black-Scholes price" in result.stdout
    assert "plain" in result.stdout
    assert "antithetic" in result.stdout
    assert "control_variate" in result.stdout


def test_cli_same_seed_is_reproducible():
    args = (
        "--spot", "100", "--strike", "100", "--vol", "0.2", "--rate", "0.05",
        "--days", "10", "--paths", "2000", "--seed", "7",
    )
    a = run_cli(*args)
    b = run_cli(*args)
    assert a.stdout == b.stdout


def test_cli_supports_put_option_type():
    result = run_cli(
        "--spot", "100", "--strike", "95", "--vol", "0.3", "--rate", "0.05",
        "--days", "60", "--paths", "50000", "--seed", "1", "--option-type", "put",
    )
    assert result.returncode == 0, result.stderr
    assert "option=put" in result.stdout


def test_cli_rejects_invalid_option_type():
    result = run_cli(
        "--spot", "100", "--strike", "100", "--vol", "0.2", "--rate", "0.05",
        "--days", "10", "--paths", "1000", "--seed", "1", "--option-type", "straddle",
    )
    assert result.returncode != 0

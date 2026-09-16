import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mcsim.exotic", *args],
        capture_output=True,
        text=True,
    )


def test_asian_cli_reports_arithmetic_and_geometric():
    result = run_cli(
        "asian", "--spot", "100", "--strike", "100", "--vol", "0.25", "--rate", "0.07",
        "--days", "30", "--paths", "50000", "--seed", "42",
    )
    assert result.returncode == 0, result.stderr
    assert "Arithmetic Asian" in result.stdout
    assert "Geometric Asian (MC)" in result.stdout
    assert "Geometric Asian (closed)" in result.stdout


def test_asian_cli_same_seed_is_reproducible():
    args = ("asian", "--spot", "100", "--strike", "100", "--vol", "0.2", "--rate", "0.05",
            "--days", "10", "--paths", "2000", "--seed", "7")
    a = run_cli(*args)
    b = run_cli(*args)
    assert a.stdout == b.stdout


def test_asian_cli_supports_put_option_type():
    result = run_cli(
        "asian", "--spot", "100", "--strike", "95", "--vol", "0.3", "--rate", "0.05",
        "--days", "60", "--paths", "50000", "--seed", "1", "--option-type", "put",
    )
    assert result.returncode == 0, result.stderr


def test_barrier_cli_reports_in_out_and_identity():
    result = run_cli(
        "barrier", "--spot", "100", "--strike", "100", "--barrier", "120", "--vol", "0.25",
        "--rate", "0.07", "--days", "30", "--paths", "50000", "--seed", "42", "--direction", "up",
    )
    assert result.returncode == 0, result.stderr
    assert "Knock-in" in result.stdout
    assert "Knock-out" in result.stdout
    assert "identity gap" in result.stdout


def test_barrier_cli_rejects_up_barrier_below_spot():
    result = run_cli(
        "barrier", "--spot", "100", "--strike", "100", "--barrier", "90", "--vol", "0.25",
        "--rate", "0.07", "--days", "30", "--paths", "1000", "--seed", "1", "--direction", "up",
    )
    assert result.returncode != 0


def test_cli_requires_subcommand():
    result = run_cli()
    assert result.returncode != 0

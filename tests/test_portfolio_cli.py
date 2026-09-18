import json
import subprocess
import sys

from mcsim.portfolio import build_parser, run, to_contract

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


def _run(**overrides):
    args = build_parser().parse_args(
        ["--fixture", FIXTURE, "--horizon-days", "20", "--confidence", "0.95", "0.99", "--paths", "50000", "--seed", "42"]
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return run(args)


def test_to_contract_matches_the_v03_shape_next_steps_committed_to():
    result = _run()
    contract = to_contract(result)
    assert set(contract) == {"risk"}
    risk = contract["risk"]
    assert set(risk) == {"var_95", "cvar_95", "max_dd_sim", "horizon_days"}
    assert risk["horizon_days"] == 20
    row_95 = next(r for r in result["by_confidence"] if r["confidence"] == 0.95)
    assert risk["var_95"] == row_95["historical"]["var"]
    assert risk["cvar_95"] == row_95["historical"]["cvar"]
    assert risk["max_dd_sim"] == result["drawdown_stats"]["mean"]


def test_to_contract_requires_a_95pct_confidence_level():
    result = _run(confidence=[0.99])
    try:
        to_contract(result)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "0.95" in str(exc)


def test_cli_writes_the_risk_contract(tmp_path):
    out = tmp_path / "risk_contract.json"
    result = run_cli(
        "--fixture", FIXTURE, "--horizon-days", "20", "--confidence", "0.95", "0.99",
        "--paths", "50000", "--seed", "42", "--contract", str(out),
    )
    assert result.returncode in (0, 1), result.stderr
    assert "wrote risk contract" in result.stdout

    contract = json.loads(out.read_text())
    assert set(contract["risk"]) == {"var_95", "cvar_95", "max_dd_sim", "horizon_days"}
    assert contract["risk"]["horizon_days"] == 20

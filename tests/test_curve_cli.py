import subprocess
import sys

CURVE = "fixtures/commodity/WTI_futures_curve.csv"
PRICES = "fixtures/commodity/WTI_C1.csv"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mcsim.curve", *args],
        capture_output=True,
        text=True,
    )


def test_cli_reports_regime_mix_and_identity_gap():
    result = run_cli("--curve", CURVE, "--prices", PRICES, "--horizon-days", "20", "--confidence", "0.95")
    assert result.returncode == 0, result.stderr
    assert "contango" in result.stdout
    assert "backwardation" in result.stdout
    assert "identity gap" in result.stdout
    assert "current regime" in result.stdout


def test_cli_reports_var_cvar_for_both_regimes_at_default_horizon():
    # The committed fixture has enough windows of each regime at a 20-day
    # horizon for both to clear MIN_REGIME_WINDOWS - if this regresses, the
    # regime-conditioning has stopped doing anything useful for this fixture.
    result = run_cli("--curve", CURVE, "--prices", PRICES, "--horizon-days", "20", "--confidence", "0.95")
    lines = {ln.split(":")[0].strip(): ln for ln in result.stdout.splitlines() if ":" in ln}
    assert "VaR=" in lines.get("contango", "")
    assert "VaR=" in lines.get("backwardation", "")


def test_cli_same_inputs_are_reproducible():
    args = ("--curve", CURVE, "--prices", PRICES, "--horizon-days", "10", "--confidence", "0.95")
    a = run_cli(*args)
    b = run_cli(*args)
    assert a.stdout == b.stdout
    assert a.returncode == b.returncode == 0


def test_cli_rejects_invalid_confidence():
    result = run_cli("--curve", CURVE, "--prices", PRICES, "--horizon-days", "20", "--confidence", "1.5")
    assert result.returncode != 0
    assert "confidence" in (result.stdout + result.stderr).lower()

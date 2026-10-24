from mcsim import oosvar


def test_cli_runs_on_the_committed_fixture_and_reports_both_models(capsys):
    res = oosvar.run("fixtures/ohlcv/RELIANCE_NS.csv", window=250, refit_every=20)
    assert set(res) >= {"garch_t", "gaussian_control"} and res["garch_t"]["by_confidence"]["0.99"]["n_test"] == 250
    assert isinstance(res["garch_t"]["verdict"]["all_pass"], bool)

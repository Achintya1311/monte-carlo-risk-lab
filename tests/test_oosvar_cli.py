from mcsim import oosvar


def test_cli_runs_on_the_committed_fixture_and_reports_both_models(capsys):
    res = oosvar.run("fixtures/ohlcv/RELIANCE_NS.csv", window=250, refit_every=20)
    assert set(res) >= {"garch_t", "gaussian_control"} and res["garch_t"]["by_confidence"]["0.99"]["n_test"] == 250
    assert isinstance(res["garch_t"]["verdict"]["all_pass"], bool)


def test_main_prints_the_sebi_disclaimer(capsys):
    oosvar.main(["--fixture", "fixtures/ohlcv/RELIANCE_NS.csv", "--window", "250", "--refit-every", "20"])
    out = capsys.readouterr().out
    assert (
        "Educational research on historical data. Not investment advice. "
        "Not a SEBI-registered research analyst."
    ) in out


GULF = "fixtures/ohlcv/GULFOILLUB_NS.csv"


def test_preregistered_gulfoillub_result_is_pinned():
    # P79: the README's lead v2 finding, under docs/PREREG_oos_var.md's fixed configuration.
    res = oosvar.run(GULF, window=500, refit_every=20)
    g, c = res["garch_t"], res["gaussian_control"]
    for m in (g, c):
        assert m["by_confidence"]["0.95"]["n_test"] == 2506 and m["by_confidence"]["0.99"]["n_test"] == 2506
    assert g["by_confidence"]["0.95"]["n_exceptions"] == 111
    assert g["by_confidence"]["0.99"]["n_exceptions"] == 21
    assert c["by_confidence"]["0.95"]["n_exceptions"] == 101
    assert c["by_confidence"]["0.99"]["n_exceptions"] == 42
    assert round(g["by_confidence"]["0.99"]["kupiec_p"], 3) == 0.402
    assert round(g["by_confidence"]["0.99"]["p_cc"], 3) == 0.589
    assert round(g["by_confidence"]["0.95"]["kupiec_p"], 3) == 0.182
    assert round(g["by_confidence"]["0.95"]["p_cc"], 3) == 0.166
    assert round(c["by_confidence"]["0.99"]["kupiec_p"], 3) == 0.002
    assert g["verdict"]["all_pass"] is True and c["verdict"]["all_pass"] is False
    # P80: every GARCH fit converged; a silent non-converged fit would feed the verdict.
    assert g["fits"] == {"n_fits": 126, "not_converged": 0}
    assert c["fits"]["not_converged"] == 0


def test_non_converged_garch_fits_are_counted(monkeypatch):
    class NeverConverges(oosvar.GarchTModel):
        def __init__(self, train):
            super().__init__(train)
            self.converged = False

    monkeypatch.setattr(oosvar, "GarchTModel", NeverConverges)
    res = oosvar.run("fixtures/ohlcv/RELIANCE_NS.csv", window=250, refit_every=50)
    assert res["garch_t"]["fits"]["n_fits"] == res["garch_t"]["fits"]["not_converged"] > 0


def test_cli_prints_a_readable_verdict_and_fit_count(capsys):
    # P81: one line per pre-registered criterion, not a raw Python dict.
    oosvar.main(["--fixture", "fixtures/ohlcv/RELIANCE_NS.csv", "--window", "250", "--refit-every", "50"])
    out = capsys.readouterr().out
    assert "{'" not in out
    assert "criteria: Kupiec 95% " in out and "conditional coverage 99% " in out and "99% rate in 0.5-1.5% " in out
    assert "overall: " in out
    assert "fits: " in out and " not converged" in out

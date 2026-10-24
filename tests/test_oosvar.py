import numpy as np
import pytest

from mcsim.garch import GarchTModel, GaussianModel
from mcsim.risk import christoffersen_tests, k_lr, kupiec_pof_test, rolling_var_backtest


def test_independent_breaches_are_not_flagged_but_clustered_ones_are():
    rng = np.random.default_rng(0)
    iid = (rng.random(4000) < 0.05).astype(int)
    assert christoffersen_tests(iid, 0.95)["p_ind"] > 0.01
    clustered = np.zeros(4000, int)
    for s in range(0, 4000, 80):
        clustered[s:s + 4] = 1                                    # 5% breach rate, but in bursts of four
    ch = christoffersen_tests(clustered, 0.95)
    assert ch["p_ind"] < 1e-6 and ch["p_cc"] < 1e-6


def test_k_lr_matches_the_existing_kupiec_test():
    r = np.r_[np.full(95, 0.01), np.full(5, -0.05)]
    old = kupiec_pof_test(r, -0.02, 0.95)
    assert k_lr(5, 100, 0.05) == pytest.approx(old["lr_stat"], abs=1e-9)


def test_rolling_backtest_never_shows_the_model_the_day_it_forecasts():
    seen = []

    class Spy:
        def __init__(self, train): self.train = train
        def var(self, history, confidences):
            seen.append((history[-1], len(history)))
            return {c: -1.0 for c in confidences}

    r = np.arange(1.0, 61.0) / 1000.0                              # strictly increasing, so a leak is visible
    rolling_var_backtest(r, Spy, window=20, refit_every=7, confidences=(0.95,))
    for t, (last_seen, n) in zip(range(20, 60), seen):
        assert n == 20 and last_seen == pytest.approx(r[t - 1])    # history ends the day BEFORE day t


def test_gaussian_model_is_calibrated_on_gaussian_data():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0004, 0.01, 4000)
    res = rolling_var_backtest(r, GaussianModel, window=250, refit_every=20, confidences=(0.95, 0.99))
    assert res[0.95]["kupiec_p"] > 0.01 and res[0.99]["kupiec_p"] > 0.01


def test_garch_t_beats_the_gaussian_when_volatility_clusters():
    rng = np.random.default_rng(11)
    n, sig2, out = 2600, 1e-4, []
    for _ in range(n):                                             # true GARCH(1,1) with fat-tailed shocks
        e = np.sqrt(sig2) * rng.standard_t(6) / np.sqrt(6 / 4)
        out.append(e)
        sig2 = 2e-6 + 0.1 * e * e + 0.88 * sig2
    r = np.array(out)
    g = rolling_var_backtest(r, GarchTModel, window=500, refit_every=20, confidences=(0.99,))[0.99]
    c = rolling_var_backtest(r, GaussianModel, window=500, refit_every=20, confidences=(0.99,))[0.99]
    assert g["p_cc"] > c["p_cc"]                                   # the conditional model handles clustering

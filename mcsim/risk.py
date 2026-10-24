"""Portfolio risk: VaR and CVaR by three methods, a Monte Carlo drawdown
distribution, and a Kupiec proportion-of-failures backtest.

Every VaR/CVaR figure here is a *return* (e.g. -0.032 = a 3.2% loss), the
same sign convention as the contract in ``NEXT_STEPS.md``: more negative
means worse, and CVaR is always at or below its VaR at the same confidence
because it is the average of the tail VaR only cuts off.

Historical, parametric and Monte Carlo answer different questions about
the same tail:

- **Historical** replays the actual (non-normal, autocorrelated,
  fat-tailed) sample - no distributional assumption, but only as many
  independent multi-day observations as the fixture window allows.
- **Parametric** assumes daily log returns are iid Gaussian and scales
  the fitted mean/std to the horizon in closed form - cheap and stable,
  blind to skew and fat tails by construction.
- **Monte Carlo** simulates GBM paths under the *physical* (real-world)
  drift fitted from the same historical sample - same Gaussian-increment
  assumption as parametric, but priced through simulation instead of a
  closed form, which is what lets it also produce the drawdown
  distribution below (a path-dependent quantity closed-form VaR cannot
  give you).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

from mcsim.gbm import simulate_gbm_paths


def historical_var_cvar(daily_log_returns: np.ndarray, confidence: float, horizon_days: int) -> dict:
    """Historical simulation VaR/CVaR from overlapping horizon-day windows.

    Overlapping windows are the only way to get more than one horizon-day
    observation out of a short fixture, at the cost of adjacent windows
    sharing most of their days - they are not independent draws, which
    understates the true sampling uncertainty of the quantile. Flagged in
    the README, not hidden.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if len(daily_log_returns) < horizon_days:
        raise ValueError(
            f"need at least {horizon_days} daily returns for a {horizon_days}-day window, "
            f"got {len(daily_log_returns)}"
        )

    cumulative = np.convolve(daily_log_returns, np.ones(horizon_days), mode="valid")
    n_windows = len(cumulative)

    var = float(np.quantile(cumulative, 1.0 - confidence))
    tail = cumulative[cumulative <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var

    return {"var": var, "cvar": cvar, "n_windows": n_windows, "overlapping": True}


def parametric_var_cvar(mu_daily: float, sigma_daily: float, confidence: float, horizon_days: int) -> dict:
    """Closed-form Gaussian VaR and expected shortfall (CVaR), horizon-scaled.

    Scales the fitted daily mean/std to the horizon under the iid-Gaussian
    assumption (mu_h = mu*h, sigma_h = sigma*sqrt(h)), then uses the known
    normal-tail formulas:

        VaR_c  = mu_h + sigma_h * z
        CVaR_c = mu_h - sigma_h * phi(z) / (1 - c)

    where z = Phi^-1(1 - c) and phi is the standard normal density.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if sigma_daily <= 0:
        raise ValueError(f"sigma_daily must be positive, got {sigma_daily}")

    mu_h = mu_daily * horizon_days
    sigma_h = sigma_daily * np.sqrt(horizon_days)
    z = norm.ppf(1.0 - confidence)

    var = float(mu_h + sigma_h * z)
    cvar = float(mu_h - sigma_h * norm.pdf(z) / (1.0 - confidence))
    return {"var": var, "cvar": cvar, "mu_h": float(mu_h), "sigma_h": float(sigma_h)}


def simulate_risk_paths(
    spot: float, mu_annual: float, sigma_annual: float, horizon_days: int, n_paths: int, seed: int | None
) -> np.ndarray:
    """GBM paths under the *physical* drift, for risk measurement rather than pricing.

    Same simulator as Days 1-4 (``mcsim.gbm.simulate_gbm_paths``), but the
    drift passed in is the sample mean *log* return (``mu_annual =
    mu_daily * 252``, see ``mcsim.returns.annualize``), not a risk-neutral
    *rate* - there is no hedgeable payoff here to price under Q, only a
    real-world loss probability to estimate under P.

    ``simulate_gbm_paths`` always subtracts ``0.5*vol**2*dt`` from its
    ``rate`` argument to turn an arithmetic growth rate into a log-return
    mean (correct for Days 1-4, where ``rate`` is the risk-free rate).
    ``mu_annual`` here is already a log-return mean with no such gap to
    remove, so passing it straight through would subtract the convexity
    term a second time (D14). Adding it back before the call cancels that
    subtraction, so the simulated paths' own mean log return matches
    ``mu_annual`` exactly, as a physical-measure simulation requires.
    """
    return simulate_gbm_paths(
        spot=spot, rate=mu_annual + 0.5 * sigma_annual**2, vol=sigma_annual,
        days=horizon_days, n_paths=n_paths, seed=seed,
    )


def mc_var_cvar_from_paths(paths: np.ndarray, spot: float, confidence: float) -> dict:
    """VaR/CVaR from the empirical quantile of simulated horizon log returns."""
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    terminals = paths[:, -1]
    returns = np.log(terminals / spot)

    var = float(np.quantile(returns, 1.0 - confidence))
    tail = returns[returns <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return {"var": var, "cvar": cvar, "n_paths": paths.shape[0]}


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Per-path maximum drawdown: min over t of (S_t / running-max(S_0..t) - 1).

    Vectorized across all paths at once. A drawdown of -0.21 means the
    path fell 21% from its own running peak at some point in the horizon.
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdown = paths / running_max - 1.0
    return drawdown.min(axis=1)


def realized_drawdown_horizon_matched(closes: np.ndarray, horizon_days: int) -> dict:
    """Max drawdown over every overlapping ``horizon_days``-length window of the
    realized price history, instead of over the whole fixture window.

    Comparing the simulated max-drawdown distribution (built at ``horizon_days``)
    against the single realized drawdown over the *entire* fixture is not
    apples-to-apples: a multi-year drawdown is mechanically deeper than a 20-day
    one regardless of tail shape (D20). This uses the same overlapping-window
    construction ``historical_var_cvar`` already uses for its quantile estimate --
    adjacent windows share most of their days, so these are not independent draws
    -- applied to price windows instead of summed log returns. A window spans
    ``horizon_days + 1`` prices (``horizon_days`` daily returns), the same path
    length ``simulate_risk_paths`` produces for the same horizon, so the two are
    genuinely comparable.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    window = horizon_days + 1
    if len(closes) < window:
        raise ValueError(
            f"need at least {window} closes for a {horizon_days}-day window, "
            f"got {len(closes)}"
        )

    windows = np.lib.stride_tricks.sliding_window_view(closes, window)
    per_window_dd = max_drawdowns(windows)

    return {
        "horizon_days": horizon_days,
        "n_windows": int(len(per_window_dd)),
        "mean": float(per_window_dd.mean()),
        "worst": float(per_window_dd.min()),
    }


def drawdown_distribution_stats(drawdowns: np.ndarray) -> dict:
    """Summary of a per-path max-drawdown sample: not one number, a distribution."""
    percentiles = {p: float(np.percentile(drawdowns, p)) for p in (5, 25, 50, 75, 95)}
    return {
        "n_paths": int(len(drawdowns)),
        "mean": float(drawdowns.mean()),
        "worst": float(drawdowns.min()),
        "percentiles": percentiles,
    }


def plot_drawdown_distribution(drawdowns: np.ndarray, out_path: str) -> None:
    """Write a histogram of the simulated per-path max-drawdown distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(drawdowns, bins=60, color="#4c72b0", edgecolor="none")
    ax.axvline(float(np.median(drawdowns)), color="black", linestyle="--", label=f"median = {np.median(drawdowns):.2%}")
    ax.axvline(float(drawdowns.min()), color="firebrick", linestyle=":", label=f"worst = {drawdowns.min():.2%}")
    ax.set_xlabel("max drawdown over horizon")
    ax.set_ylabel("simulated paths")
    ax.set_title("Monte Carlo max-drawdown distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def kupiec_pof_test(daily_log_returns: np.ndarray, var_threshold: float, confidence: float) -> dict:
    """Kupiec (1995) proportion-of-failures backtest for a 1-day VaR model.

    An "exception" is a day the realized return breached the VaR
    threshold. Under a correctly calibrated model, exceptions should occur
    at rate ``p = 1 - confidence``. The likelihood-ratio statistic

        LR = -2 * ln[ (1-p)^(n-x) p^x ] + 2 * ln[ (1-x/n)^(n-x) (x/n)^x ]

    is asymptotically chi-squared with 1 degree of freedom under the null
    that the true exception rate is p; a small p-value rejects the model,
    in either direction (too many *or* too few exceptions).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n = len(daily_log_returns)
    p = 1.0 - confidence
    exceptions = daily_log_returns < var_threshold
    x = int(exceptions.sum())
    x_hat = x / n

    log_l_null = (n - x) * np.log(1.0 - p) + x * np.log(p) if 0 < p < 1 else 0.0
    if x == 0:
        log_l_alt = (n - x) * np.log(1.0 - x_hat)
    elif x == n:
        log_l_alt = x * np.log(x_hat)
    else:
        log_l_alt = (n - x) * np.log(1.0 - x_hat) + x * np.log(x_hat)

    lr_stat = float(-2.0 * (log_l_null - log_l_alt))
    p_value = float(1.0 - chi2.cdf(lr_stat, df=1))

    return {
        "n_obs": n,
        "n_exceptions": x,
        "expected_exceptions": p * n,
        "exception_rate": x_hat,
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject_at_5pct": p_value < 0.05,
    }


def christoffersen_tests(exceptions: np.ndarray, confidence: float) -> dict:
    """Christoffersen (1998) independence and conditional-coverage tests on a 0/1 exception series.

    Kupiec only counts breaches. Christoffersen asks whether they arrive in clusters: if a breach makes
    the next day's breach more likely than baseline, the model is not tracking volatility even when the
    total count looks right. ``lr_ind`` compares a first-order Markov transition matrix with a single
    breach probability (chi-squared, 1 df). ``lr_cc`` = Kupiec LR + ``lr_ind`` (chi-squared, 2 df).
    """
    e = np.asarray(exceptions, dtype=int)
    if len(e) < 2:
        raise ValueError("need at least 2 observations")
    p = 1.0 - confidence
    n = len(e)
    x = int(e.sum())

    def ll(k, m):  # log-likelihood of k events in m trials at the MLE rate, 0*log(0) = 0
        if m == 0:
            return 0.0
        q = k / m
        return (k * np.log(q) if k else 0.0) + ((m - k) * np.log(1 - q) if m - k else 0.0)

    prev, cur = e[:-1], e[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    l_markov = ll(n01, n00 + n01) + ll(n11, n10 + n11)
    l_iid = ll(n01 + n11, n00 + n01 + n10 + n11)
    lr_ind = max(float(-2.0 * (l_iid - l_markov)), 0.0)

    l_null = (n - x) * np.log(1.0 - p) + (x * np.log(p) if x else 0.0)
    lr_pof = max(float(-2.0 * (l_null - ll(x, n))), 0.0)
    lr_cc = lr_pof + lr_ind
    return {
        "n_obs": n, "n_exceptions": x, "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "lr_ind": lr_ind, "p_ind": float(1.0 - chi2.cdf(lr_ind, df=1)),
        "lr_cc": lr_cc, "p_cc": float(1.0 - chi2.cdf(lr_cc, df=2)),
    }


def rolling_var_backtest(returns: np.ndarray, fit_fn, window: int, refit_every: int, confidences: tuple[float, ...]) -> dict:
    """Out-of-sample 1-day VaR backtest with no look-ahead.

    Parameters are refit every ``refit_every`` days on the preceding ``window`` returns:
    ``model = fit_fn(train)``. Every day t is then forecast as ``model.var(history, confidences)``, where
    ``history`` is the ``window`` returns strictly before day t, so the volatility estimate can react to the
    shocks of the days since the last refit without ever seeing day t itself.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) <= window:
        raise ValueError(f"need more than {window} returns, got {len(r)}")
    var = {c: np.full(len(r), np.nan) for c in confidences}
    for start in range(window, len(r), refit_every):
        model = fit_fn(r[start - window:start])
        for t in range(start, min(start + refit_every, len(r))):
            fc = model.var(r[t - window:t], confidences)
            for c in confidences:
                var[c][t] = fc[c]
    out = {}
    for c in confidences:
        mask = ~np.isnan(var[c])
        exc = (r[mask] < var[c][mask]).astype(int)
        x, n, p = int(exc.sum()), len(exc), 1.0 - c
        lr = k_lr(x, n, p)
        ch = christoffersen_tests(exc, c)
        out[c] = {"n_test": n, "n_exceptions": x, "expected_exceptions": p * n, "exception_rate": x / n,
                  "kupiec_lr": lr, "kupiec_p": float(1.0 - chi2.cdf(lr, df=1)),
                  "p_ind": ch["p_ind"], "lr_cc": ch["lr_cc"], "p_cc": ch["p_cc"], "transitions": ch["transitions"]}
    return out


def k_lr(x: int, n: int, p: float) -> float:
    """Kupiec likelihood-ratio statistic from a breach count (shared by the rolling backtest)."""
    x_hat = x / n
    log_null = (n - x) * np.log(1.0 - p) + (x * np.log(p) if x else 0.0)
    log_alt = ((n - x) * np.log(1.0 - x_hat) if x < n else 0.0) + (x * np.log(x_hat) if x else 0.0)
    return max(float(-2.0 * (log_null - log_alt)), 0.0)

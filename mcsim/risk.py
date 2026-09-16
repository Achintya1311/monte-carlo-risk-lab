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
    drift passed in is the sample mean return, not a risk-neutral rate -
    there is no hedgeable payoff here to price under Q, only a real-world
    loss probability to estimate under P.
    """
    return simulate_gbm_paths(spot=spot, rate=mu_annual, vol=sigma_annual, days=horizon_days, n_paths=n_paths, seed=seed)


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

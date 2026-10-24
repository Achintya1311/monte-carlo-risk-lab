# Pre-registration: out-of-sample VaR backtest (written before any result was computed)

**Why.** The v1 Kupiec test fits one constant-volatility Gaussian on the whole sample and tests it on the same data. That is in-sample, ignores volatility clustering and fat tails, and does not test the independence of breaches. The rejection it produced says the Gaussian is a poor tail model. It does not say whether a better model passes. This test asks that question honestly.

**Fixed in advance, no tuning afterwards:**

- Data: committed fixtures `fixtures/ohlcv/GULFOILLUB_NS.csv` (primary) and `RELIANCE_NS.csv` (secondary, reported for completeness; its 500 returns leave about 250 test days, so its 99% test has almost no power).
- Quantity tested: the 1-day VaR, at 95% and 99%, on daily log returns.
- Procedure: rolling window, no look-ahead. At day t the model uses only the previous `window` returns, forecasts day t's VaR, and day t is then compared with it. Windows: 500 returns for GULFOILLUB, 250 for RELIANCE. Parameters are refit every 20 days.
- Candidate model: GARCH(1,1) with Student-t innovations and a constant mean (`arch` package, returns scaled by 100), one-step-ahead conditional mean and volatility, Student-t quantile.
- Control model: constant-volatility Gaussian fitted on the same rolling window (the v1 model made out-of-sample).
- Tests: Kupiec proportion of failures, Christoffersen independence, and their sum, the conditional-coverage test (chi-squared, 2 degrees of freedom). Significance level 0.05.

**Pass criteria (all must hold, on GULFOILLUB, the primary series):**

1. Kupiec p >= 0.05 at 95% and at 99%.
2. Conditional-coverage p >= 0.05 at 95% and at 99%.
3. The 99% breach rate is between 0.5% and 1.5%.

**Reporting rules.** The Gaussian control is reported whatever the result. RELIANCE is reported with a power statement and is not part of the pass decision. If the GARCH-t model fails any criterion, the failure is published and no further model changes are made under this pre-registration. A new model would need its own pre-registration and would count as another trial.

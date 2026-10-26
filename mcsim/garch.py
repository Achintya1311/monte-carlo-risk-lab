"""Models for the rolling out-of-sample VaR backtest (``mcsim.risk.rolling_var_backtest``).

Both have the interface ``fit(train) -> model`` and ``model.var(history, confidences) -> {c: VaR}``, so the
comparison changes one thing at a time:

- ``GaussianModel``: constant-volatility Gaussian fitted on the training window. This is the v1 model
  (a Gaussian fitted to the sample) made honest by fitting it only on past data.
- ``GarchTModel``: GARCH(1,1) with Student-t innovations and a constant mean (the ``arch`` package, returns
  scaled by 100). The conditional volatility reacts to recent shocks (volatility clustering), and the
  Student-t allows fat tails. Parameters are fixed between refits; the variance recursion is run daily over
  the supplied history, which ends the day before the forecast.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t


class GaussianModel:
    def __init__(self, train: np.ndarray):
        self.mu, self.sigma = float(np.mean(train)), float(np.std(train, ddof=1))
        self.converged = True  # closed form, nothing to converge

    def var(self, history: np.ndarray, confidences) -> dict:
        return {c: self.mu + self.sigma * norm.ppf(1.0 - c) for c in confidences}


class GarchTModel:
    def __init__(self, train: np.ndarray):
        from arch import arch_model

        y = 100.0 * np.asarray(train, dtype=float)
        res = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
        self.converged = int(res.convergence_flag) == 0  # P80: warnings are off, so read the flag
        p = res.params
        self.mu, self.omega, self.alpha, self.beta, self.nu = (float(p[k]) for k in ("mu", "omega", "alpha[1]", "beta[1]", "nu"))
        self.sigma2_0 = float(np.var(y, ddof=1))
        self.scale = np.sqrt((self.nu - 2.0) / self.nu)  # arch's Student-t is unit-variance

    def var(self, history: np.ndarray, confidences) -> dict:
        y = 100.0 * np.asarray(history, dtype=float)
        s2 = self.sigma2_0
        for e in y - self.mu:                                   # filter through every return before the forecast day
            s2 = self.omega + self.alpha * e * e + self.beta * s2
        sig = np.sqrt(s2)
        return {c: (self.mu + sig * self.scale * student_t.ppf(1.0 - c, self.nu)) / 100.0 for c in confidences}

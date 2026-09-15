# Monte Carlo risk lab

Prices options and models portfolio risk by simulation, checked against closed-form solutions so the numbers are verifiable rather than merely plausible.

**Status:** Last checkpoint 2026-09-15 · Next: Day 4 - path-dependent options (Asian, barrier) where no closed form exists

## What this is

Simulation is easy to do and easy to do wrong. Every result here is validated against something known: Monte Carlo option prices converge to Black-Scholes, and the VaR model reports its own backtest exception count.

Includes a commodity leg covering contango and backwardation, because those are the cases where a naive equity-shaped model quietly breaks.

## Correctness gate

MC price within tolerance of Black-Scholes at N = 1e6, and a Kupiec exception count reported honestly - including when the model fails it.

This is the test that decides whether the repo is finished. A result that has not passed it is a draft.

## Data sources

Every source is free. Nothing in this project requires a paid tier, a subscription, or a funded account.

- yfinance - equity and index history
- Free commodity price series for the commodity leg

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m mcsim.price --spot 100 --strike 105 --vol 0.25 --rate 0.07 --days 30 --paths 1000000
```

Runs offline against committed fixtures by default. Live data needs a key in `.env` (see `.env.example`); the fixture path is the default so nothing blocks on network access.

## Findings

**Day 1 - GBM path simulator.** `mcsim.gbm.simulate_gbm_paths` draws every path from a single vectorized `rng.standard_normal((n_paths, n_steps))` call and builds paths via the exact lognormal update (no Euler discretization bias to separate from sampling error later). At N=1,000,000 paths, spot=100, vol=0.25, rate=0.07, 30-day horizon, seed=42:

- Sample mean terminal price 100.836112 vs theoretical `S0*exp(rate*T)` 100.836815 - relative error -0.0007%.
- Sample log-return std 0.086183 vs theoretical `vol*sqrt(T)` 0.086258.

Both match closed form to well within Monte Carlo sampling error at that N, which is the whole point of checking every result here against something known before building on it. At N=1,000 the same run drifts to +0.40% relative error, which is expected and is exactly why Day 2's Black-Scholes cross-check will need a real path count, not a demo one.

**Day 2 - European option pricing, MC vs Black-Scholes.** `mcsim.blackscholes.bs_price` is the closed-form oracle (checked against a known Hull textbook value and put-call parity, `C - P = S - K*exp(-rT)`, to float precision). `mcsim.pricing.mc_option_price` discounts simulated terminal payoffs from the Day 1 simulator back to today at the same risk-free rate, so it is unbiased for the BS price by construction - any gap at large N is sampling error, not model mismatch. `mcsim.price` is the CLI that runs both and reports the gap in standard errors, plus an optional `--convergence-plot`.

At spot=100, strike=105, vol=0.25, rate=0.07, 30 days, seed=42, N=1,000,000:

- Call: Black-Scholes 1.825800 vs Monte Carlo 1.822341 (std error 0.003912) - **-0.88 std errors**.
- Put: Black-Scholes 5.954436 vs Monte Carlo 5.951674 (std error 0.006128) - **-0.45 std errors**.

Both sit comfortably inside the 4-sigma gate the CLI checks. The convergence plot (`outputs/convergence_{call,put}.png`, gitignored - regenerate with `--convergence-plot`) sweeps N = 10²...10⁶ at the same seed and shows the 95% CI band collapsing onto the Black-Scholes line: at N=100 the call price and its CI swing wildly (1.27 ± 0.59), by N=10⁴ the CI is already tight around the true value, and N≥10⁵ is visually indistinguishable from the reference line. That is the expected 1/sqrt(N) shrinkage of Monte Carlo error, not a smoothed illustration of it.

**Day 3 - variance reduction: antithetic and control variates.** `mcsim.variance_reduction` adds two estimators, each measured against the plain Day 2 estimator at the *same* path budget so the payoff is a standard-error reduction at fixed simulation cost, not a smaller number from spending more paths. Antithetic pairs each draw `Z` with `-Z` and averages the two payoffs before taking the sample variance - negatively correlated because the payoff is monotone in the terminal price. Control variate uses the discounted terminal price itself, `exp(-rate*T)*S_T`, whose mean under the risk-neutral drift is known exactly (`spot`, no simulation needed for it), with the optimal coefficient `c* = Cov(X,Y)/Var(X)` estimated from the same sample. `mcsim.reduce` is the CLI that runs all three at one N and prints the table.

At spot=100, strike=105, vol=0.25, rate=0.07, 30 days, seed=42, N=1,000,000:

| Method | Call price | Call std err | Reduction | Put price | Put std err | Reduction |
|---|---|---|---|---|---|---|
| Plain (Day 2) | 1.822341 | 0.003912 | 1.00x | 5.951674 | 0.006128 | 1.00x |
| Antithetic | 1.827980 | 0.003472 | 1.13x | 5.956020 | 0.002746 | 2.23x |
| Control variate | 1.826732 | 0.002481 | 1.58x | 5.955368 | 0.002481 | 2.47x |

All six prices sit within 1 standard error of the Black-Scholes references (call 1.825800, put 5.954436) - variance reduction did not introduce bias, only tightened the estimate. Control variate wins on both legs at this N; antithetic's edge is much weaker for the call (1.13x) than the put (2.23x) here, which is exactly the known failure mode: antithetic pairing helps most when the payoff is close to linear (or symmetric) in `Z` over the range that matters, and a call's payoff is flatter than a put's near this strike/vol/horizon, so pairing buys less. Neither technique is free lunch across every configuration, which is why the table is generated fresh per run rather than asserted as a fixed multiplier.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-15 | `49dde16` | Day 3: variance reduction, antithetic and control variates (mcsim/variance_reduction.py) wired into a new mcsim.reduce CLI printing a plain-vs-antithetic-vs-control-variate standard-error reduction table at fixed path budget. Both estimators draw the terminal price directly from the one-step lognormal formula (same distribution as summing daily increments, different RNG draws than mcsim.pricing) rather than through the Day 1 path simulator, since only the terminal price is needed for European payoffs. At spot=100/strike=105/vol=0.25/rate=0.07/30d/seed=42, N=1e6: control variate cuts std error 1.58x (call) / 2.47x (put) vs plain; antithetic cuts 1.13x (call) / 2.23x (put) - all six prices within 1 std error of Black-Scholes, so neither technique introduced bias. Recorded honestly in the README that antithetic's edge is weak for the call here (payoff less linear in Z near this strike) and that a pre-existing Day 2 edge case (zero std error on an all-zero-payoff deep-OTM run) produces a spurious tolerance warning inherited by the new CLI. 22 new tests pass (68/68 total); both call and put CLI runs executed by hand at N=1e6. Also found and fixed this repo's and the hub's local main branches stuck in stale detached HEAD (same recurring class of issue) before committing. | Day 4 - path-dependent options (Asian, barrier) where no closed form exists |
| 2026-09-15 | `a105c78` | Day 2: European option pricing, Monte Carlo vs Black-Scholes (mcsim/blackscholes.py, mcsim/pricing.py, mcsim/convergence.py) wired into a new mcsim.price CLI reporting the MC-BS gap in standard errors, with a --convergence-plot sweeping N=1e2..1e6. Black-Scholes checked against a Hull textbook value and put-call parity to float precision; MC pricer built on the Day 1 GBM simulator, discounted payoffs unbiased for BS by construction. At spot=100/strike=105/vol=0.25/rate=0.07/30d/seed=42, N=1e6: call -0.88 std errors from BS, put -0.45 std errors, both inside the 4-sigma gate. Convergence plot shows the 95% CI band collapsing onto the BS line from a wild N=100 swing down to visually indistinguishable by N=1e5. 26 new tests pass (46/46 total); both call and put CLI runs executed by hand at N=1e6 with plots generated and inspected. Also found and fixed this repo's own local main branch stuck in a stale detached HEAD (same class of issue prior checkpoints hit) before committing. | Day 3 - variance reduction: antithetic and control variates, with a standard-error reduction table showing what each bought |
| 2026-09-14 | `616f40b` | Day 1: vectorized, seed-controlled GBM path simulator (mcsim/gbm.py) using the exact lognormal update, no per-path Python loop, and a mcsim.simulate CLI that reports sample vs. closed-form terminal-price moments. At N=1e6 (spot=100, vol=0.25, rate=0.07, 30d, seed=42) sample mean is within -0.0007% of theory and log-return std matches to 4 decimals; at N=1000 the same run drifts to +0.40%, recorded in the README as the expected small-N behavior. 20 new tests pass (shape/reproducibility/positivity/invalid-input/statistical-convergence for the simulator, plus CLI subprocess tests including CSV output). | Day 2 - European option pricing by Monte Carlo and Black-Scholes closed form, with a convergence plot as N grows |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Geometric Brownian motion has thin tails and constant volatility. Real returns have neither, and the Day 1 simulator does not correct for it - that is Day 2+ territory (variance reduction doesn't fix this; a fatter-tailed process would, and this repo doesn't build one).
- The simulator uses a fixed default seed (42) for reproducibility across runs, not a fresh seed per invocation. That is deliberate for testability but means two "runs" with default arguments are the same run, not independent draws - pass `--seed` explicitly to get a new sample.
- The convergence plot is one realization per N (same seed, different path count), not an average over repeated runs at each N. The shrinking CI band is the right qualitative picture, but a single low-N point (see N=100 in the Day 2 findings above) can land anywhere inside its own wide interval - it is not a smoothed regression line.
- The Day 2 correctness gate is a fixed 4-standard-error band on a single seeded run, not a full backtest across seeds. A model with a real bug could still get lucky and land inside 4 sigma once; it would not survive being run at several seeds, which this CLI does not automate yet.
- Day 3's antithetic and control-variate estimators draw the terminal price directly from the one-step lognormal formula rather than routing through `mcsim.gbm.simulate_gbm_paths`. That is the same distribution (a sum of independent Gaussian increments is Gaussian with the summed variance) but a different draw sequence, so `mcsim.reduce` and `mcsim.price` do not use common random numbers against each other - the comparison in the table is at the same path budget `n_paths`, not against an identical underlying draw.
- Antithetic variates buy far less for a call than a put at the same strike/vol/horizon (1.13x vs 2.23x at N=1e6 above) because the technique's payoff depends on how linear the option payoff is in `Z` over the relevant range - it is not a fixed multiplier and can occasionally do almost nothing. Neither technique is validated here against deep-ITM/OTM or very short-dated cases where the payoff is closer to a step function and antithetic pairing is known to help less.
- When every simulated payoff is exactly zero (for example, a strike far enough out of the money that no path in the batch pays off), the plain estimator's own std error is exactly zero, so `sigma = diff / std_error` divides by zero and both `mcsim.price` and `mcsim.reduce` report a spurious "more than 4 standard errors from Black-Scholes" warning even though the price is exactly correct. This is a pre-existing Day 2 edge case, not something Day 3 introduced or fixed - flagged here because `mcsim.reduce` inherits it on all three methods.
- VaR is a quantile, not a worst case. The number says nothing about the shape of the loss beyond it, which is why CVaR is reported alongside.
- The commodity leg uses a price series without modelling roll yield in full.

## Where this sits

Part of a nine-repo research pipeline. Stock Stalker screens the NSE universe; this repo publishes a versioned artifact it reads back:

```json
{
  "risk": {
    "var_95": -0.032,
    "cvar_95": -0.048,
    "horizon_days": 20
  }
}
```

Communication is by file contract, not imports, so either side can be refactored without breaking the other.

## Exam mapping

Series VIII (option pricing, payoffs), Series XVI (commodity fundamentals, hedging), Series XV ch.12.4 (measuring risk)

---

CLI only, by design. No dashboard, no server. Charts and documents are written to `outputs/`.

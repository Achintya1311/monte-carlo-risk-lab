# Monte Carlo risk lab

Prices options and models portfolio risk by simulation, checked against closed-form solutions so the numbers are verifiable rather than merely plausible.

**Status:** Last checkpoint 2026-09-16 · Next: Day 5 - Portfolio risk: VaR and CVaR at 95/99, historical vs parametric vs Monte Carlo side by side, drawdown distribution

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
- `fixtures/ohlcv/RELIANCE_NS.csv` - 2 years of NSE daily closes, the same committed fixture Stock Stalker screens against (copied in for Day 5's historical/parametric/Monte Carlo VaR comparison, not fetched live)

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

**Day 4 - path-dependent options: Asian and barrier, where no direct closed form exists.** These are the first payoffs in the repo that need the full simulated path (`mcsim.gbm.simulate_gbm_paths`), not just the terminal price. Neither has a closed form checked directly, so each is validated by a different route.

`mcsim.asian` prices average-price (Asian) options. Arithmetic-average Asian options have no closed form under GBM - a sum of lognormals is not lognormal - so the correctness check is two steps removed: the *geometric*-average Asian option does have a closed form (discrete-monitoring Kemna-Vorst, an adjusted-vol/adjusted-rate Black-Scholes formula, `geometric_asian_price`, which collapses exactly to plain Black-Scholes at n_fixings=1 - checked as its own test), so the geometric MC estimator is validated against that first, and the arithmetic MC price is then checked against the geometric one via Jensen's inequality (arithmetic mean of positive numbers >= geometric mean, and a call payoff is nondecreasing in the average, so arithmetic call price >= geometric call price; the inequality flips for puts). At spot=100/strike=100/vol=0.25/rate=0.07/30d/seed=42, N=1e6: geometric MC 2.209375 vs closed form 2.211382 (-0.63 std errors) for the call, 1.845712 vs 1.845146 (+0.21 std errors) for the put - both well inside tolerance. Arithmetic MC then prices at 2.243958 for the call (+0.034583 above geometric, consistent with Jensen) and 1.818380 for the put (-0.027331 below geometric, also consistent) - the ordering that must hold, not a number checked against a reference.

`mcsim.barrier` prices knock-in/knock-out pairs. There is a closed form for continuously-monitored GBM barriers (Reiner-Rubinstein), but the simulator only observes daily closes, so a discretely-monitored barrier does not match that formula and no attempt is made to compare against it here. Instead, `mc_barrier_pair_price` uses an exact identity: every path either breaches the barrier or it doesn't, so knock-in payoff + knock-out payoff = vanilla payoff on every single path, and summing the two MC prices computed from the *same* simulated paths must reproduce the vanilla MC price to floating-point precision, not just within sampling error (`identity_gap` in the output, printed as `+0.00e+00` in every run below). That vanilla MC price (now built from the full path simulator rather than Day 2/3's terminal-price shortcut) is separately checked against Black-Scholes within the usual std-error tolerance - a second, independent gate on the same run. At spot=100/strike=100/vol=0.25/rate=0.07/30d/seed=42, N=1e6: up-and-{in,out} call at barrier=120 gives knock-in 0.641106 + knock-out 3.211945 = 3.853051, matching vanilla MC 3.853051 exactly, itself -0.67 std errors from Black-Scholes 3.856763 (breach fraction 3.0%); down-and-{in,out} put at barrier=85 gives knock-in 0.659575 + knock-out 2.364302 = 3.023877, matching vanilla MC 3.023877 exactly, -0.67 std errors from Black-Scholes 3.026893 (breach fraction 4.3%).

**Day 5 - portfolio risk: VaR/CVaR by three methods, a Monte Carlo drawdown distribution, and a Kupiec backtest.** First use in the repo of a real historical fixture (`fixtures/ohlcv/RELIANCE_NS.csv`, the same 2-year NSE daily-close series Stock Stalker screens against) rather than assumed parameters, and the first place GBM is driven by a *physical* drift (`mcsim.returns` fits the sample mean/std of daily log returns) instead of the risk-neutral rate Days 1-4 use for pricing - there is no hedge to price here, only a real-world loss probability to estimate. `mcsim.risk` implements three VaR/CVaR estimators that all read the same fitted sample: **historical** simulation takes the empirical quantile of overlapping horizon-day windows built directly from the fixture; **parametric** scales the fitted mean/std to the horizon under an iid-Gaussian assumption and uses the closed-form normal VaR/expected-shortfall formulas (checked in tests against the textbook standard-normal constants, VaR₉₅=-1.6449, ES₉₅=-2.0627); **Monte Carlo** simulates GBM paths under that same fitted drift/vol and takes the empirical quantile of simulated horizon returns - same distributional assumption as parametric, different route, and the only one of the three that also yields path-dependent output (the drawdown distribution below). `mcsim.portfolio` is the CLI that runs all three at 95% and 99%, plus a Kupiec (1995) proportion-of-failures backtest of the 1-day parametric VaR against the full historical sample.

At horizon=20 trading days, N=1e6 MC paths, seed=42, fitted from 500 daily returns (mu=-0.000255/day, sigma=0.013069/day, annualized mu=-6.42%, sigma=20.75%):

| Confidence | Method | VaR | CVaR |
|---|---|---|---|
| 95% | historical | -0.0915 | -0.1082 |
| 95% | parametric | -0.1012 | -0.1257 |
| 95% | Monte Carlo | -0.1028 | -0.1273 |
| 99% | historical | -0.1207 | -0.1270 |
| 99% | parametric | -0.1411 | -0.1609 |
| 99% | Monte Carlo | -0.1427 | -0.1626 |

Parametric and Monte Carlo track each other closely at both confidence levels (same Gaussian assumption, different route to it, as expected) and both sit more extreme than historical - this single name's realized 20-day return distribution over the fixture window happens to be less fat-tailed on the downside than a Gaussian fit to its own daily moments predicts, the opposite of the usual "real markets are fatter-tailed" direction, and a reminder that one ~2-year window is a small sample to draw that conclusion from either way.

The Kupiec backtest is the sharper, more honest test, and it fails at 99%: the 1-day parametric VaR (-2.18% at 95%, -3.07% at 99%) is breached 20/500 days at the 95% threshold (25.0 expected, LR=1.127, p=0.29, **not rejected**) but 10/500 days at the 99% threshold (5.0 expected - twice the expected exception count, LR=3.914, p=0.048, **REJECTED** at the 5% test significance). `mcsim.portfolio` reports this straight - exit code 1 and a stderr note, not a smoothed-over pass - because it is exactly the textbook failure mode the README has flagged since Day 1: GBM's Gaussian tails are thin, and thin tails are wrong furthest out in the tail, where 99% VaR lives and 95% VaR mostly doesn't reach. This looks like it contradicts the table above, where historical sits *inside* (less extreme than) the parametric/MC 20-day quantiles - it doesn't, once the horizons are separated: the 1-day daily tail is genuinely fatter than Gaussian (that's what Kupiec caught), but the 20-day overlapping-window historical quantile is built from only ~25 non-overlapping 20-day blocks worth of independent information out of 500 days, and aggregating fat-tailed daily moves over 20 days pulls the sum toward Gaussian by the central limit theorem faster than a small, overlap-correlated sample can show it. Two true things at two different horizons, not one inconsistent finding.

The Monte Carlo max-drawdown distribution (`--drawdown-plot`, same 1e6 paths and 20-day horizon, `outputs/drawdown_dist.png`, gitignored) has mean -5.96%, median -5.42%, and a fat right-skewed left tail out to a worst-of-1M-paths -25.02% (p5=-11.53%, p95=-2.21%). The single realized drawdown actually observed in the fixture window itself is -23.87% - inside the simulated distribution's range but beyond its 5th percentile, consistent with the same thin-tails gap the Kupiec test caught rather than an independent finding.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-16 | `f582519` | Day 4: path-dependent options, Asian and barrier (mcsim/asian.py, mcsim/barrier.py), wired into a new mcsim.exotic CLI with 'asian' and 'barrier' subcommands. Neither payoff has a closed form checked directly, so each earns correctness a different way: geometric-average Asian is priced by the discrete-monitoring Kemna-Vorst closed form (which collapses exactly to Black-Scholes at n_fixings=1) and the arithmetic-average MC price is then checked against it only via Jensen's inequality (arithmetic call >= geometric call, and vice versa for puts) since no direct tolerance check exists for arithmetic; barrier options use an exact per-path identity instead - knock-in + knock-out must reproduce the vanilla MC price computed from the same paths to floating-point precision, with that vanilla MC price separately checked against Black-Scholes within the usual std-error tolerance. At spot=100/strike=100/vol=0.25/rate=0.07/30d/seed=42, N=1e6: geometric MC Asian call -0.63 std errors from closed form (put +0.21), arithmetic priced 2.243958 (call, above geometric as Jensen requires) and 1.818380 (put, below); barrier up-call at barrier=120 gave knock-in 0.641106 + knock-out 3.211945 = vanilla MC 3.853051 exactly, itself -0.67 std errors from Black-Scholes 3.856763. README records honestly that the Jensen check only verifies ordering, not arithmetic's absolute accuracy, and that discrete barrier monitoring is not compared against the continuous-monitoring Reiner-Rubinstein closed form since the two are known to differ systematically. 24 new tests pass (92/92 total); both asian and barrier CLI subcommands run by hand at N=1e6 for calls and puts. Also found and fixed this repo's and the hub's local main branches stuck in stale detached HEAD (same recurring class of issue prior checkpoints hit) before committing. | Day 5 - Portfolio risk: VaR and CVaR at 95/99, historical vs parametric vs Monte Carlo side by side, drawdown distribution |
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
- The arithmetic-average Asian option has no closed form checked against it directly in this repo. `mcsim.exotic asian` validates the geometric-average estimator against its own closed form, then only checks that the arithmetic price sits on the correct side of the geometric price (Jensen's inequality) - that confirms the ordering is right, not that the arithmetic price itself is numerically correct to some tolerance. A bug that shifted the arithmetic price by a plausible amount while preserving the ordering would not be caught by this check.
- `mcsim.barrier` monitors the barrier only at the simulator's daily closes (discrete monitoring), not continuously. Continuously-monitored GBM barriers have a closed form (Reiner-Rubinstein); this repo's discretely-monitored price is systematically different from that formula (discrete monitoring is known to underestimate the true breach probability, biasing knock-out prices up and knock-in prices down relative to continuous monitoring) and no attempt is made to compare against it - the correctness check used instead (knock-in + knock-out = vanilla, exactly) is agnostic to the monitoring frequency by construction, which is what makes it usable here.
- Both `mcsim.asian` and `mcsim.barrier` price off the full path simulator (`mcsim.gbm.simulate_gbm_paths`), unlike Day 3's terminal-price shortcut - a third RNG consumption pattern in the repo alongside Day 2/3's, so none of the three are driven by common random numbers against each other even at the same seed and path count.
- VaR is a quantile, not a worst case. The number says nothing about the shape of the loss beyond it, which is why CVaR is reported alongside.
- The historical VaR/CVaR estimator uses overlapping horizon-day windows (481 of them at a 20-day horizon from 500 daily returns) because the fixture is too short to give more than ~25 genuinely independent 20-day observations. Overlapping windows share most of their days with their neighbours, so the empirical quantile is a real number computed from real data, but its own sampling uncertainty is understated - this is not a large-sample historical simulation.
- The Kupiec backtest is in-sample: the same 500 historical days both fit the parametric model's mu/sigma and get backtested against it. A model can look better in-sample than it will out-of-sample by construction; this repo does not hold out a test window or walk the fit forward.
- The Kupiec backtest only checks 1-day VaR (the horizon at which 500 days gives enough exceptions for the test to have power). It does not directly backtest the 20-day horizon VaR the table above reports - the two are linked by the same fitted mu/sigma, but a model that fails the 1-day test is not thereby proven to fail (or pass) at 20 days, only strongly suspected to.
- One equity, one ~2-year fixture window, one realized drawdown. The "realized max drawdown" figure compared against the simulated distribution is a sample of size one - agreement or disagreement with the simulation is suggestive, not a validated backtest of the drawdown distribution itself, which is a Day 5 gap: only the VaR/CVaR side gets the Kupiec treatment, the drawdown side does not get an equivalent historical check.
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

# Monte Carlo risk lab

Prices options and models portfolio risk by simulation, checked against closed-form solutions so the numbers are verifiable rather than merely plausible.

**Status:** Not started · Next: Day 1 - vectorized seeded GBM simulator

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

Nothing yet. This section fills in as the work lands, including the results that do not flatter the method.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Geometric Brownian motion has thin tails and constant volatility. Real returns have neither.
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

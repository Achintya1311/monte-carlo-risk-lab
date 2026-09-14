"""CLI: run the GBM simulator and report terminal-price statistics against
the known closed-form lognormal moments.

    python -m mcsim.simulate --spot 100 --vol 0.25 --rate 0.07 --days 30 \\
        --paths 100000 --seed 42

This is a sanity check, not the pricer - Day 2 adds option pricing with a
Black-Scholes cross-check. Here we only confirm the simulator itself
converges to what a lognormal terminal distribution requires:
E[S_T] = spot * exp(rate * T).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from mcsim.gbm import TRADING_DAYS_PER_YEAR, simulate_gbm_paths, terminal_prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate GBM price paths and report terminal statistics."
    )
    parser.add_argument("--spot", type=float, required=True, help="initial price")
    parser.add_argument("--vol", type=float, required=True, help="annualized volatility, e.g. 0.25")
    parser.add_argument("--rate", type=float, required=True, help="annualized drift / risk-free rate, e.g. 0.07")
    parser.add_argument("--days", type=int, required=True, help="horizon in trading days")
    parser.add_argument("--paths", type=int, default=100_000, help="number of simulated paths (default: 100000)")
    parser.add_argument("--steps-per-day", type=int, default=1, help="sub-day time steps (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducibility (default: 42)")
    parser.add_argument("--out", type=str, default=None, help="optional CSV path to write terminal prices to")
    return parser


def run(args: argparse.Namespace) -> dict:
    paths = simulate_gbm_paths(
        spot=args.spot,
        rate=args.rate,
        vol=args.vol,
        days=args.days,
        n_paths=args.paths,
        steps_per_day=args.steps_per_day,
        seed=args.seed,
    )
    terminals = terminal_prices(paths)

    T = args.days / TRADING_DAYS_PER_YEAR
    theoretical_mean = args.spot * np.exp(args.rate * T)
    sample_mean = float(terminals.mean())
    rel_error = (sample_mean - theoretical_mean) / theoretical_mean

    theoretical_log_std = args.vol * np.sqrt(T)
    sample_log_std = float(np.log(terminals / args.spot).std(ddof=1))

    return {
        "n_paths": args.paths,
        "days": args.days,
        "seed": args.seed,
        "sample_mean": sample_mean,
        "theoretical_mean": theoretical_mean,
        "relative_error": rel_error,
        "sample_log_std": sample_log_std,
        "theoretical_log_std": theoretical_log_std,
        "terminals": terminals,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)

    print(f"paths={result['n_paths']} days={result['days']} seed={result['seed']}")
    print(f"sample mean terminal price   : {result['sample_mean']:.6f}")
    print(f"theoretical mean (S0*e^(rT)) : {result['theoretical_mean']:.6f}")
    print(f"relative error               : {result['relative_error']:+.4%}")
    print(f"sample log-return std        : {result['sample_log_std']:.6f}")
    print(f"theoretical log-return std   : {result['theoretical_log_std']:.6f}")

    if result["n_paths"] >= 100_000 and abs(result["relative_error"]) > 0.01:
        print(
            "WARNING: relative error exceeds 1% at this path count - "
            "check the seed or inputs before trusting the run.",
            file=sys.stderr,
        )

    if args.out:
        import csv

        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["terminal_price"])
            for value in result["terminals"]:
                writer.writerow([value])
        print(f"wrote {result['n_paths']} terminal prices to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

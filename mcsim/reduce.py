"""CLI: variance reduction table - plain, antithetic, and control-variate
Monte Carlo, at the same path budget, checked against Black-Scholes.

    python -m mcsim.reduce --spot 100 --strike 105 --vol 0.25 --rate 0.07 \\
        --days 30 --paths 100000 --seed 42 --option-type call

This is Day 3's correctness gate: every method's price must still sit
within a Monte Carlo error band of Black-Scholes (variance reduction must
not introduce bias), and the table reports each method's standard-error
reduction factor relative to the plain estimator at the same N.
"""
from __future__ import annotations

import argparse
import sys

from mcsim.blackscholes import VALID_OPTION_TYPES, bs_price
from mcsim.variance_reduction import variance_reduction_table

# Same tolerance mcsim.price uses: a run more than this many standard errors
# from Black-Scholes is flagged rather than silently trusted.
SIGMA_TOLERANCE = 4.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare plain, antithetic, and control-variate Monte Carlo standard errors at the same path budget."
    )
    parser.add_argument("--spot", type=float, required=True, help="initial price")
    parser.add_argument("--strike", type=float, required=True, help="strike price")
    parser.add_argument("--vol", type=float, required=True, help="annualized volatility, e.g. 0.25")
    parser.add_argument("--rate", type=float, required=True, help="annualized drift / risk-free rate, e.g. 0.07")
    parser.add_argument("--days", type=int, required=True, help="time to expiry in trading days")
    parser.add_argument("--paths", type=int, default=100_000, help="path budget per method (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducibility (default: 42)")
    parser.add_argument(
        "--option-type", type=str, default="call", choices=VALID_OPTION_TYPES, help="call or put (default: call)"
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    theoretical = bs_price(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        option_type=args.option_type,
    )
    rows = variance_reduction_table(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        n_paths=args.paths, option_type=args.option_type, seed=args.seed,
    )
    for row in rows:
        diff = row["price"] - theoretical
        row["diff"] = diff
        row["sigma"] = diff / row["std_error"] if row["std_error"] > 0 else float("nan")
        row["within_tolerance"] = abs(row["sigma"]) <= SIGMA_TOLERANCE

    return {"bs_price": theoretical, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)

    print(f"option={args.option_type} paths={args.paths} seed={args.seed}")
    print(f"Black-Scholes price: {result['bs_price']:.6f}")
    print()
    header = f"{'method':<16}{'price':>12}{'std_error':>14}{'sigma':>10}{'reduction':>12}"
    print(header)
    print("-" * len(header))
    any_out_of_tolerance = False
    for row in result["rows"]:
        print(
            f"{row['method']:<16}{row['price']:>12.6f}{row['std_error']:>14.6f}"
            f"{row['sigma']:>+10.2f}{row['reduction_factor']:>11.2f}x"
        )
        if not row["within_tolerance"]:
            any_out_of_tolerance = True

    if any_out_of_tolerance:
        print(
            f"WARNING: at least one method is more than {SIGMA_TOLERANCE:.0f} standard errors from "
            "Black-Scholes - variance reduction should not introduce bias, check before trusting this run.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

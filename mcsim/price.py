"""CLI: price a European option by Monte Carlo, cross-checked against the
Black-Scholes closed form, with an optional convergence plot as N grows.

    python -m mcsim.price --spot 100 --strike 105 --vol 0.25 --rate 0.07 \\
        --days 30 --paths 1000000 --seed 42 --option-type call \\
        --convergence-plot outputs/convergence_call.png

This is the repo's correctness gate: the MC price at N=paths must sit
within a Monte Carlo error band of Black-Scholes, and the convergence plot
must show that band shrinking as N grows.
"""
from __future__ import annotations

import argparse
import sys

from mcsim.blackscholes import VALID_OPTION_TYPES, bs_price
from mcsim.convergence import DEFAULT_PATH_GRID, convergence_sweep, plot_convergence
from mcsim.pricing import mc_option_price

# How many standard errors the MC price may sit from Black-Scholes before
# the run is flagged. At 4 sigma, a false alarm on a correct implementation
# is expected roughly 1 run in 15,000 - tight enough to catch a real bug,
# loose enough not to cry wolf on ordinary sampling noise.
SIGMA_TOLERANCE = 4.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Price a European option by Monte Carlo, checked against Black-Scholes."
    )
    parser.add_argument("--spot", type=float, required=True, help="initial price")
    parser.add_argument("--strike", type=float, required=True, help="strike price")
    parser.add_argument("--vol", type=float, required=True, help="annualized volatility, e.g. 0.25")
    parser.add_argument("--rate", type=float, required=True, help="annualized drift / risk-free rate, e.g. 0.07")
    parser.add_argument("--days", type=int, required=True, help="time to expiry in trading days")
    parser.add_argument("--paths", type=int, default=1_000_000, help="number of MC paths (default: 1000000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducibility (default: 42)")
    parser.add_argument(
        "--option-type", type=str, default="call", choices=VALID_OPTION_TYPES, help="call or put (default: call)"
    )
    parser.add_argument(
        "--convergence-plot", type=str, default=None,
        help="optional path to write a PNG showing MC price vs Black-Scholes across a grid of N",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    theoretical = bs_price(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        option_type=args.option_type,
    )
    mc = mc_option_price(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        n_paths=args.paths, option_type=args.option_type, seed=args.seed,
    )
    diff = mc["price"] - theoretical
    sigma = diff / mc["std_error"] if mc["std_error"] > 0 else float("nan")

    return {
        "option_type": args.option_type,
        "n_paths": mc["n_paths"],
        "seed": args.seed,
        "bs_price": theoretical,
        "mc_price": mc["price"],
        "std_error": mc["std_error"],
        "diff": diff,
        "sigma": sigma,
        "within_tolerance": abs(sigma) <= SIGMA_TOLERANCE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)

    print(f"option={result['option_type']} paths={result['n_paths']} seed={result['seed']}")
    print(f"Black-Scholes price   : {result['bs_price']:.6f}")
    print(f"Monte Carlo price     : {result['mc_price']:.6f}  (std error {result['std_error']:.6f})")
    print(f"difference            : {result['diff']:+.6f}  ({result['sigma']:+.2f} std errors)")

    if not result["within_tolerance"]:
        print(
            f"WARNING: MC price is more than {SIGMA_TOLERANCE:.0f} standard errors from "
            "Black-Scholes - check the seed, path count, or the pricer before trusting this run.",
            file=sys.stderr,
        )

    if args.convergence_plot:
        rows = convergence_sweep(
            spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
            option_type=args.option_type, seed=args.seed, path_grid=DEFAULT_PATH_GRID,
        )
        plot_convergence(rows, result["bs_price"], args.convergence_plot, args.option_type)
        print(f"wrote convergence plot ({len(rows)} points, N={DEFAULT_PATH_GRID}) to {args.convergence_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

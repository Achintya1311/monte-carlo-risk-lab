"""CLI: path-dependent options where no closed form is exercised directly -
Asian (average-price) and barrier options - each checked against something
verifiable instead.

    python -m mcsim.exotic asian --spot 100 --strike 100 --vol 0.25 \\
        --rate 0.07 --days 30 --paths 500000 --seed 42 --option-type call

    python -m mcsim.exotic barrier --spot 100 --strike 100 --barrier 120 \\
        --vol 0.25 --rate 0.07 --days 30 --paths 500000 --seed 42 \\
        --option-type call --direction up

Asian: the geometric-average MC price is checked against the Kemna-Vorst
closed form (std-error tolerance), then the arithmetic MC price is checked
against the geometric price via Jensen's inequality (arithmetic mean >=
geometric mean, and a call payoff is nondecreasing in the average).

Barrier: knock-in + knock-out must reproduce the vanilla Monte Carlo price
from the same paths to floating-point precision (an exact identity, not a
tolerance), and that vanilla MC price is separately checked against
Black-Scholes within the Day 2 std-error tolerance.
"""
from __future__ import annotations

import argparse
import sys

from mcsim.asian import geometric_asian_price, mc_asian_option_price
from mcsim.barrier import VALID_DIRECTIONS, mc_barrier_pair_price
from mcsim.blackscholes import VALID_OPTION_TYPES, bs_price

SIGMA_TOLERANCE = 4.0


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spot", type=float, required=True, help="initial price")
    parser.add_argument("--strike", type=float, required=True, help="strike price")
    parser.add_argument("--vol", type=float, required=True, help="annualized volatility, e.g. 0.25")
    parser.add_argument("--rate", type=float, required=True, help="annualized drift / risk-free rate, e.g. 0.07")
    parser.add_argument("--days", type=int, required=True, help="time to expiry in trading days")
    parser.add_argument("--paths", type=int, default=500_000, help="number of MC paths (default: 500000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducibility (default: 42)")
    parser.add_argument(
        "--option-type", type=str, default="call", choices=VALID_OPTION_TYPES, help="call or put (default: call)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Price path-dependent options (Asian, barrier) with no direct closed-form check."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    asian = subparsers.add_parser("asian", help="average-price Asian option")
    _common_args(asian)

    barrier = subparsers.add_parser("barrier", help="knock-in/knock-out barrier option pair")
    _common_args(barrier)
    barrier.add_argument("--barrier", type=float, required=True, help="barrier level")
    barrier.add_argument(
        "--direction", type=str, default="up", choices=VALID_DIRECTIONS,
        help="'up' (barrier above spot) or 'down' (barrier below spot) (default: up)",
    )

    return parser


def run_asian(args: argparse.Namespace) -> dict:
    arithmetic = mc_asian_option_price(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        n_paths=args.paths, option_type=args.option_type, average_type="arithmetic", seed=args.seed,
    )
    geometric_mc = mc_asian_option_price(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        n_paths=args.paths, option_type=args.option_type, average_type="geometric", seed=args.seed,
    )
    geometric_closed_form = geometric_asian_price(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        n_fixings=args.days, option_type=args.option_type,
    )

    diff = geometric_mc["price"] - geometric_closed_form
    sigma = diff / geometric_mc["std_error"] if geometric_mc["std_error"] > 0 else float("nan")

    jensen_gap = arithmetic["price"] - geometric_mc["price"]
    combined_se = (arithmetic["std_error"] ** 2 + geometric_mc["std_error"] ** 2) ** 0.5
    if args.option_type == "call":
        jensen_holds = jensen_gap >= -SIGMA_TOLERANCE * combined_se
    else:
        jensen_holds = jensen_gap <= SIGMA_TOLERANCE * combined_se

    return {
        "option_type": args.option_type,
        "n_paths": args.paths,
        "seed": args.seed,
        "arithmetic": arithmetic,
        "geometric_mc": geometric_mc,
        "geometric_closed_form": geometric_closed_form,
        "geometric_sigma": sigma,
        "geometric_within_tolerance": abs(sigma) <= SIGMA_TOLERANCE,
        "jensen_gap": jensen_gap,
        "jensen_holds": jensen_holds,
    }


def run_barrier(args: argparse.Namespace) -> dict:
    result = mc_barrier_pair_price(
        spot=args.spot, strike=args.strike, barrier=args.barrier, rate=args.rate, vol=args.vol,
        days=args.days, n_paths=args.paths, option_type=args.option_type, direction=args.direction,
        seed=args.seed,
    )
    theoretical = bs_price(
        spot=args.spot, strike=args.strike, rate=args.rate, vol=args.vol, days=args.days,
        option_type=args.option_type,
    )
    vanilla_mc = result["vanilla_mc"]
    diff = vanilla_mc["price"] - theoretical
    sigma = diff / vanilla_mc["std_error"] if vanilla_mc["std_error"] > 0 else float("nan")

    result["bs_price"] = theoretical
    result["vanilla_sigma"] = sigma
    result["vanilla_within_tolerance"] = abs(sigma) <= SIGMA_TOLERANCE
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "asian":
        result = run_asian(args)
        print(f"option={result['option_type']} paths={result['n_paths']} seed={result['seed']}")
        print(f"Arithmetic Asian (MC)   : {result['arithmetic']['price']:.6f}  (std error {result['arithmetic']['std_error']:.6f})")
        print(f"Geometric Asian (MC)    : {result['geometric_mc']['price']:.6f}  (std error {result['geometric_mc']['std_error']:.6f})")
        print(f"Geometric Asian (closed): {result['geometric_closed_form']:.6f}")
        print(f"geometric MC vs closed  : {result['geometric_sigma']:+.2f} std errors")
        print(f"arithmetic - geometric  : {result['jensen_gap']:+.6f}  (Jensen: arithmetic call >= geometric call)")

        ok = True
        if not result["geometric_within_tolerance"]:
            ok = False
            print(
                f"WARNING: geometric MC price is more than {SIGMA_TOLERANCE:.0f} standard errors from "
                "the closed form - check the pricer before trusting this run.",
                file=sys.stderr,
            )
        if not result["jensen_holds"]:
            ok = False
            print(
                "WARNING: arithmetic-vs-geometric average ordering (Jensen's inequality) is violated "
                "beyond sampling error - check the pricer before trusting this run.",
                file=sys.stderr,
            )
        return 0 if ok else 1

    result = run_barrier(args)
    print(f"option={args.option_type} direction={args.direction} barrier={args.barrier} paths={args.paths} seed={args.seed}")
    print(f"Knock-in  (MC)  : {result['knock_in']['price']:.6f}  (std error {result['knock_in']['std_error']:.6f})")
    print(f"Knock-out (MC)  : {result['knock_out']['price']:.6f}  (std error {result['knock_out']['std_error']:.6f})")
    print(f"in + out        : {result['knock_in']['price'] + result['knock_out']['price']:.6f}")
    print(f"Vanilla (MC)    : {result['vanilla_mc']['price']:.6f}  (std error {result['vanilla_mc']['std_error']:.6f})")
    print(f"identity gap    : {result['identity_gap']:+.2e}  (in + out - vanilla_mc, should be ~0)")
    print(f"Vanilla (BS)    : {result['bs_price']:.6f}")
    print(f"vanilla MC vs BS: {result['vanilla_sigma']:+.2f} std errors")
    print(f"breach fraction : {result['breach_fraction']:.4f}")

    ok = True
    if abs(result["identity_gap"]) > 1e-9:
        ok = False
        print(
            "WARNING: knock-in + knock-out does not reproduce the vanilla MC price to floating-point "
            "precision - this should be an exact identity, check the payoff partition.",
            file=sys.stderr,
        )
    if not result["vanilla_within_tolerance"]:
        ok = False
        print(
            f"WARNING: vanilla MC price (from barrier-run paths) is more than {SIGMA_TOLERANCE:.0f} standard "
            "errors from Black-Scholes - check the seed, path count, or the pricer before trusting this run.",
            file=sys.stderr,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

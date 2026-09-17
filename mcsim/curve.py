"""CLI: WTI futures term structure - contango/backwardation classification,
an exact curve identity check, and historical VaR/CVaR conditioned on the
curve regime observed at the start of each horizon window.

    python -m mcsim.curve --curve fixtures/commodity/WTI_futures_curve.csv \\
        --prices fixtures/commodity/WTI_C1.csv --horizon-days 20 --confidence 0.95

Reuses mcsim.risk's overlapping-window historical-simulation method (see
mcsim.commodity.regime_conditioned_var_cvar) on the front-month contract,
same as mcsim.portfolio does for the equity fixture in Day 5 - the new
part is conditioning it on curve shape instead of pooling every day.
"""
from __future__ import annotations

import argparse

from mcsim.commodity import (
    annualized_basis,
    classify_regime,
    curve_identity_gap,
    load_curve,
    regime_conditioned_var_cvar,
    regime_summary,
)
from mcsim.returns import load_daily_log_returns

MONTHS_C1_TO_C4 = 3.0
REGIME_ORDER = ("unconditional", "contango", "backwardation", "flat")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WTI futures curve state (contango/backwardation) and regime-conditioned historical VaR/CVaR."
    )
    parser.add_argument(
        "--curve", type=str, default="fixtures/commodity/WTI_futures_curve.csv",
        help="CSV with date,c1,c2,c3,c4 contract-month closes (default: fixtures/commodity/WTI_futures_curve.csv)",
    )
    parser.add_argument(
        "--prices", type=str, default="fixtures/commodity/WTI_C1.csv",
        help="CSV with date,close for the front-month contract (default: fixtures/commodity/WTI_C1.csv)",
    )
    parser.add_argument("--horizon-days", type=int, default=20, help="risk horizon in trading days (default: 20)")
    parser.add_argument("--confidence", type=float, default=0.95, help="VaR/CVaR confidence level (default: 0.95)")
    return parser


def run(args: argparse.Namespace) -> dict:
    if not 0 < args.confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {args.confidence}")

    curve = load_curve(args.curve)
    regime = classify_regime(curve["c1"], curve["c4"])
    basis = annualized_basis(curve["c1"], curve["c4"], MONTHS_C1_TO_C4)
    identity_gap = curve_identity_gap(curve["c1"], curve["c2"], curve["c3"], curve["c4"])
    summary = regime_summary(regime)

    log_returns = load_daily_log_returns(args.prices)
    # log_returns[i] is the return from day i to day i+1, so day i's regime
    # (regime[:-1] drops the return-less final row) is what "starts" it.
    window_regime = regime[:-1]
    var_cvar = regime_conditioned_var_cvar(log_returns, window_regime, args.confidence, args.horizon_days)

    return {
        "curve_path": args.curve,
        "prices_path": args.prices,
        "n_days": summary["n_days"],
        "counts": summary["counts"],
        "pct": summary["pct"],
        "current_regime": summary["current"],
        "current_basis_annualized": float(basis[-1]),
        "identity_gap": identity_gap,
        "confidence": args.confidence,
        "horizon_days": args.horizon_days,
        "var_cvar": var_cvar,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)

    print(f"curve={result['curve_path']} prices={result['prices_path']} n_days={result['n_days']}")
    print(
        f"current regime: {result['current_regime']} "
        f"(annualized c1->c4 basis {result['current_basis_annualized']:+.4f})"
    )
    print("regime mix over the fixture window:")
    for label in ("contango", "backwardation", "flat"):
        print(f"  {label:>13}: {result['counts'][label]:4d} days ({result['pct'][label]:6.1%})")
    print(f"curve identity gap (segment carries vs direct c1->c4 log-carry): {result['identity_gap']:.2e}")

    print()
    print(
        f"historical VaR/CVaR at {result['confidence']:.0%}, {result['horizon_days']}-day horizon, "
        "by curve regime at window start"
    )
    for label in REGIME_ORDER:
        row = result["var_cvar"][label]
        if row["var"] is None:
            print(f"  {label:>13}: n_windows={row['n_windows']:4d}  (below min sample, not reported)")
        else:
            print(f"  {label:>13}: n_windows={row['n_windows']:4d}  VaR={row['var']:+.4f}  CVaR={row['cvar']:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

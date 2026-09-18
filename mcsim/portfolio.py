"""CLI: portfolio risk - VaR/CVaR at 95%/99% by three methods side by side,
a Monte Carlo max-drawdown distribution, and a Kupiec backtest of the
1-day parametric VaR against the historical sample.

    python -m mcsim.portfolio --fixture fixtures/ohlcv/RELIANCE_NS.csv \\
        --horizon-days 20 --confidence 0.95 0.99 --paths 1000000 --seed 42 \\
        --drawdown-plot outputs/drawdown_dist.png

Historical, parametric and Monte Carlo are computed from the *same*
fitted sample (daily log returns from the fixture), so a gap between
methods here is a genuine modeling difference (fat tails, non-normality
the closed form cannot see), not different input data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mcsim.returns import annualize, load_closes, load_daily_log_returns
from mcsim.risk import (
    drawdown_distribution_stats,
    historical_var_cvar,
    kupiec_pof_test,
    max_drawdowns,
    mc_var_cvar_from_paths,
    parametric_var_cvar,
    plot_drawdown_distribution,
    simulate_risk_paths,
)

DEFAULT_SPOT = 100.0  # VaR/CVaR are returns, so the normalization level is arbitrary.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portfolio VaR/CVaR (historical, parametric, Monte Carlo) and a drawdown distribution."
    )
    parser.add_argument(
        "--fixture", type=str, default="fixtures/ohlcv/RELIANCE_NS.csv",
        help="OHLCV CSV with a 'close' column (default: fixtures/ohlcv/RELIANCE_NS.csv)",
    )
    parser.add_argument(
        "--confidence", type=float, nargs="+", default=[0.95, 0.99],
        help="confidence levels, e.g. 0.95 0.99 (default: 0.95 0.99)",
    )
    parser.add_argument("--horizon-days", type=int, default=20, help="risk horizon in trading days (default: 20)")
    parser.add_argument("--paths", type=int, default=1_000_000, help="number of MC paths (default: 1000000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducibility (default: 42)")
    parser.add_argument(
        "--drawdown-plot", type=str, default=None,
        help="optional path to write a PNG histogram of the simulated max-drawdown distribution",
    )
    parser.add_argument(
        "--contract",
        metavar="PATH",
        default=None,
        help="write the v0.3 risk contract block (see to_contract()) as JSON to this path, "
        "for the spine to read as a file -- never as a Python import",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    for c in args.confidence:
        if not 0 < c < 1:
            raise ValueError(f"confidence must be in (0, 1), got {c}")

    log_returns = load_daily_log_returns(args.fixture)
    closes = load_closes(args.fixture)
    mu_daily = float(log_returns.mean())
    sigma_daily = float(log_returns.std(ddof=1))
    mu_annual, sigma_annual = annualize(mu_daily, sigma_daily)

    paths = simulate_risk_paths(
        spot=DEFAULT_SPOT, mu_annual=mu_annual, sigma_annual=sigma_annual,
        horizon_days=args.horizon_days, n_paths=args.paths, seed=args.seed,
    )

    by_confidence = []
    for c in args.confidence:
        hist = historical_var_cvar(log_returns, c, args.horizon_days)
        param = parametric_var_cvar(mu_daily, sigma_daily, c, args.horizon_days)
        mc = mc_var_cvar_from_paths(paths, DEFAULT_SPOT, c)

        var_1day = parametric_var_cvar(mu_daily, sigma_daily, c, 1)["var"]
        kupiec = kupiec_pof_test(log_returns, var_1day, c)

        by_confidence.append(
            {"confidence": c, "historical": hist, "parametric": param, "mc": mc, "kupiec": kupiec, "var_1day": var_1day}
        )

    drawdowns = max_drawdowns(paths)
    dd_stats = drawdown_distribution_stats(drawdowns)
    realized_max_dd = float(max_drawdowns(closes.reshape(1, -1))[0])

    if args.drawdown_plot:
        plot_drawdown_distribution(drawdowns, args.drawdown_plot)

    return {
        "fixture": args.fixture,
        "n_obs": len(log_returns),
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
        "horizon_days": args.horizon_days,
        "n_paths": args.paths,
        "seed": args.seed,
        "by_confidence": by_confidence,
        "drawdown_stats": dd_stats,
        "realized_max_drawdown": realized_max_dd,
        "drawdown_plot": args.drawdown_plot,
    }


def to_contract(result: dict) -> dict:
    """The ``risk`` block this repo publishes to the spine (v0.3), matching
    the shape NEXT_STEPS.md committed to before this module existed:
    ``{"risk": {"var_95", "cvar_95", "max_dd_sim", "horizon_days"}}``.

    ``var_95``/``cvar_95`` come from the historical method, not parametric or
    MC -- it is the one of the three that makes no distributional assumption,
    so it is the number this repo is most willing to stand behind as a file
    contract read by a repo that never sees the other two for comparison.
    """
    row_95 = next((r for r in result["by_confidence"] if r["confidence"] == 0.95), None)
    if row_95 is None:
        raise ValueError("--contract requires 0.95 to be one of the --confidence levels")
    return {
        "risk": {
            "var_95": row_95["historical"]["var"],
            "cvar_95": row_95["historical"]["cvar"],
            "max_dd_sim": result["drawdown_stats"]["mean"],
            "horizon_days": result["horizon_days"],
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)

    print(f"fixture={result['fixture']} n_obs={result['n_obs']} horizon_days={result['horizon_days']} paths={result['n_paths']} seed={result['seed']}")
    print(f"fitted daily mu={result['mu_daily']:+.6f} sigma={result['sigma_daily']:.6f}  (annualized mu={result['mu_annual']:+.4f} sigma={result['sigma_annual']:.4f})")
    print()
    print("VaR / CVaR by method and confidence (returns over the horizon, negative = loss)")
    print(f"{'conf':>6} {'method':>12} {'VaR':>9} {'CVaR':>9}")
    for row in result["by_confidence"]:
        c = row["confidence"]
        for method in ("historical", "parametric", "mc"):
            m = row[method]
            print(f"{c:>6.0%} {method:>12} {m['var']:>9.4f} {m['cvar']:>9.4f}")

    print()
    print("Monte Carlo max-drawdown distribution over the horizon")
    dd = result["drawdown_stats"]
    print(f"n_paths={dd['n_paths']} mean={dd['mean']:.4f} worst={dd['worst']:.4f}")
    print(
        "percentiles: "
        + ", ".join(f"p{p}={v:+.4f}" for p, v in dd["percentiles"].items())
    )
    print(f"realized max drawdown over the fixture window (single historical path): {result['realized_max_drawdown']:+.4f}")

    print()
    print("Kupiec proportion-of-failures backtest (1-day parametric VaR vs full historical sample, in-sample)")
    ok = True
    for row in result["by_confidence"]:
        k = row["kupiec"]
        c = row["confidence"]
        verdict = "REJECTED" if k["reject_at_5pct"] else "not rejected"
        print(
            f"{c:>6.0%}: var_1day={row['var_1day']:+.4f} exceptions={k['n_exceptions']}/{k['n_obs']} "
            f"(expected {k['expected_exceptions']:.1f}) LR={k['lr_stat']:.3f} p={k['p_value']:.4f} -> {verdict}"
        )
        if k["reject_at_5pct"]:
            ok = False

    if not ok:
        print(
            "NOTE: at least one confidence level's VaR model is rejected by the Kupiec test at the 5% level - "
            "reported honestly, not smoothed over. See README limitations.",
            file=sys.stderr,
        )

    if result["drawdown_plot"]:
        print(f"wrote drawdown distribution plot ({dd['n_paths']} paths) to {result['drawdown_plot']}")

    if args.contract:
        contract_path = Path(args.contract)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(to_contract(result), indent=2) + "\n")
        print(f"wrote risk contract to {contract_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

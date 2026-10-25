"""CLI: out-of-sample VaR backtest, GARCH-t against the constant-volatility Gaussian control.

    python -m mcsim.oosvar --fixture fixtures/ohlcv/GULFOILLUB_NS.csv --window 500

The procedure and pass criteria were fixed before any result was computed: ``docs/PREREG_oos_var.md``.
Exits 0 either way; the verdict is printed, not hidden in an exit code.
"""
from __future__ import annotations

import argparse
import json

from mcsim.garch import GarchTModel, GaussianModel
from mcsim.returns import load_daily_log_returns
from mcsim.risk import rolling_var_backtest

CONF = (0.95, 0.99)


def verdict(res: dict) -> dict:
    ok = {"kupiec_95": res[0.95]["kupiec_p"] >= 0.05, "kupiec_99": res[0.99]["kupiec_p"] >= 0.05,
          "cc_95": res[0.95]["p_cc"] >= 0.05, "cc_99": res[0.99]["p_cc"] >= 0.05,
          "rate_99_in_band": 0.005 <= res[0.99]["exception_rate"] <= 0.015}
    return {**ok, "all_pass": all(ok.values())}


def run(fixture: str, window: int, refit_every: int) -> dict:
    r = load_daily_log_returns(fixture)
    out = {"fixture": fixture, "window": window, "refit_every": refit_every, "n_returns": int(len(r))}
    for name, cls in (("garch_t", GarchTModel), ("gaussian_control", GaussianModel)):
        res = rolling_var_backtest(r, cls, window, refit_every, CONF)
        out[name] = {"by_confidence": {str(c): v for c, v in res.items()}, "verdict": verdict(res)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", default="fixtures/ohlcv/GULFOILLUB_NS.csv")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--refit-every", type=int, default=20)
    ap.add_argument("--json", default=None, help="write the full result to this path")
    a = ap.parse_args(argv)
    out = run(a.fixture, a.window, a.refit_every)
    print(
        "Educational research on historical data. Not investment advice. "
        "Not a SEBI-registered research analyst."
    )
    for name in ("garch_t", "gaussian_control"):
        print(f"\n{name}  (window {a.window}, refit every {a.refit_every}, {out['n_returns'] - a.window} test days)")
        for c, v in out[name]["by_confidence"].items():
            print(f"  {float(c):.0%}  breaches {v['n_exceptions']}/{v['n_test']} (expected {v['expected_exceptions']:.1f}, rate {v['exception_rate']:.2%})  "
                  f"Kupiec p={v['kupiec_p']:.3f}  independence p={v['p_ind']:.3f}  conditional coverage p={v['p_cc']:.3f}")
        print("  verdict:", {k: v for k, v in out[name]["verdict"].items()})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

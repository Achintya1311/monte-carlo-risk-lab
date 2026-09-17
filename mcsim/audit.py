"""Day 7: pipeline audit for the commodity curve module.

Two checks, both mechanical rather than by-inspection, in the spirit of the
correctness identities the rest of this repo leans on (Day 4's barrier
knock-in/knock-out, Day 6's curve telescoping identity):

1. **Causality.** ``classify_regime`` and ``annualized_basis`` are the two
   functions that turn a raw futures curve into the regime label and basis
   number everything downstream (the CLI printout, ``regime_plot``, and the
   window labels fed to ``regime_conditioned_var_cvar``) depends on. Both
   are elementwise today, so they are causal by construction - but "by
   construction" is exactly the kind of claim a later refactor (a smoothed
   regime signal, a rolling-average basis) can quietly break without
   anyone noticing, since nothing else in the test suite would catch it.
   This checks it directly: a function's value for day *i* on the full
   series must equal its value for day *i* when every day after *i* is
   simply not there yet. If truncating the future ever changes a past
   value, the function is not causal and the audit fails.

2. **Fixture alignment.** ``mcsim.curve`` reads the futures curve
   (``load_curve``, which keeps dates) and the front-month price series
   (``mcsim.returns.load_daily_log_returns``, which drops them) as two
   separate files and lines them up *positionally* - ``window_regime =
   regime[:-1]`` assumes row *i* of one file is the same trading day as row
   *i* of the other. Nothing enforces that. For the committed fixtures it
   happens to hold (checked below), but a future fixture refresh or a live
   feed with a gap or holiday mismatch would silently mislabel every
   regime-conditioned VaR window without raising an error - the exact
   failure mode a mechanical check exists to catch instead of assuming
   away.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mcsim.commodity import annualized_basis, classify_regime, load_curve
from mcsim.curve import MONTHS_C1_TO_C4
from mcsim.returns import load_dates

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Sample every Nth bar, same rationale as STOCKSTALKER's audit: O(n) calls to
# the function under test per sample, and every bar would make this O(n^2)
# for no extra confidence once the sample spans burn-in, mid-series and tail.
SAMPLE_STRIDE = 15


def causality_violations(
    name: str,
    fn: Callable[..., np.ndarray],
    arrays: tuple[np.ndarray, ...],
    dates: list[str],
    min_index: int = 0,
) -> list[str]:
    """Bars where truncating every input array after bar i changes bar i's value.

    ``fn`` takes the same positional arrays as ``arrays`` (the full series)
    and returns a series aligned to their shared length. Compares that
    series computed on the full arrays against the same series recomputed
    on arrays truncated to bar ``i``, for a sample of bars. Returns one
    message per bar where the two disagree; an empty list means no
    look-ahead was detected.
    """
    full = fn(*arrays)
    n = len(full)
    violations = []
    for i in range(min_index, n, SAMPLE_STRIDE):
        truncated = fn(*(a[: i + 1] for a in arrays))
        full_value = full[i]
        truncated_value = truncated[-1]
        if not _close_enough(full_value, truncated_value):
            violations.append(
                f"{name}: bar {i} ({dates[i]}) = {full_value!r} on the full series but "
                f"{truncated_value!r} truncated at that bar - depends on data that had not "
                "happened yet"
            )
    return violations


def _close_enough(a: Any, b: Any, tol: float = 1e-9) -> bool:
    try:
        return bool(abs(float(a) - float(b)) <= tol)
    except (TypeError, ValueError):
        return bool(a == b)


def curve_checks() -> dict[str, Callable[..., np.ndarray]]:
    """The curve-derived functions everything downstream depends on.

    One entry per function actually wired into ``mcsim.curve.run`` - if a
    new curve-derived signal joins it without an entry here, it runs
    unaudited.
    """
    return {
        "classify_regime(c1,c4)": lambda c1, c4: classify_regime(c1, c4),
        "annualized_basis(c1,c4)": lambda c1, c4: annualized_basis(c1, c4, MONTHS_C1_TO_C4),
    }


def audit_curve(curve_path: str) -> list[str]:
    curve = load_curve(curve_path)
    violations = []
    for name, fn in curve_checks().items():
        violations += causality_violations(name, fn, (curve["c1"], curve["c4"]), curve["dates"])
    return violations


def alignment_check(curve_path: str, prices_path: str) -> dict[str, Any]:
    """Whether the curve and price fixtures share one trading-day calendar, in order.

    This is what makes ``mcsim.curve``'s positional ``window_regime =
    regime[:-1]`` alignment valid - the CLI itself never checks it.
    """
    curve_dates = load_curve(curve_path)["dates"]
    price_dates = load_dates(prices_path)

    n = min(len(curve_dates), len(price_dates))
    mismatches = [
        {"index": i, "curve_date": curve_dates[i], "price_date": price_dates[i]}
        for i in range(n)
        if curve_dates[i] != price_dates[i]
    ]
    aligned = not mismatches and len(curve_dates) == len(price_dates)

    return {
        "curve_path": curve_path,
        "prices_path": prices_path,
        "n_curve_days": len(curve_dates),
        "n_price_days": len(price_dates),
        "aligned": aligned,
        "mismatches": mismatches,
    }


def run_audit(curve_path: str, prices_path: str) -> dict[str, Any]:
    causality_findings = audit_curve(curve_path)
    alignment = alignment_check(curve_path, prices_path)
    return {
        "curve_path": curve_path,
        "prices_path": prices_path,
        "causality_findings": causality_findings,
        "causality_clean": not causality_findings,
        "alignment": alignment,
        "clean": not causality_findings and alignment["aligned"],
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Pipeline audit - {report['curve_path']} vs {report['prices_path']}")

    if report["causality_clean"]:
        print("  causality: no look-ahead detected in classify_regime / annualized_basis")
    else:
        print(f"  causality: {len(report['causality_findings'])} violation(s)")
        for v in report["causality_findings"]:
            print(f"    - {v}")

    a = report["alignment"]
    if a["aligned"]:
        print(
            f"  alignment: {a['n_curve_days']} trading days match 1:1 in order between the two "
            "fixtures - mcsim.curve's positional window_regime alignment holds"
        )
    else:
        print(
            f"  alignment: MISMATCH - {a['n_curve_days']} curve day(s) vs {a['n_price_days']} "
            f"price day(s), {len(a['mismatches'])} row(s) disagree - mcsim.curve's positional "
            "window_regime alignment would silently mislabel windows"
        )
        for m in a["mismatches"][:10]:
            print(f"    - row {m['index']}: curve={m['curve_date']} price={m['price_date']}")


def write_markdown(report: dict[str, Any], path: Path, generated: str) -> None:
    lines = [
        f"# Pipeline audit — {generated}",
        "",
        "Checks two things about the commodity curve module: that "
        "`classify_regime`/`annualized_basis` are causal (a value for day "
        "*i* survives truncating every day after it), and that the curve "
        "and front-month price fixtures share one trading-day calendar in "
        "the same order, which is what makes `mcsim.curve`'s positional "
        "`window_regime` alignment valid.",
        "",
        f"**Causality:** {'clean' if report['causality_clean'] else 'violations found'}. "
        f"**Alignment:** {'holds' if report['alignment']['aligned'] else 'MISMATCH'}.",
        "",
    ]
    if report["causality_findings"]:
        lines.append("## Causality violations")
        lines.append("")
        lines += [f"- {v}" for v in report["causality_findings"]]
        lines.append("")
    a = report["alignment"]
    if not a["aligned"]:
        lines.append("## Alignment mismatches")
        lines.append("")
        lines.append(f"curve={a['curve_path']} ({a['n_curve_days']} days), prices={a['prices_path']} ({a['n_price_days']} days)")
        lines.append("")
        for m in a["mismatches"][:50]:
            lines.append(f"- row {m['index']}: curve={m['curve_date']} price={m['price_date']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--curve", type=str, default="fixtures/commodity/WTI_futures_curve.csv",
        help="CSV with date,c1,c2,c3,c4 contract-month closes (default: fixtures/commodity/WTI_futures_curve.csv)",
    )
    parser.add_argument(
        "--prices", type=str, default="fixtures/commodity/WTI_C1.csv",
        help="CSV with date,close for the front-month contract (default: fixtures/commodity/WTI_C1.csv)",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--date", default=None, help="override the generated date (mainly for tests)")
    args = parser.parse_args(argv)

    generated = args.date or date.today().isoformat()
    report = run_audit(args.curve, args.prices)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(report, out_dir / f"audit_{generated}.md", generated)

    print_report(report)
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

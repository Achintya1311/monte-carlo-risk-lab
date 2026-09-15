"""Convergence sweep and plot: MC option price vs Black-Scholes as N grows.

Each grid point is an independent Monte Carlo run (same seed, different
path count - the RNG draws a differently shaped block of normals per N,
not a prefix of one big run), so the shrinking error band reflects
independent sampling, not overlap between points.
"""
from __future__ import annotations

DEFAULT_PATH_GRID: tuple[int, ...] = (100, 1_000, 10_000, 100_000, 1_000_000)


def convergence_sweep(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    days: int,
    option_type: str = "call",
    seed: int | None = 42,
    path_grid: tuple[int, ...] = DEFAULT_PATH_GRID,
) -> list[dict]:
    """Run ``mc_option_price`` at each N in ``path_grid``, same inputs and seed."""
    from mcsim.pricing import mc_option_price

    return [
        mc_option_price(
            spot=spot, strike=strike, rate=rate, vol=vol, days=days,
            n_paths=n, option_type=option_type, seed=seed,
        )
        for n in path_grid
    ]


def plot_convergence(rows: list[dict], bs_reference: float, out_path: str, option_type: str) -> None:
    """Write a log-N convergence plot: MC price ± 95% CI vs the BS reference line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n_paths"] for r in rows]
    prices = [r["price"] for r in rows]
    ci95 = [1.96 * r["std_error"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ns, prices, yerr=ci95, fmt="o-", capsize=4, label="Monte Carlo price (95% CI)")
    ax.axhline(bs_reference, color="black", linestyle="--", label=f"Black-Scholes = {bs_reference:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("number of paths (N)")
    ax.set_ylabel(f"{option_type} price")
    ax.set_title("Monte Carlo convergence to Black-Scholes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

"""17_engine_market_blend.py — Convex blend engine ⊕ market on full 380.

The engine and the market are two probabilistic predictors. If the
engine adds even a small amount of orthogonal signal — even if it
loses on Brier alone — the linear blend

    p_blend(alpha) = alpha * p_engine + (1 - alpha) * p_market

minimises Brier at some alpha* > 0. Beating the market with the
blend is a market-participation result: you don't need the engine to
be better standalone, only to be non-collinear.

This experiment reuses the posteriors cached by
`experiments/16_reliability_calibration.py` — so it's a post-hoc,
seconds-long sweep once exp 16 has run.

Sweep resolution: 0.02 over [0.00, 1.00] → 51 points. Bootstrap 90%
CI on the delta at alpha* against alpha=0 (pure market).

Run with (after exp 16):

    PYTHONPATH=src python3 experiments/17_engine_market_blend.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SEED = 20260702
CACHE_PATH = ROOT / "experiments" / "_cache_16_posteriors.csv"


def load_rows():
    if not CACHE_PATH.exists():
        raise SystemExit(
            f"Missing cached posteriors at {CACHE_PATH}.\n"
            f"Run experiments/16_reliability_calibration.py first."
        )
    with CACHE_PATH.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in list(r.keys()):
            if k in ("match_id", "date", "event",
                      "actual_hda", "actual_ou"):
                continue
            r[k] = float(r[k])
    return rows


def brier_for_alpha(rows, alpha: float, market: str) -> np.ndarray:
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        if market == "hda":
            keys = [("h", "home"), ("d", "draw"), ("a", "away")]
            actual = r["actual_hda"]
        else:
            keys = [("o", "over"), ("u", "under")]
            actual = r["actual_ou"]
        b = 0.0
        for suf, side in keys:
            p = alpha * r[f"eng_{suf}"] + (1 - alpha) * r[f"mkt_{suf}"]
            y = 1.0 if actual == side else 0.0
            b += (p - y) ** 2
        out[i] = b
    return out


def sweep_alpha(rows, market: str, grid: np.ndarray):
    means = np.empty_like(grid)
    for i, a in enumerate(grid):
        means[i] = brier_for_alpha(rows, float(a), market).mean()
    return means


def bootstrap_ci_delta_alpha(rows, alpha_star: float, market: str,
                              n_boot=2000):
    """CI on the per-match Brier delta at alpha_star vs alpha=0 (market)."""
    brier_star = brier_for_alpha(rows, alpha_star, market)
    brier_mkt = brier_for_alpha(rows, 0.0, market)
    deltas = brier_star - brier_mkt
    rng = np.random.default_rng(SEED)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(deltas, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def main() -> None:
    rows = load_rows()
    print("=== Orbita: engine ⊕ market alpha-blend on full 380 ===")
    print(f"Panel : {len(rows)} matches (from cache)")
    print()

    grid = np.arange(0.00, 1.01, 0.02)
    for market in ("hda", "ou"):
        means = sweep_alpha(rows, market, grid)
        i_star = int(np.argmin(means))
        a_star = float(grid[i_star])
        b_star = float(means[i_star])
        b_mkt = float(means[0])
        b_eng = float(means[-1])
        print(f"--- {market.upper()} sweep ---")
        print(f"  alpha=0.00 (market)  Brier={b_mkt:.4f}")
        print(f"  alpha=1.00 (engine)  Brier={b_eng:.4f}")
        print(f"  alpha={a_star:.2f} (best)  Brier={b_star:.4f}  "
              f"delta_vs_market={b_star - b_mkt:+.4f}")
        if a_star == 0.0:
            print("  optimum at alpha=0 — engine adds NO orthogonal signal.")
        elif a_star == 1.0:
            print("  optimum at alpha=1 — market adds NO orthogonal signal.")
        else:
            print(f"  optimum interior at alpha={a_star:.2f} — "
                  f"engine contributes {a_star*100:.0f}% weight.")
            d, lo, hi = bootstrap_ci_delta_alpha(rows, a_star, market)
            v = ("blend BEATS market" if hi < 0 else
                 "market BEATS blend" if lo > 0 else "tied (CI ∋ 0)")
            print(f"  bootstrap 90% CI on delta vs market: "
                  f"d={d:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {v}")
        print("  grid (subsampled 0.00, 0.10, ..., 1.00):")
        for j in range(0, len(grid), 5):
            marker = " <" if int(np.argmin(means)) == j else ""
            print(f"    alpha={grid[j]:.2f}  Brier={means[j]:.4f}{marker}")
        print()


if __name__ == "__main__":
    main()

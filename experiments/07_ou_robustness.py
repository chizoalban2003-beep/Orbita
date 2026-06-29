"""07_ou_robustness.py — Stress-test the Goals O/U Brier win.

Experiment 06 found that the multi-market joint posterior (Config B from
experiment 05) beats the de-vigged Bet365 closing line on Goals O/U 2.5
by 0.020 Brier on a 50-match EPL panel. This experiment runs three
robustness checks:

1. **Monte Carlo noise**: re-run at N_TRIALS = 100 to confirm the win
   isn't a fluctuation of the N=30 estimate.
2. **LOOCV stability**: leave-one-match-out cross-validation on the
   panel — bin each match's per-trial Brier into the LOOCV mean.
3. **Bootstrap confidence interval**: 2000 bootstrap resamples of the
   per-match Brier deltas, report the 5/95 percentiles. If zero lies
   inside the CI, the win is not statistically distinguishable from
   noise on this sample.

Uses Config B (multi-market, no players) — it had the best O/U hit-rate
(62%) and beats both the market and Config C on O/U Brier. Players don't
help on O/U because they live on the home/away axis, not the goals axis.

Run with::

    PYTHONPATH=src python3 experiments/07_ou_robustness.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import EventSpace  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e05", ROOT / "experiments" / "05_player_attractor_panel.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)


def per_match_brier(matches, n_trials: int) -> tuple:
    """Return (engine_brier_per_match, market_brier_per_match) for O/U."""
    eng = []
    mkt = []
    e05.N_TRIALS = n_trials
    for i, m in enumerate(matches, 1):
        wells_m = e05.multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"],
            m["p_over"], m["p_under"],
        )
        space_m = EventSpace(wells_m)
        joint = e05.run(space_m, space_m)
        ou_eng = e05.marginal_ou(joint)
        ou_mkt = {"over": m["p_over"], "under": m["p_under"]}
        eng.append(e05.brier(ou_eng, m["actual_ou"]))
        mkt.append(e05.brier(ou_mkt, m["actual_ou"]))
        if i % 10 == 0:
            print(f"  [{i:>2d}/{len(matches)}] processed", flush=True)
    return np.array(eng), np.array(mkt)


def bootstrap_ci(deltas: np.ndarray, n_boot: int, seed: int = 20260629):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def loocv(eng: np.ndarray, mkt: np.ndarray) -> tuple:
    n = len(eng)
    eng_loo = np.empty(n)
    mkt_loo = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        eng_loo[i] = eng[mask].mean()
        mkt_loo[i] = mkt[mask].mean()
    return float(eng_loo.mean()), float(mkt_loo.mean())


def main() -> None:
    print("=== Orbita: O/U robustness on Config B ===")
    matches = e05.load_matches_with_ou()
    print(f"Panel size : {len(matches)}\n")

    print("Pass 1: N_TRIALS = 30 (baseline)")
    eng30, mkt30 = per_match_brier(matches, 30)
    print(f"  engine Brier = {eng30.mean():.4f}")
    print(f"  market Brier = {mkt30.mean():.4f}")
    print(f"  delta        = {eng30.mean() - mkt30.mean():+.4f}\n")

    print("Pass 2: N_TRIALS = 100 (low-noise)")
    eng100, mkt100 = per_match_brier(matches, 100)
    print(f"  engine Brier = {eng100.mean():.4f}")
    print(f"  market Brier = {mkt100.mean():.4f}")
    print(f"  delta        = {eng100.mean() - mkt100.mean():+.4f}\n")

    print("LOOCV (N=100 cache)")
    eng_loo, mkt_loo = loocv(eng100, mkt100)
    print(f"  engine LOOCV Brier = {eng_loo:.4f}")
    print(f"  market LOOCV Brier = {mkt_loo:.4f}")
    print(f"  LOOCV delta        = {eng_loo - mkt_loo:+.4f}\n")

    deltas = eng100 - mkt100
    mean, lo, hi = bootstrap_ci(deltas, n_boot=2000)
    print("Bootstrap (2000 resamples of per-match deltas, N=100 cache)")
    print(f"  mean delta = {mean:+.4f}")
    print(f"  90% CI     = [{lo:+.4f}, {hi:+.4f}]")
    if hi < 0:
        print("  >>> CI excludes zero on the engine side: engine BEATS market.")
    elif lo > 0:
        print("  >>> CI excludes zero on the market side: market BEATS engine.")
    else:
        print("  >>> CI includes zero: difference not statistically detectable.")

    print()
    print("Per-match deltas (negative = engine beats market on that match):")
    for i, (m, d) in enumerate(zip(matches, deltas)):
        marker = "▼" if d < 0 else "▲"
        print(f"  [{i+1:>2d}] {marker} {d:+.3f}  {m['date']} {m['home_team']} "
              f"vs {m['away_team']}  ({m['actual_ou']})")
    win_count = int((deltas < 0).sum())
    print(f"\nEngine wins on {win_count}/{len(deltas)} matches.")


if __name__ == "__main__":
    main()

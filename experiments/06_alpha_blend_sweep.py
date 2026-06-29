"""06_alpha_blend_sweep.py — Find the optimal market-engine blend.

The 50-match panel (experiment 05) revealed an interesting asymmetry:

* Config C (multi-market + player attractors) picks the right H/D/L
  outcome 46% of the time vs the bookmaker's 44%.
* Yet Config C's Brier is 0.652 vs the market's 0.628.

That gap is the signature of *over-spread* probabilities. The engine
finds the right basin more often than the market — but spreads too
much mass to the other two outcomes once it gets there. The classic
treatment is a convex blend with the market:

    p_blend = alpha * p_engine + (1 - alpha) * p_market

We sweep alpha across [0, 1] and look for the point where Brier
minimises. If the minimum lies strictly inside (0, 1) the engine is
contributing orthogonal signal to the market.

Caches Config C joint posteriors so the sweep itself is cheap.

Run with::

    PYTHONPATH=src python3 experiments/06_alpha_blend_sweep.py
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import EventSpace  # noqa: E402

# Import experiment 05 for the panel + simulators
spec = importlib.util.spec_from_file_location(
    "e05", ROOT / "experiments" / "05_player_attractor_panel.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)
e03 = e05.e03


def cache_engine_posteriors(matches):
    """Run Config C once per match, return per-match dicts of
    {hda: {...}, ou: {...}} engine posteriors."""
    elo_cache: dict = {}
    out = []
    for i, m in enumerate(matches, 1):
        wells_m = e05.multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"],
            m["p_over"], m["p_under"],
        )
        space_m = EventSpace(wells_m)
        r_h = e03.elo_for(m["home_team"], m["date"], elo_cache)
        r_a = e03.elo_for(m["away_team"], m["date"], elo_cache)
        home_xi = e05.synth_lineup(m["home_team"], +1, r_h)
        away_xi = e05.synth_lineup(m["away_team"], -1, r_a)
        players = (e05.player_attractors(home_xi, m["home_label"]) +
                   e05.player_attractors(away_xi, m["away_label"]))
        force_space = EventSpace(list(space_m.attractors) + players)
        joint_c = e05.run(force_space, space_m)
        hda = e05.marginal_hda(joint_c, m["home_label"], m["away_label"])
        ou = e05.marginal_ou(joint_c)
        out.append((m, hda, ou))
        print(f"[{i:>2d}/{len(matches)}] cached "
              f"{m['date']} {m['home_team']} vs {m['away_team']}",
              flush=True)
    return out


def blend(p_engine: dict, p_market: dict, alpha: float) -> dict:
    return {k: alpha * p_engine[k] + (1 - alpha) * p_market[k]
            for k in p_market}


def brier(probs: dict, actual: str) -> float:
    return sum((p - (1.0 if l == actual else 0.0)) ** 2
               for l, p in probs.items())


def main() -> None:
    print("=== Orbita: alpha-blend sweep on Config C ===")
    matches = e05.load_matches_with_ou()
    print(f"Panel size : {len(matches)} matches")
    print()

    cache = cache_engine_posteriors(matches)
    print()

    alphas = np.linspace(0.0, 1.0, 21)
    rows = []
    for a in alphas:
        b_hda = []
        b_ou = []
        for m, eng_hda, eng_ou in cache:
            mkt_hda = {m["home_label"]: m["p_h"],
                       "draw": m["p_d"],
                       m["away_label"]: m["p_a"]}
            mkt_ou = {"over": m["p_over"], "under": m["p_under"]}
            bl_hda = blend(eng_hda, mkt_hda, a)
            bl_ou = blend(eng_ou, mkt_ou, a)
            b_hda.append(brier(bl_hda, m["actual_hda"]))
            b_ou.append(brier(bl_ou, m["actual_ou"]))
        rows.append((float(a), float(np.mean(b_hda)), float(np.mean(b_ou))))

    print(f"{'alpha':>7s}  {'HDA Brier':>10s}  {'O/U Brier':>10s}")
    for a, hda, ou in rows:
        print(f"{a:>7.2f}  {hda:>10.4f}  {ou:>10.4f}")

    best_hda = min(rows, key=lambda r: r[1])
    best_ou = min(rows, key=lambda r: r[2])
    print()
    print(f"best alpha (HDA): {best_hda[0]:.2f}  Brier={best_hda[1]:.4f}  "
          f"(market alone = {rows[0][1]:.4f}, engine alone = {rows[-1][1]:.4f})")
    print(f"best alpha (O/U): {best_ou[0]:.2f}  Brier={best_ou[2]:.4f}  "
          f"(market alone = {rows[0][2]:.4f}, engine alone = {rows[-1][2]:.4f})")

    delta_hda = rows[0][1] - best_hda[1]
    delta_ou = rows[0][2] - best_ou[2]
    print()
    if delta_hda > 0:
        print(f"H/D/L: blend BEATS market by {delta_hda:+.4f} Brier at "
              f"alpha={best_hda[0]:.2f}")
    else:
        print(f"H/D/L: blend does NOT beat market.")
    if delta_ou > 0:
        print(f"O/U:   blend BEATS market by {delta_ou:+.4f} Brier at "
              f"alpha={best_ou[0]:.2f}")
    else:
        print(f"O/U:   blend does NOT beat market.")


if __name__ == "__main__":
    main()

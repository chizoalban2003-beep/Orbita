"""10_anisotropic_drag_sweep.py — Does anisotropic drag move H/D/L Brier?

The H/D/L event-space has a natural geometric asymmetry:

  * x-axis = home/away win axis. A body with momentum along x is
    expressing "directional momentum" — a side that's pressing.
  * y-axis = draw axis. A body drifting on y has no directional
    pressure, it's just being pulled into the central well.

Symmetric isotropic drag treats both axes the same. But intuitively
we may want to bleed the y-component (resist drift into draw) while
preserving x-momentum (let directional pressure carry through to a
win-side well). Or the opposite: bleed x (force convergence to the
favourite), preserve y. The experiment is which (if either) wins.

We sweep the C_d_x × C_d_y plane on the 50-match panel against the
de-vigged Bet365 closing line, using the multi-market (Config B)
event-space so we score both H/D/L Brier and O/U Brier.

Run with::

    PYTHONPATH=src python3 experiments/10_anisotropic_drag_sweep.py
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
e03 = e05.e03

# Pin trial count for a clean comparison.
e05.N_TRIALS = 30

# Sweep grid: two coefficients ∈ {0.0, 0.02, 0.04, 0.08, 0.16}.
# 0.04 is the current isotropic default (the diagonal cell is the
# baseline; we want to see if any off-diagonal cell beats it).
GRID = [0.0, 0.02, 0.04, 0.08, 0.16]


def run_one_config(cd_x: float, cd_y: float, matches: list) -> tuple:
    """Returns (mean_hda_brier, mean_ou_brier) for Config B over the panel."""
    cd = np.array([cd_x, cd_y])
    hda = []
    ou = []
    # Temporarily monkey-patch e05's C_D used by .run()
    orig = e05.C_D
    e05.C_D = cd
    try:
        for m in matches:
            wells_m = e05.multimarket_wells(
                m["home_label"], m["away_label"],
                m["p_h"], m["p_d"], m["p_a"],
                m["p_over"], m["p_under"],
            )
            space_m = EventSpace(wells_m)
            joint = e05.run(space_m, space_m)
            hda_post = e05.marginal_hda(joint, m["home_label"], m["away_label"])
            ou_post = e05.marginal_ou(joint)
            hda.append(e05.brier(hda_post, m["actual_hda"]))
            ou.append(e05.brier(ou_post, m["actual_ou"]))
    finally:
        e05.C_D = orig
    return float(np.mean(hda)), float(np.mean(ou))


def main() -> None:
    matches = e05.load_matches_with_ou()
    print("=== Orbita: anisotropic drag sweep on Config B ===")
    print(f"Panel : {len(matches)} matches (2024/25 EPL Oct–Nov)")
    print(f"Trials: {e05.N_TRIALS}/match")
    print(f"Grid  : C_d_x, C_d_y ∈ {GRID}")
    print()

    # Market baseline
    mkt_hda = []
    mkt_ou = []
    for m in matches:
        p_mkt_hda = {m["home_label"]: m["p_h"],
                     "draw": m["p_d"],
                     m["away_label"]: m["p_a"]}
        p_mkt_ou = {"over": m["p_over"], "under": m["p_under"]}
        mkt_hda.append(e05.brier(p_mkt_hda, m["actual_hda"]))
        mkt_ou.append(e05.brier(p_mkt_ou, m["actual_ou"]))
    mkt_hda_mean = float(np.mean(mkt_hda))
    mkt_ou_mean = float(np.mean(mkt_ou))
    print(f"Market Brier baseline: HDA={mkt_hda_mean:.4f}  O/U={mkt_ou_mean:.4f}")
    print()

    print("--- H/D/L Brier (engine - market; - means engine wins) ---")
    header = "C_d_x\\C_d_y  " + "  ".join(f"{c:>7.2f}" for c in GRID)
    print(header)
    hda_grid = {}
    for cd_x in GRID:
        cells = []
        for cd_y in GRID:
            h, o = run_one_config(cd_x, cd_y, matches)
            hda_grid[(cd_x, cd_y)] = (h, o)
            cells.append(h - mkt_hda_mean)
        print(f"{cd_x:>9.2f}    " + "  ".join(f"{c:>+7.4f}" for c in cells),
              flush=True)

    print()
    print("--- O/U Brier (engine - market; - means engine wins) ---")
    print(header)
    for cd_x in GRID:
        cells = []
        for cd_y in GRID:
            h, o = hda_grid[(cd_x, cd_y)]
            cells.append(o - mkt_ou_mean)
        print(f"{cd_x:>9.2f}    " + "  ".join(f"{c:>+7.4f}" for c in cells))

    # Find best cell on each market
    print()
    best_hda = min(hda_grid.items(), key=lambda kv: kv[1][0])
    best_ou = min(hda_grid.items(), key=lambda kv: kv[1][1])
    (bx, by), (bh, _) = best_hda
    print(f"Best HDA cell: C_d=({bx:.2f}, {by:.2f})  Brier={bh:.4f}  "
          f"delta={bh - mkt_hda_mean:+.4f}")
    (bx, by), (_, bo) = best_ou
    print(f"Best O/U cell: C_d=({bx:.2f}, {by:.2f})  Brier={bo:.4f}  "
          f"delta={bo - mkt_ou_mean:+.4f}")


if __name__ == "__main__":
    main()

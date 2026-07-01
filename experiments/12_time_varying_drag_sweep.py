"""12_time_varying_drag_sweep.py — Does a drag SCHEDULE beat constant drag?

Experiment 11 showed anisotropic drag (0.00, 0.16) halves the full-season
Brier gap vs isotropic. But `0.16` is still a single number applied
uniformly over the 90 minutes. Match minutes aren't homogeneous:

  * Early game: fresh legs, low fatigue → little drag needed.
  * Mid game: teams settle, momentum patterns emerge.
  * Late game: fatigue accumulates, desperation reverses (a losing side
    may thrust *toward* an away-win well instead of decaying toward draw).

Physical hypothesis: a schedule that RAMPS y-drag up over the match beats
a constant y-drag applied throughout. We sweep a linear ramp
`Cy(t) = Cy_start + (Cy_end − Cy_start) · t/duration` on the 50-match panel.

x-drag pinned at 0 (from experiment 10's finding). y-drag start/end each
∈ {0.00, 0.08, 0.16, 0.32}, 4×4 = 16 configs. The diagonal (start=end)
reproduces the constant-drag anisotropic baseline; the ramp-up cells are
the new proposal.

Run with::

    PYTHONPATH=src python3 experiments/12_time_varying_drag_sweep.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import EventSpace, linear_ramp_schedule  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e05", ROOT / "experiments" / "05_player_attractor_panel.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)
e03 = e05.e03

e05.N_TRIALS = 30

GRID = [0.00, 0.08, 0.16, 0.32]


def run_one_schedule(cy_start: float, cy_end: float, matches: list) -> tuple:
    """Return (mean_hda_brier, mean_ou_brier) for Config B under a
    linear-ramp y-drag schedule."""
    sched = linear_ramp_schedule(
        [0.0, cy_start], [0.0, cy_end], duration=e05.DURATION,
    )
    orig = e05.C_D
    e05.C_D = sched
    hda = []
    ou = []
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
    print("=== Orbita: time-varying drag sweep (linear ramp on y-axis) ===")
    print(f"Panel : {len(matches)} matches (2024/25 EPL Oct–Nov)")
    print(f"Trials: {e05.N_TRIALS}/match")
    print(f"Grid  : Cy_start, Cy_end ∈ {GRID}  (x-drag pinned to 0)")
    print()

    mkt_hda = []
    mkt_ou = []
    for m in matches:
        p_mkt_hda = {m["home_label"]: m["p_h"], "draw": m["p_d"],
                     m["away_label"]: m["p_a"]}
        p_mkt_ou = {"over": m["p_over"], "under": m["p_under"]}
        mkt_hda.append(e05.brier(p_mkt_hda, m["actual_hda"]))
        mkt_ou.append(e05.brier(p_mkt_ou, m["actual_ou"]))
    mkt_hda_mean = float(np.mean(mkt_hda))
    mkt_ou_mean = float(np.mean(mkt_ou))
    print(f"Market baseline: HDA={mkt_hda_mean:.4f}  O/U={mkt_ou_mean:.4f}")
    print()

    print("--- H/D/L Brier (engine - market; - means engine wins) ---")
    header = "start\\end   " + "  ".join(f"{c:>7.2f}" for c in GRID)
    print(header)
    grid = {}
    for cs in GRID:
        cells = []
        for ce in GRID:
            h, o = run_one_schedule(cs, ce, matches)
            grid[(cs, ce)] = (h, o)
            cells.append(h - mkt_hda_mean)
        print(f"{cs:>9.2f}    " + "  ".join(f"{c:>+7.4f}" for c in cells),
              flush=True)

    print()
    print("--- O/U Brier (engine - market; - means engine wins) ---")
    print(header)
    for cs in GRID:
        cells = []
        for ce in GRID:
            _, o = grid[(cs, ce)]
            cells.append(o - mkt_ou_mean)
        print(f"{cs:>9.2f}    " + "  ".join(f"{c:>+7.4f}" for c in cells))

    print()
    best_hda = min(grid.items(), key=lambda kv: kv[1][0])
    best_ou = min(grid.items(), key=lambda kv: kv[1][1])
    (bs, be), (bh, _) = best_hda
    print(f"Best HDA schedule: Cy(0→90m)=({bs:.2f} → {be:.2f})  "
          f"Brier={bh:.4f}  delta={bh - mkt_hda_mean:+.4f}")
    (bs, be), (_, bo) = best_ou
    print(f"Best O/U schedule: Cy(0→90m)=({bs:.2f} → {be:.2f})  "
          f"Brier={bo:.4f}  delta={bo - mkt_ou_mean:+.4f}")

    # Ramp-up vs constant comparison — is the diagonal beaten?
    print()
    print("--- Ramp-up vs constant-drag (diagonals) ---")
    for cs, ce in [(0.00, 0.16), (0.00, 0.32), (0.08, 0.32), (0.16, 0.32)]:
        h, o = grid[(cs, ce)]
        const_h, const_o = grid[(ce, ce)]
        print(f"  ramp {cs:.2f}→{ce:.2f}  vs const {ce:.2f}: "
              f"HDA {h - const_h:+.4f}  O/U {o - const_o:+.4f}")


if __name__ == "__main__":
    main()

"""11_anisotropic_full_season.py — Anisotropic drag on the full season.

Experiment 10 swept the (C_d_x, C_d_y) grid on the 50-match panel and
found (0.00, 0.16) as the O/U Brier optimum — Brier 0.4484 vs market
0.4587, delta −0.010. But that's a 50-match sample and a grid winner
is guaranteed to look good on the panel it was picked from.

The honest test is whether the same setting still beats the market on
the FULL 2024/25 EPL season (380 matches) — the panel where the
isotropic-drag Config B lost O/U by +0.048 (CI [+0.025, +0.071]).

If (0.00, 0.16) still wins O/U on 380 matches with CI excluding zero
in the engine's favour, we have a genuine architectural result.
If the delta collapses back toward the +0.048 isotropic-drag baseline,
the 50-match win was sweep-hyperparameter noise.

Also reports HDA Brier and CI on the same panel — mostly as a
sanity-check that we're not just wrecking H/D/L to win O/U.

Run with::

    PYTHONPATH=src python3 experiments/11_anisotropic_full_season.py
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

spec2 = importlib.util.spec_from_file_location(
    "e08", ROOT / "experiments" / "08_full_season_panel.py")
e08 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(e08)

# From experiment 10 grid winner
BEST_CD_OU = np.array([0.00, 0.16])
BEST_CD_HDA_NAME = "(0.00, 0.00)"  # sweep winner was 0-drag; report only

# Two comparison configurations
CONFIGS = {
    "isotropic (baseline)": 0.04,
    "anisotropic best-O/U (0.00, 0.16)": BEST_CD_OU,
}

N_TRIALS = 30
e05.N_TRIALS = N_TRIALS


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000):
    rng = np.random.default_rng(e05.SEED)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(deltas, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def score_config(matches: list, C_d) -> tuple:
    """Return (hda_briers, ou_briers) arrays for Config B under drag=C_d."""
    orig = e05.C_D
    e05.C_D = C_d
    hda = []
    ou = []
    try:
        for i, m in enumerate(matches, 1):
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
            if i % 50 == 0:
                print(f"  [{i:>3d}/{len(matches)}]  "
                      f"HDA={np.mean(hda):.4f}  O/U={np.mean(ou):.4f}",
                      flush=True)
    finally:
        e05.C_D = orig
    return np.array(hda), np.array(ou)


def main() -> None:
    matches = e08.load_full_season()
    print("=== Orbita: anisotropic drag, out-of-sample full season ===")
    print(f"Panel : {len(matches)} matches (2024/25 EPL, entire season)")
    print(f"Trials: {N_TRIALS}/match")
    print(f"Best drag (from 50-match sweep, e10): C_d = {BEST_CD_OU}")
    print()

    # Market baseline
    mkt_hda = []
    mkt_ou = []
    for m in matches:
        p_mkt_hda = {m["home_label"]: m["p_h"], "draw": m["p_d"],
                     m["away_label"]: m["p_a"]}
        p_mkt_ou = {"over": m["p_over"], "under": m["p_under"]}
        mkt_hda.append(e05.brier(p_mkt_hda, m["actual_hda"]))
        mkt_ou.append(e05.brier(p_mkt_ou, m["actual_ou"]))
    mkt_hda = np.array(mkt_hda)
    mkt_ou = np.array(mkt_ou)
    print(f"Market baseline : HDA={mkt_hda.mean():.4f}  O/U={mkt_ou.mean():.4f}")
    print()

    results = {}
    for name, cd in CONFIGS.items():
        print(f"--- Scoring: {name}  (C_d={cd}) ---")
        hda, ou = score_config(matches, cd)
        results[name] = (hda, ou)
        print(f"  engine HDA Brier = {hda.mean():.4f}   O/U Brier = {ou.mean():.4f}")
        print()

    print("=== summary + bootstrap 90% CIs (engine − market) ===")
    for name, (hda, ou) in results.items():
        d_hda = hda - mkt_hda
        d_ou = ou - mkt_ou
        h_mean, h_lo, h_hi = bootstrap_ci(d_hda)
        o_mean, o_lo, o_hi = bootstrap_ci(d_ou)
        h_verdict = ("engine BEATS" if h_hi < 0 else
                     "market BEATS" if h_lo > 0 else "tied (CI ∋ 0)")
        o_verdict = ("engine BEATS" if o_hi < 0 else
                     "market BEATS" if o_lo > 0 else "tied (CI ∋ 0)")
        print(f"{name}")
        print(f"  HDA delta={h_mean:+.4f}  CI=[{h_lo:+.4f}, {h_hi:+.4f}]  {h_verdict}")
        print(f"  O/U delta={o_mean:+.4f}  CI=[{o_lo:+.4f}, {o_hi:+.4f}]  {o_verdict}")


if __name__ == "__main__":
    main()

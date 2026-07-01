"""15_per_team_drag_calibration.py — Per-team y-drag from observed intensity.

Anisotropic drag `(0.00, 0.16)` treats every match the same. But
intuitively, high-intensity teams (many goals, high pressing) live in
a regime where y-momentum should decay faster; low-intensity teams
live in a regime where the body drifts toward draw and heavy y-drag
just accelerates the pull.

We fit one `C_d_y` per team from the FIRST half of the season and
apply it to the SECOND half. Fitting rule:

  intensity_team = average (goals_scored + goals_conceded) per match
                   in the training half
  C_d_y(team)    = interpolate(intensity, breakpoints, drag_values)

with `breakpoints = [1.5, 2.5, 3.5]` goals/match and drag values
`[0.32, 0.16, 0.08]` (lower drag for higher-intensity teams).
A match uses `C_d_y = 0.5 * (C_d_y(home) + C_d_y(away))`.

Train/test split is temporal — no leakage. Comparison config uses the
constant (0.00, 0.16) on the same test half.

Run with::

    PYTHONPATH=src python3 experiments/15_per_team_drag_calibration.py
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

spec2 = importlib.util.spec_from_file_location(
    "e14", ROOT / "experiments" / "14_momentum_ic_features.py")
e14 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(e14)

e05.N_TRIALS = 30
SEED = 20260701

BREAKPOINTS = np.array([1.5, 2.5, 3.5])
DRAG_VALUES = np.array([0.32, 0.16, 0.08])
BASELINE_CD = np.array([0.00, 0.16])


def team_intensity(matches: list) -> dict:
    """Average (goals_scored + goals_conceded) per team over ``matches``."""
    totals = {}
    counts = {}
    for m in matches:
        gh = m["actual_goals_home"]
        ga = m["actual_goals_away"]
        for team, gs, gc in [(m["home_team"], gh, ga),
                              (m["away_team"], ga, gh)]:
            totals[team] = totals.get(team, 0) + gs + gc
            counts[team] = counts.get(team, 0) + 1
    return {t: totals[t] / counts[t] for t in totals}


def fit_drag_per_team(intensities: dict) -> dict:
    out = {}
    for team, intensity in intensities.items():
        cy = float(np.interp(intensity, BREAKPOINTS, DRAG_VALUES))
        out[team] = cy
    return out


def score(matches, cd_for_match) -> tuple:
    """cd_for_match(match) -> C_d 2-vector."""
    hda = []
    ou = []
    for i, m in enumerate(matches, 1):
        wells = e05.multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"],
            m["p_over"], m["p_under"],
        )
        space = EventSpace(wells)
        cd = cd_for_match(m)
        rng = np.random.default_rng(seed=e05.SEED)
        q_scale = np.array([0.3, 0.2]) * e05.IC_SCALE
        p_scale = np.array([0.15, 0.15]) * e05.IC_SCALE
        f_pos, f_mass, _ = e05._stack(space)
        o_pos, o_mass, o_labels = e05._stack(space)
        acc = {l: 0.0 for l in o_labels}
        for _ in range(e05.N_TRIALS):
            q0 = rng.normal(scale=q_scale)
            p0 = rng.normal(scale=p_scale)
            q_end = e05.fast_final_q(f_pos, f_mass, q0, p0,
                                     duration=e05.DURATION, dt=e05.DT, C_d=cd)
            probs = e05.fast_posterior(q_end, o_pos, o_mass, o_labels,
                                       alpha=e05.SOFT_ALPHA)
            for label, p in probs.items():
                acc[label] += p
        total = sum(acc.values())
        joint = {l: v / total for l, v in acc.items()}
        hda_post = e05.marginal_hda(joint, m["home_label"], m["away_label"])
        ou_post = e05.marginal_ou(joint)
        hda.append(e05.brier(hda_post, m["actual_hda"]))
        ou.append(e05.brier(ou_post, m["actual_ou"]))
        if i % 50 == 0:
            print(f"  [{i:>3d}/{len(matches)}]  "
                  f"HDA={np.mean(hda):.4f}  O/U={np.mean(ou):.4f}",
                  flush=True)
    return np.array(hda), np.array(ou)


def bootstrap_ci(deltas, n_boot=2000):
    rng = np.random.default_rng(SEED)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(deltas, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def main() -> None:
    matches = e14.load_full_season_with_goals()
    n = len(matches)
    split = n // 2
    train, test = matches[:split], matches[split:]

    print("=== Orbita: per-team y-drag calibration (train/test split) ===")
    print(f"Panel : {n} matches, train={len(train)}, test={len(test)}")
    print(f"Trials: {e05.N_TRIALS}/match")
    print(f"Breakpoints (goals): {BREAKPOINTS.tolist()}")
    print(f"Drag values (C_d_y): {DRAG_VALUES.tolist()}")
    print()

    intensities = team_intensity(train)
    per_team_cd_y = fit_drag_per_team(intensities)
    print("--- Fitted per-team C_d_y (top 5, bottom 5 by intensity) ---")
    ranked = sorted(intensities.items(), key=lambda kv: -kv[1])
    for team, intensity in ranked[:5]:
        print(f"  {team:<20s} intensity={intensity:.2f}  "
              f"C_d_y={per_team_cd_y[team]:.3f}")
    print("  ...")
    for team, intensity in ranked[-5:]:
        print(f"  {team:<20s} intensity={intensity:.2f}  "
              f"C_d_y={per_team_cd_y[team]:.3f}")
    print()

    mkt_hda = []
    mkt_ou = []
    for m in test:
        p_mkt_hda = {m["home_label"]: m["p_h"], "draw": m["p_d"],
                     m["away_label"]: m["p_a"]}
        p_mkt_ou = {"over": m["p_over"], "under": m["p_under"]}
        mkt_hda.append(e05.brier(p_mkt_hda, m["actual_hda"]))
        mkt_ou.append(e05.brier(p_mkt_ou, m["actual_ou"]))
    mkt_hda = np.array(mkt_hda)
    mkt_ou = np.array(mkt_ou)
    print(f"Market (test): HDA={mkt_hda.mean():.4f}  O/U={mkt_ou.mean():.4f}")
    print()

    def cd_baseline(_m): return BASELINE_CD
    def cd_per_team(m):
        cy = 0.5 * (per_team_cd_y.get(m["home_team"], 0.16) +
                    per_team_cd_y.get(m["away_team"], 0.16))
        return np.array([0.0, cy])

    print("--- Scoring: constant (0.00, 0.16) on test ---", flush=True)
    hda_b, ou_b = score(test, cd_baseline)
    print(f"  engine HDA={hda_b.mean():.4f}  O/U={ou_b.mean():.4f}")
    print()
    print("--- Scoring: per-team fitted drag on test ---", flush=True)
    hda_c, ou_c = score(test, cd_per_team)
    print(f"  engine HDA={hda_c.mean():.4f}  O/U={ou_c.mean():.4f}")
    print()

    print("=== summary + bootstrap 90% CIs (engine − market, TEST) ===")
    for name, (hda, ou) in [("constant baseline", (hda_b, ou_b)),
                              ("per-team calibrated", (hda_c, ou_c))]:
        d_hda = hda - mkt_hda
        d_ou = ou - mkt_ou
        h_m, h_lo, h_hi = bootstrap_ci(d_hda)
        o_m, o_lo, o_hi = bootstrap_ci(d_ou)
        h_v = ("engine BEATS" if h_hi < 0 else
               "market BEATS" if h_lo > 0 else "tied (CI ∋ 0)")
        o_v = ("engine BEATS" if o_hi < 0 else
               "market BEATS" if o_lo > 0 else "tied (CI ∋ 0)")
        print(f"{name}")
        print(f"  HDA d={h_m:+.4f}  CI=[{h_lo:+.4f}, {h_hi:+.4f}]  {h_v}")
        print(f"  O/U d={o_m:+.4f}  CI=[{o_lo:+.4f}, {o_hi:+.4f}]  {o_v}")


if __name__ == "__main__":
    main()

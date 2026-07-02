"""18_joint_3d_correlation.py — Does 3D joint outperform 2D-independent?

The v0.3.6 n-D event space lets us build a 12-well 3D H/D/A × O/U ×
BTTS joint space. The 2D-independent baseline (exp 05 Config B) uses
a 6-well H/D/A × O/U space and treats BTTS as a separate market
(here, priced from a Poisson-independence null derived from O/U).

If real match outcomes have correlation the market doesn't fully
capture — e.g., high-scoring matches co-occur with BTTS at higher
than the Poisson rate — the 3D joint's H/D/A and O/U marginals should
score BETTER Brier than the 2D-independent posteriors on the same
380 matches.

Positive result would be a genuinely NEW way to beat the market: not
by adding data, but by encoding cross-market correlation in the
geometry of the event space.

Football-data.co.uk ships FTHG/FTAG (so BTTS actuals are derivable)
but no B365 BTTS closing odds. BTTS "market baseline" here is
constructed from O/U 2.5 via the Poisson-independence assumption.

Run with::

    PYTHONPATH=src python3 experiments/18_joint_3d_correlation.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import Attractor, EventSpace  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e05", ROOT / "experiments" / "05_player_attractor_panel.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)

spec2 = importlib.util.spec_from_file_location(
    "e14", ROOT / "experiments" / "14_momentum_ic_features.py")
e14 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(e14)

e05.N_TRIALS = 30
SEED = 20260702
BEST_CD_2D = np.array([0.00, 0.16])
# Third axis (BTTS): no evidence yet on best drag; start with 0.10
# in the middle of the range that mattered in exp 10.
BEST_CD_3D = np.array([0.00, 0.16, 0.10])

WIN_X = 5.0
OVER_Y = 3.0
UNDER_Y = -3.0
BTTS_Z = 2.0

CACHE_16 = ROOT / "experiments" / "_cache_16_posteriors.csv"
CACHE_18 = ROOT / "experiments" / "_cache_18_3d_posteriors.csv"


def build_3d_space(m: dict, p_btts_prior: float) -> EventSpace:
    """12-well 3D H/D/A × O/U × BTTS. Well mass = joint prior product."""
    wells = []
    for x, outcome_label, p_match in [
        (WIN_X,  m["home_label"], m["p_h"]),
        (0.0,    "draw",          m["p_d"]),
        (-WIN_X, m["away_label"], m["p_a"]),
    ]:
        for y, ou_side, p_ou in [(OVER_Y, "over", m["p_over"]),
                                   (UNDER_Y, "under", m["p_under"])]:
            for z, bt_side, p_bt in [(BTTS_Z, "btts", p_btts_prior),
                                       (-BTTS_Z, "no_btts", 1.0 - p_btts_prior)]:
                mass = p_match * p_ou * p_bt
                wells.append(Attractor(
                    label=f"{outcome_label}|{ou_side}|{bt_side}",
                    position=[x, y, z],
                    mass=mass,
                ))
    return EventSpace(wells)


def poisson_btts_from_ou(p_over: float) -> float:
    """Estimate P(BTTS) from P(over 2.5) under Poisson independence.

    For expected total goals λ (implied from p_over), assume home/away
    each scores Poisson(λ/2). Then P(BTTS) = (1 - exp(-λ/2))^2.

    We invert p_over -> λ numerically via a table.
    """
    lams = np.linspace(0.5, 6.0, 100)
    p_overs = 1.0 - np.exp(-lams) * (1 + lams + lams**2 / 2)
    lam = float(np.interp(p_over, p_overs, lams))
    half = lam / 2.0
    return float((1.0 - np.exp(-half)) ** 2)


def score_3d(matches: list, out_path: Path) -> list:
    """Run 3D joint 12-well posterior on full season."""
    rows = []
    for i, m in enumerate(matches, 1):
        p_btts_prior = poisson_btts_from_ou(m["p_over"])
        space = build_3d_space(m, p_btts_prior)
        rng = np.random.default_rng(seed=e05.SEED)
        q_scale = np.array([0.3, 0.2, 0.15]) * e05.IC_SCALE
        p_scale = np.array([0.15, 0.15, 0.10]) * e05.IC_SCALE
        f_pos, f_mass, _ = e05._stack(space)
        o_pos, o_mass, o_labels = e05._stack(space)
        acc = {l: 0.0 for l in o_labels}
        for _ in range(e05.N_TRIALS):
            q0 = rng.normal(scale=q_scale)
            p0 = rng.normal(scale=p_scale)
            q_end = e05.fast_final_q(f_pos, f_mass, q0, p0,
                                     duration=e05.DURATION, dt=e05.DT,
                                     C_d=BEST_CD_3D)
            probs = e05.fast_posterior(q_end, o_pos, o_mass, o_labels,
                                       alpha=e05.SOFT_ALPHA)
            for label, p in probs.items():
                acc[label] += p
        total = sum(acc.values())
        joint = {l: v / total for l, v in acc.items()}

        # Marginals from 12-well joint
        hda_marg = {"home": 0.0, "draw": 0.0, "away": 0.0}
        ou_marg = {"over": 0.0, "under": 0.0}
        bt_marg = {"btts": 0.0, "no_btts": 0.0}
        for label, p in joint.items():
            outcome, ou_side, bt_side = label.split("|")
            if outcome == m["home_label"]:
                hda_marg["home"] += p
            elif outcome == m["away_label"]:
                hda_marg["away"] += p
            else:
                hda_marg["draw"] += p
            ou_marg[ou_side] += p
            bt_marg[bt_side] += p

        actual_hda_label = m["actual_hda"]
        actual_hda_side = ("home" if actual_hda_label == m["home_label"]
                           else "away" if actual_hda_label == m["away_label"]
                           else "draw")
        actual_btts = ("btts" if (m["actual_goals_home"] > 0
                                    and m["actual_goals_away"] > 0)
                        else "no_btts")
        rows.append({
            "match_id": i,
            "date": m["date"],
            "actual_hda": actual_hda_side,
            "actual_ou": m["actual_ou"],
            "actual_btts": actual_btts,
            "eng3d_h": hda_marg["home"],
            "eng3d_d": hda_marg["draw"],
            "eng3d_a": hda_marg["away"],
            "eng3d_o": ou_marg["over"],
            "eng3d_u": ou_marg["under"],
            "eng3d_btts": bt_marg["btts"],
            "eng3d_no_btts": bt_marg["no_btts"],
            "prior_btts": p_btts_prior,
        })
        if i % 50 == 0:
            print(f"  [{i:>3d}/{len(matches)}] scored", flush=True)
    fields = list(rows[0].keys())
    with out_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved 3D posteriors -> {out_path}", flush=True)
    return rows


def brier(p_dict: dict, y_label: str) -> float:
    return sum((v - (1.0 if k == y_label else 0.0)) ** 2
               for k, v in p_dict.items())


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
    print("=== Orbita: 3D joint vs 2D-independent Brier ===")
    print(f"Panel : {len(matches)} matches (full 2024/25 EPL)")
    print(f"Trials: {e05.N_TRIALS}/match")
    print(f"Drag  : 3D = {BEST_CD_3D}  (Cd_z=0.10 as prior guess)")
    print()

    if CACHE_18.exists():
        print(f"Loading 3D cache from {CACHE_18}")
        with CACHE_18.open() as fh:
            rows_3d = list(csv.DictReader(fh))
        for r in rows_3d:
            for k in list(r.keys()):
                if k in ("match_id", "date", "actual_hda",
                          "actual_ou", "actual_btts"):
                    continue
                r[k] = float(r[k])
            r["match_id"] = int(r["match_id"])
    else:
        print("--- Running 3D joint 12-well posterior ---", flush=True)
        rows_3d = score_3d(matches, CACHE_18)

    if not CACHE_16.exists():
        raise SystemExit(
            f"Missing 2D cache at {CACHE_16}. Run experiment 16 first.")
    with CACHE_16.open() as fh:
        rows_2d = list(csv.DictReader(fh))
    for r in rows_2d:
        for k in list(r.keys()):
            if k in ("match_id", "date", "event",
                      "actual_hda", "actual_ou"):
                continue
            r[k] = float(r[k])
        r["match_id"] = int(r["match_id"])

    # Align on match_id.
    d2d = {r["match_id"]: r for r in rows_2d}
    aligned = [(r3d, d2d[r3d["match_id"]]) for r3d in rows_3d
                if r3d["match_id"] in d2d]
    print(f"Aligned {len(aligned)} matches across 2D and 3D caches.")
    print()

    # Per-match Brier deltas: 3D marginal − 2D marginal, on H/D/A and O/U.
    d_hda = []
    d_ou = []
    for r3d, r2d in aligned:
        hda_3d = {"home": r3d["eng3d_h"], "draw": r3d["eng3d_d"],
                   "away": r3d["eng3d_a"]}
        hda_2d = {"home": r2d["eng_h"], "draw": r2d["eng_d"],
                   "away": r2d["eng_a"]}
        ou_3d = {"over": r3d["eng3d_o"], "under": r3d["eng3d_u"]}
        ou_2d = {"over": r2d["eng_o"], "under": r2d["eng_u"]}
        d_hda.append(brier(hda_3d, r3d["actual_hda"]) -
                     brier(hda_2d, r2d["actual_hda"]))
        d_ou.append(brier(ou_3d, r3d["actual_ou"]) -
                    brier(ou_2d, r2d["actual_ou"]))

    d_hda = np.array(d_hda)
    d_ou = np.array(d_ou)
    print("=== 3D joint vs 2D-independent Brier deltas (bootstrap 90% CI) ===")
    for name, deltas in [("HDA (3D - 2D)", d_hda), ("O/U (3D - 2D)", d_ou)]:
        m, lo, hi = bootstrap_ci(deltas)
        v = ("3D BEATS 2D" if hi < 0 else
             "2D BEATS 3D" if lo > 0 else "tied (CI ∋ 0)")
        print(f"  {name:<18s}  d={m:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {v}")
    print()

    # BTTS: 3D marginal vs Poisson-independence null.
    d_btts = []
    for r3d, _ in aligned:
        bt_3d = {"btts": r3d["eng3d_btts"], "no_btts": r3d["eng3d_no_btts"]}
        bt_null = {"btts": r3d["prior_btts"],
                    "no_btts": 1.0 - r3d["prior_btts"]}
        d_btts.append(brier(bt_3d, r3d["actual_btts"]) -
                      brier(bt_null, r3d["actual_btts"]))
    d_btts = np.array(d_btts)
    m, lo, hi = bootstrap_ci(d_btts)
    v = ("3D BEATS Poisson null" if hi < 0 else
         "Poisson null BEATS 3D" if lo > 0 else "tied (CI ∋ 0)")
    print("=== BTTS 3D marginal vs Poisson-independence null ===")
    print(f"  d={m:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {v}")
    print()

    # Absolute Brier means (sanity).
    print("=== Absolute Brier means ===")
    b_hda_3d = np.mean([brier({"home": r3d["eng3d_h"], "draw": r3d["eng3d_d"],
                                 "away": r3d["eng3d_a"]}, r3d["actual_hda"])
                        for r3d, _ in aligned])
    b_hda_2d = np.mean([brier({"home": r2d["eng_h"], "draw": r2d["eng_d"],
                                 "away": r2d["eng_a"]}, r2d["actual_hda"])
                        for _, r2d in aligned])
    b_ou_3d = np.mean([brier({"over": r3d["eng3d_o"], "under": r3d["eng3d_u"]},
                              r3d["actual_ou"])
                       for r3d, _ in aligned])
    b_ou_2d = np.mean([brier({"over": r2d["eng_o"], "under": r2d["eng_u"]},
                              r2d["actual_ou"])
                       for _, r2d in aligned])
    b_btts_3d = np.mean([brier({"btts": r3d["eng3d_btts"],
                                  "no_btts": r3d["eng3d_no_btts"]},
                                r3d["actual_btts"])
                         for r3d, _ in aligned])
    b_btts_null = np.mean([brier({"btts": r3d["prior_btts"],
                                    "no_btts": 1.0 - r3d["prior_btts"]},
                                  r3d["actual_btts"])
                           for r3d, _ in aligned])
    print(f"HDA  3D={b_hda_3d:.4f}  2D={b_hda_2d:.4f}")
    print(f"O/U  3D={b_ou_3d:.4f}  2D={b_ou_2d:.4f}")
    print(f"BTTS 3D={b_btts_3d:.4f}  null={b_btts_null:.4f}")


if __name__ == "__main__":
    main()

"""08_full_season_panel.py — Scale the panel to a full EPL season.

The 50-match (Oct–Nov 2024) panel from experiment 07 found Config B's
Goals O/U Brier statistically tied with the de-vigged Bet365 line —
point estimate −0.006 in the engine's favour, 90% bootstrap CI
[−0.036, +0.024] including zero. The natural next question is whether
that tie expands, evaporates, or stays neutral as the sample grows.

This run uses every football-data.co.uk 2024/25 EPL fixture (380
matches) and three configurations:

  Config A : 3 H/D/L wells, baseline (no O/U, no players).
  Config B : 6 multi-market joint wells (H/D/L × O/U 2.5), no players.
  Config C : Config B + 22 player attractors per match (Elo-seeded
             synthetic lineups, deterministic via _stable_seed).

With 380 matches each and N=50 trials per match, total runtime should
stay under ~45 min on a single CPU. Bootstrap 90% CIs are reported on
both H/D/L and O/U deltas; if the O/U CI strictly excludes zero on
this panel that's the headline finding.

Run with::

    PYTHONPATH=src python3 experiments/08_full_season_panel.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import datetime
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

# Match e07 settings.
N_TRIALS = 30
e05.N_TRIALS = N_TRIALS
# Config C (player attractors) is the expensive path — 22 extra attractors
# per match × 380 matches puts a full run past 2 hours. Toggle off to keep
# this experiment focused on the Config B (multi-market) O/U signal that
# was the headline question. The smaller panel in e05 already covers C.
INCLUDE_CONFIG_C = False


def load_full_season() -> list:
    """Like e05.load_matches_with_ou but with no date filter — the
    entire CSV is consumed."""
    csv_path = e03.fetch_csv()
    out = []
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                date = e03.parse_date(row["Date"])
                h = float(row["B365H"])
                d = float(row["B365D"])
                a = float(row["B365A"])
                o = float(row["B365>2.5"])
                u = float(row["B365<2.5"])
                fthg = int(row["FTHG"])
                ftag = int(row["FTAG"])
            except (KeyError, ValueError):
                continue
            ph, pd, pa = e03.devig(h, d, a)
            inv_o, inv_u = 1.0 / o, 1.0 / u
            s_ou = inv_o + inv_u
            po, pu = inv_o / s_ou, inv_u / s_ou
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            home_label = f"{e03._slug(home)}_win"
            away_label = f"{e03._slug(away)}_win"
            ftr = row["FTR"]
            actual_hda = {"H": home_label, "D": "draw", "A": away_label}[ftr]
            actual_ou = "over" if (fthg + ftag) > 2.5 else "under"
            out.append({
                "date": date.strftime("%Y-%m-%d"),
                "home_team": home, "away_team": away,
                "home_label": home_label, "away_label": away_label,
                "p_h": ph, "p_d": pd, "p_a": pa,
                "p_over": po, "p_under": pu,
                "actual_hda": actual_hda, "actual_ou": actual_ou,
            })
    out.sort(key=lambda m: m["date"])
    return out


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000, seed: int = 20260629):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(deltas, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def main() -> None:
    matches = load_full_season()
    print("=== Orbita: full-season EPL backtest ===")
    print(f"Season    : 2024/25 EPL")
    print(f"Panel size: {len(matches)} matches")
    print(f"Window    : {matches[0]['date']} → {matches[-1]['date']}")
    print(f"Trials    : {N_TRIALS}/match (seed = {e05.SEED})")
    print()

    elo_cache: dict = {}
    rows_a_hda = []
    rows_b_hda = []
    rows_c_hda = []
    rows_b_ou = []
    rows_c_ou = []
    rows_mkt_hda = []
    rows_mkt_ou = []
    hits_mkt_hda = hits_mkt_ou = 0
    hits_a_hda = hits_b_hda = hits_c_hda = 0
    hits_b_ou = hits_c_ou = 0

    for i, m in enumerate(matches, 1):
        wells_b = e05.baseline_wells(m["home_label"], m["away_label"],
                                      m["p_h"], m["p_d"], m["p_a"])
        wells_m = e05.multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"], m["p_over"], m["p_under"])
        space_b = EventSpace(wells_b)
        space_m = EventSpace(wells_m)

        mkt_hda = {m["home_label"]: m["p_h"],
                   "draw": m["p_d"],
                   m["away_label"]: m["p_a"]}
        mkt_ou = {"over": m["p_over"], "under": m["p_under"]}
        rows_mkt_hda.append(e05.brier(mkt_hda, m["actual_hda"]))
        rows_mkt_ou.append(e05.brier(mkt_ou, m["actual_ou"]))
        if e05.modal(mkt_hda) == m["actual_hda"]: hits_mkt_hda += 1
        if e05.modal(mkt_ou) == m["actual_ou"]: hits_mkt_ou += 1

        # Config A
        probs_a = e05.run(space_b, space_b)
        rows_a_hda.append(e05.brier(probs_a, m["actual_hda"]))
        if e05.modal(probs_a) == m["actual_hda"]: hits_a_hda += 1

        # Config B
        joint_b = e05.run(space_m, space_m)
        hda_b = e05.marginal_hda(joint_b, m["home_label"], m["away_label"])
        ou_b = e05.marginal_ou(joint_b)
        rows_b_hda.append(e05.brier(hda_b, m["actual_hda"]))
        rows_b_ou.append(e05.brier(ou_b, m["actual_ou"]))
        if e05.modal(hda_b) == m["actual_hda"]: hits_b_hda += 1
        if e05.modal(ou_b) == m["actual_ou"]: hits_b_ou += 1

        # Config C — skipped when INCLUDE_CONFIG_C is False.
        if INCLUDE_CONFIG_C:
            r_h = e03.elo_for(m["home_team"], m["date"], elo_cache)
            r_a = e03.elo_for(m["away_team"], m["date"], elo_cache)
            home_xi = e05.synth_lineup(m["home_team"], +1, r_h)
            away_xi = e05.synth_lineup(m["away_team"], -1, r_a)
            players = (e05.player_attractors(home_xi, m["home_label"]) +
                       e05.player_attractors(away_xi, m["away_label"]))
            force_space = EventSpace(list(space_m.attractors) + players)
            joint_c = e05.run(force_space, space_m)
            hda_c = e05.marginal_hda(joint_c, m["home_label"], m["away_label"])
            ou_c = e05.marginal_ou(joint_c)
            rows_c_hda.append(e05.brier(hda_c, m["actual_hda"]))
            rows_c_ou.append(e05.brier(ou_c, m["actual_ou"]))
            if e05.modal(hda_c) == m["actual_hda"]: hits_c_hda += 1
            if e05.modal(ou_c) == m["actual_ou"]: hits_c_ou += 1

        if i % 25 == 0 or i == len(matches):
            extra = ""
            if INCLUDE_CONFIG_C and rows_c_hda:
                extra = f" C_hda={np.mean(rows_c_hda):.4f}"
            print(f"  [{i:>3d}/{len(matches)}] "
                  f"mkt_hda={np.mean(rows_mkt_hda):.4f} "
                  f"B_hda={np.mean(rows_b_hda):.4f}{extra} "
                  f"mkt_ou={np.mean(rows_mkt_ou):.4f} "
                  f"B_ou={np.mean(rows_b_ou):.4f}",
                  flush=True)

    n = len(matches)
    rows_mkt_hda = np.array(rows_mkt_hda)
    rows_mkt_ou = np.array(rows_mkt_ou)
    rows_a_hda = np.array(rows_a_hda)
    rows_b_hda = np.array(rows_b_hda)
    rows_c_hda = np.array(rows_c_hda) if rows_c_hda else np.array([])
    rows_b_ou = np.array(rows_b_ou)
    rows_c_ou = np.array(rows_c_ou) if rows_c_ou else np.array([])

    print()
    print("=== aggregate (mean Brier) ===")
    print(f"{'config':<24s}  {'HDA Brier':>10s}  {'HDA hit':>8s}  "
          f"{'O/U Brier':>10s}  {'O/U hit':>8s}")
    print(f"{'bookmaker':<24s}  {rows_mkt_hda.mean():>10.4f}  "
          f"{hits_mkt_hda/n:>7.1%}  {rows_mkt_ou.mean():>10.4f}  "
          f"{hits_mkt_ou/n:>7.1%}")
    print(f"{'A_baseline':<24s}  {rows_a_hda.mean():>10.4f}  "
          f"{hits_a_hda/n:>7.1%}  {'—':>10s}  {'—':>8s}")
    print(f"{'B_multimarket':<24s}  {rows_b_hda.mean():>10.4f}  "
          f"{hits_b_hda/n:>7.1%}  {rows_b_ou.mean():>10.4f}  "
          f"{hits_b_ou/n:>7.1%}")
    if INCLUDE_CONFIG_C and len(rows_c_hda):
        print(f"{'C_multi_players':<24s}  {rows_c_hda.mean():>10.4f}  "
              f"{hits_c_hda/n:>7.1%}  {rows_c_ou.mean():>10.4f}  "
              f"{hits_c_ou/n:>7.1%}")

    print()
    print("=== bootstrap 90% CI on per-match deltas (engine − market) ===")
    configs = [
        ("A_baseline", rows_a_hda, None),
        ("B_multimarket", rows_b_hda, rows_b_ou),
    ]
    if INCLUDE_CONFIG_C and len(rows_c_hda):
        configs.append(("C_multi_players", rows_c_hda, rows_c_ou))
    for name, eng_hda, eng_ou in configs:
        mean, lo, hi = bootstrap_ci(eng_hda - rows_mkt_hda)
        verdict = ("engine BEATS" if hi < 0 else
                   "market BEATS" if lo > 0 else "tied (CI ∋ 0)")
        print(f"{name:<24s}  HDA delta={mean:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
              f"{verdict}")
        if eng_ou is not None:
            mean, lo, hi = bootstrap_ci(eng_ou - rows_mkt_ou)
            verdict = ("engine BEATS" if hi < 0 else
                       "market BEATS" if lo > 0 else "tied (CI ∋ 0)")
            print(f"{'':<24s}  O/U delta={mean:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
                  f"{verdict}")


if __name__ == "__main__":
    main()

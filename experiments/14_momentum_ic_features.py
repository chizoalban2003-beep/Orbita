"""14_momentum_ic_features.py — Bias p0 with pre-match features.

Current `p0` is drawn from `N(0, p_scale)` — a symmetric prior that
carries no information about the specific matchup. Any directional
signal in the input has to come through the wells' mass distribution.

Physical hypothesis: the initial momentum vector is the natural place
to encode pre-match state that ISN'T already in the closing line's
implied probability. Two cheap, orthogonal-ish signals from
football-data.co.uk:

  * Rest-days differential (home_rest − away_rest). Teams with more
    rest have more thrust available in the win-direction.
  * Recent goal differential over last K matches (rolling form).
    A hot-scoring side has more x-momentum toward its own win well;
    a leaky-defence pair has more y-momentum toward the "over" well.

We bias p0 with these two signals and compare Brier vs the constant
anisotropic baseline (0.00, 0.16) on the full 380-match season. If
these features aren't already priced in, the biased-p0 configuration
beats the unbiased one.

Feature scaling `LAMBDA_X, LAMBDA_Y` set so the median-magnitude bias
is on the same order as `p_scale` — bias meaningfully perturbs the
prior without swamping it.

Run with::

    PYTHONPATH=src python3 experiments/14_momentum_ic_features.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict, deque
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

spec2 = importlib.util.spec_from_file_location(
    "e08", ROOT / "experiments" / "08_full_season_panel.py")
e08 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(e08)

e05.N_TRIALS = 30
SEED = 20260701
K = 5

# Bias coefficients — scaled so median absolute bias ≈ p_scale.
# p_scale = 0.15 * IC_SCALE = 0.375 by default.
LAMBDA_X = 0.05   # per (goal_diff_home - goal_diff_away) unit
LAMBDA_Y = 0.02   # per (avg total goals L5) unit

BEST_CD_OU = np.array([0.00, 0.16])


def enrich_with_features(matches: list) -> list:
    """Walk matches chronologically, attach rolling-form + rest-days
    features to each match dict. Feature columns:
      - rest_home, rest_away (days since previous match; None early)
      - form_home, form_away (goals_for - goals_against over last K
        matches; 0 if no history)
      - goals_recent_home, goals_recent_away (avg goals scored L5)
    """
    matches = sorted(matches, key=lambda m: m["date"])
    last_date: dict = {}
    goal_hist: dict = defaultdict(lambda: deque(maxlen=K))     # (scored, conceded)
    for m in matches:
        h, a = m["home_team"], m["away_team"]
        m_date = datetime.strptime(m["date"], "%Y-%m-%d")
        m["rest_home"] = (m_date - last_date[h]).days if h in last_date else 7
        m["rest_away"] = (m_date - last_date[a]).days if a in last_date else 7

        def rolling(team):
            hist = goal_hist[team]
            if not hist:
                return 0.0, 0.0
            gs = np.mean([x[0] for x in hist])
            gc = np.mean([x[1] for x in hist])
            return float(gs), float(gc)
        gs_h, gc_h = rolling(h)
        gs_a, gc_a = rolling(a)
        m["form_home"] = gs_h - gc_h
        m["form_away"] = gs_a - gc_a
        m["goals_recent_home"] = gs_h
        m["goals_recent_away"] = gs_a
    # Second pass: update history AFTER all rows have their pre-match
    # features. Since we walk chronologically we can update inline as we go.
    goal_hist.clear()
    last_date.clear()
    for m in matches:
        h, a = m["home_team"], m["away_team"]
        m_date = datetime.strptime(m["date"], "%Y-%m-%d")
        # (Re-compute features using the fresh state.)
        m["rest_home"] = (m_date - last_date[h]).days if h in last_date else 7
        m["rest_away"] = (m_date - last_date[a]).days if a in last_date else 7

        def rolling(team):
            hist = goal_hist[team]
            if not hist:
                return 0.0, 0.0
            gs = np.mean([x[0] for x in hist])
            gc = np.mean([x[1] for x in hist])
            return float(gs), float(gc)
        gs_h, gc_h = rolling(h)
        gs_a, gc_a = rolling(a)
        m["form_home"] = gs_h - gc_h
        m["form_away"] = gs_a - gc_a
        m["goals_recent_home"] = gs_h
        m["goals_recent_away"] = gs_a

        goals_h, goals_a = m["actual_goals_home"], m["actual_goals_away"]
        goal_hist[h].append((goals_h, goals_a))
        goal_hist[a].append((goals_a, goals_h))
        last_date[h] = m_date
        last_date[a] = m_date
    return matches


def load_full_season_with_goals() -> list:
    """Full season enriched with actual FT scores so we can build history."""
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
                "event": f"EPL {home} vs {away}",
                "home_team": home, "away_team": away,
                "home_label": home_label, "away_label": away_label,
                "p_h": ph, "p_d": pd, "p_a": pa,
                "p_over": po, "p_under": pu,
                "actual_hda": actual_hda, "actual_ou": actual_ou,
                "actual_goals_home": fthg, "actual_goals_away": ftag,
            })
    out.sort(key=lambda m: m["date"])
    return out


def run_biased(space, m, C_d, use_bias: bool) -> dict:
    """Same as e05.run() but with an optional p0 bias per match."""
    rng = np.random.default_rng(seed=e05.SEED)
    q_scale = np.array([0.3, 0.2]) * e05.IC_SCALE
    p_scale = np.array([0.15, 0.15]) * e05.IC_SCALE
    if use_bias:
        rest_adv = m["rest_home"] - m["rest_away"]
        form_adv = m["form_home"] - m["form_away"]
        goal_intensity = m["goals_recent_home"] + m["goals_recent_away"]
        p_bias_x = LAMBDA_X * (form_adv + 0.2 * rest_adv)
        p_bias_y = LAMBDA_Y * (goal_intensity - 2.7)
        p_bias = np.array([p_bias_x, p_bias_y])
    else:
        p_bias = np.zeros(2)
    f_pos, f_mass, _ = e05._stack(space)
    o_pos, o_mass, o_labels = e05._stack(space)
    acc = {l: 0.0 for l in o_labels}
    for _ in range(e05.N_TRIALS):
        q0 = rng.normal(scale=q_scale)
        p0 = rng.normal(scale=p_scale) + p_bias
        q_end = e05.fast_final_q(f_pos, f_mass, q0, p0,
                                 duration=e05.DURATION, dt=e05.DT, C_d=C_d)
        probs = e05.fast_posterior(q_end, o_pos, o_mass, o_labels,
                                   alpha=e05.SOFT_ALPHA)
        for label, p in probs.items():
            acc[label] += p
    total = sum(acc.values())
    return {label: v / total for label, v in acc.items()}


def score(matches, use_bias: bool) -> tuple:
    hda = []
    ou = []
    for i, m in enumerate(matches, 1):
        wells = e05.multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"],
            m["p_over"], m["p_under"],
        )
        space = EventSpace(wells)
        joint = run_biased(space, m, C_d=BEST_CD_OU, use_bias=use_bias)
        hda_post = e05.marginal_hda(joint, m["home_label"], m["away_label"])
        ou_post = e05.marginal_ou(joint)
        hda.append(e05.brier(hda_post, m["actual_hda"]))
        ou.append(e05.brier(ou_post, m["actual_ou"]))
        if i % 100 == 0:
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
    matches = load_full_season_with_goals()
    matches = enrich_with_features(matches)
    # Skip the first ~50 matches: no history means bias is zero for them anyway
    # (form=0, rest=default 7) so leaving them in isn't a leak — but the bias
    # signal is only meaningful after teams have a few matches of history.
    print("=== Orbita: momentum-IC upgrade from pre-match features ===")
    print(f"Panel : {len(matches)} matches (full 2024/25 EPL)")
    print(f"Trials: {e05.N_TRIALS}/match  | drag = {BEST_CD_OU} (anisotropic best)")
    print(f"K     : {K}-match rolling form window")
    print(f"Bias  : LAMBDA_X={LAMBDA_X} * (form_adv + 0.2*rest_adv), "
          f"LAMBDA_Y={LAMBDA_Y} * (goals_recent_total - 2.7)")
    print()

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
    print(f"Market: HDA={mkt_hda.mean():.4f}  O/U={mkt_ou.mean():.4f}")
    print()

    print("--- Scoring: unbiased p0 (anisotropic baseline) ---", flush=True)
    hda_u, ou_u = score(matches, use_bias=False)
    print(f"  engine HDA={hda_u.mean():.4f}  O/U={ou_u.mean():.4f}")
    print()
    print("--- Scoring: biased p0 (form + rest + goal intensity) ---",
          flush=True)
    hda_b, ou_b = score(matches, use_bias=True)
    print(f"  engine HDA={hda_b.mean():.4f}  O/U={ou_b.mean():.4f}")
    print()

    print("=== summary + bootstrap 90% CIs (engine − market) ===")
    for name, (hda, ou) in [("unbiased", (hda_u, ou_u)),
                              ("biased", (hda_b, ou_b))]:
        d_hda = hda - mkt_hda
        d_ou = ou - mkt_ou
        h_mean, h_lo, h_hi = bootstrap_ci(d_hda)
        o_mean, o_lo, o_hi = bootstrap_ci(d_ou)
        h_v = ("engine BEATS" if h_hi < 0 else
               "market BEATS" if h_lo > 0 else "tied (CI ∋ 0)")
        o_v = ("engine BEATS" if o_hi < 0 else
               "market BEATS" if o_lo > 0 else "tied (CI ∋ 0)")
        print(f"{name}")
        print(f"  HDA d={h_mean:+.4f}  CI=[{h_lo:+.4f}, {h_hi:+.4f}]  {h_v}")
        print(f"  O/U d={o_mean:+.4f}  CI=[{o_lo:+.4f}, {o_hi:+.4f}]  {o_v}")


if __name__ == "__main__":
    main()

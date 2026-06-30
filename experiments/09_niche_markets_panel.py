"""09_niche_markets_panel.py — does the physics engine add signal on
markets the bookmaker isn't tightly pricing?

football-data.co.uk doesn't publish corner/card/BTTS odds, so we can't
score against a market closing line. Instead we score against three
progressively-stronger statistical baselines:

  1. UNIFORM       — every outcome 50/50 (a sanity floor).
  2. LEAGUE        — Poisson with the season's league-mean λ (no
                     per-match information at all).
  3. TEAM-ROLLING  — Poisson with team-conditional λ from the last K
                     matches each team played. Rich baseline, free of
                     future leakage.

Then the engine: a 2-well event space (over/under) with masses primed
from the TEAM-ROLLING Poisson AND shifted by team-Elo asymmetry. If
the engine clears TEAM-ROLLING on Brier, the physics is adding
orthogonal signal. If it can't, derivative markets need richer
inputs (lineup data, in-play observations) before the engine can
contribute.

Markets covered:

  * total corners > 9.5
  * total cards   > 4.5 (yellows + reds)
  * BTTS (both teams to score)

Run with::

    PYTHONPATH=src python3 experiments/09_niche_markets_panel.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict, deque
from math import exp, factorial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import Attractor, EventSpace  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e05", ROOT / "experiments" / "05_player_attractor_panel.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)
e03 = e05.e03

ROLLING_K = 5
N_TRIALS = 30
SEED = 20260629
DT = 0.1
DURATION = 300.0
C_D = 0.04
IC_SCALE = 2.5
SOFT_ALPHA = 2.0

# Market thresholds.
CORNER_LINE = 9.5
CARD_LINE = 4.5


def load_full_season() -> list:
    csv_path = e03.fetch_csv()
    out = []
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                date = e03.parse_date(row["Date"])
                h = float(row["B365H"]); d = float(row["B365D"]); a = float(row["B365A"])
                hc = int(row["HC"]); ac = int(row["AC"])
                hy = int(row["HY"]); ay = int(row["AY"])
                hr = int(row["HR"]); ar = int(row["AR"])
                fthg = int(row["FTHG"]); ftag = int(row["FTAG"])
            except (KeyError, ValueError):
                continue
            ph, pd, pa = e03.devig(h, d, a)
            out.append({
                "date": date.strftime("%Y-%m-%d"),
                "home_team": row["HomeTeam"], "away_team": row["AwayTeam"],
                "p_h": ph, "p_d": pd, "p_a": pa,
                "actual_corners": hc + ac,
                "actual_cards": hy + ay + hr + ar,
                "actual_btts": int(fthg > 0 and ftag > 0),
                "fthg": fthg, "ftag": ftag,
                "home_corners": hc, "away_corners": ac,
                "home_cards_for": hy + hr, "away_cards_for": ay + ar,
            })
    out.sort(key=lambda m: m["date"])
    return out


def poisson_cdf(k: int, lam: float) -> float:
    """Cumulative Poisson up to and including k."""
    s = 0.0
    p = exp(-lam)  # P(X=0)
    s += p
    for i in range(1, k + 1):
        p *= lam / i
        s += p
    return min(s, 1.0)


def p_over(lam: float, line: float) -> float:
    """P(X > line) where X ~ Poisson(lam) and line is a x.5 half-integer."""
    return 1.0 - poisson_cdf(int(line), lam)


def rolling_lambdas(matches: list) -> list:
    """For each match compute (lam_corners_home, lam_corners_away,
    lam_cards_home, lam_cards_away, lam_goals_home, lam_goals_away)
    using ONLY matches that precede it. First K matches per team fall
    back to league-average λ."""
    league_corners_home = np.mean([m["home_corners"] for m in matches])
    league_corners_away = np.mean([m["away_corners"] for m in matches])
    league_cards_home = np.mean([m["home_cards_for"] for m in matches])
    league_cards_away = np.mean([m["away_cards_for"] for m in matches])
    league_goals_home = np.mean([m["fthg"] for m in matches])
    league_goals_away = np.mean([m["ftag"] for m in matches])

    by_team_home_corners: dict = defaultdict(lambda: deque(maxlen=ROLLING_K))
    by_team_away_corners: dict = defaultdict(lambda: deque(maxlen=ROLLING_K))
    by_team_home_cards: dict = defaultdict(lambda: deque(maxlen=ROLLING_K))
    by_team_away_cards: dict = defaultdict(lambda: deque(maxlen=ROLLING_K))
    by_team_home_goals: dict = defaultdict(lambda: deque(maxlen=ROLLING_K))
    by_team_away_goals: dict = defaultdict(lambda: deque(maxlen=ROLLING_K))

    lambdas = []
    for m in matches:
        h, a = m["home_team"], m["away_team"]

        def mean_or_fallback(buf, fallback):
            return float(np.mean(buf)) if len(buf) > 0 else fallback

        lam_c_h = mean_or_fallback(by_team_home_corners[h], league_corners_home)
        lam_c_a = mean_or_fallback(by_team_away_corners[a], league_corners_away)
        lam_y_h = mean_or_fallback(by_team_home_cards[h], league_cards_home)
        lam_y_a = mean_or_fallback(by_team_away_cards[a], league_cards_away)
        lam_g_h = mean_or_fallback(by_team_home_goals[h], league_goals_home)
        lam_g_a = mean_or_fallback(by_team_away_goals[a], league_goals_away)

        lambdas.append({
            "corners_home": lam_c_h, "corners_away": lam_c_a,
            "cards_home": lam_y_h, "cards_away": lam_y_a,
            "goals_home": lam_g_h, "goals_away": lam_g_a,
        })

        # Update rolling buffers AFTER this match's λ is computed
        by_team_home_corners[h].append(m["home_corners"])
        by_team_away_corners[a].append(m["away_corners"])
        by_team_home_cards[h].append(m["home_cards_for"])
        by_team_away_cards[a].append(m["away_cards_for"])
        by_team_home_goals[h].append(m["fthg"])
        by_team_away_goals[a].append(m["ftag"])
    return lambdas


def two_well_engine(p_over_prior: float, elo_skew: float) -> dict:
    """Two-well event space (over at +5x, under at -5x). Initial masses
    blend the rolling-Poisson prior with an Elo-derived skew.

    Returns {"over": p, "under": p}.
    """
    # Elo skew nudges the over/under split slightly. We're testing whether
    # the engine's well dynamics produce a sharper posterior than the
    # pure Poisson prior; the skew is intentionally small so we measure
    # the physics, not just a re-weighted Poisson.
    p_o = float(np.clip(p_over_prior + 0.10 * elo_skew, 0.05, 0.95))
    p_u = 1.0 - p_o
    space = EventSpace([
        Attractor("over", [5.0, 0.0], p_o),
        Attractor("under", [-5.0, 0.0], p_u),
    ])
    pos, masses, labels = e05._stack(space)

    rng = np.random.default_rng(seed=SEED)
    q_scale = np.array([0.3, 0.2]) * IC_SCALE
    p_scale = np.array([0.15, 0.15]) * IC_SCALE
    acc = {"over": 0.0, "under": 0.0}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=q_scale)
        p0 = rng.normal(scale=p_scale)
        q_end = e05.fast_final_q(pos, masses, q0, p0,
                                  duration=DURATION, dt=DT, C_d=C_D)
        post = e05.fast_posterior(q_end, pos, masses, labels,
                                   alpha=SOFT_ALPHA)
        for k in acc:
            acc[k] += post[k]
    s = sum(acc.values())
    return {k: v / s for k, v in acc.items()}


def brier(probs: dict, actual_over: bool) -> float:
    truth = {"over": 1.0 if actual_over else 0.0,
             "under": 0.0 if actual_over else 1.0}
    return sum((probs[k] - truth[k]) ** 2 for k in probs)


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000):
    rng = np.random.default_rng(SEED)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(deltas, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def main() -> None:
    matches = load_full_season()
    print("=== Orbita: niche-market baseline showdown ===")
    print(f"Panel    : {len(matches)} matches (2024/25 EPL)")
    print(f"Rolling K: {ROLLING_K} matches/team (separate home & away)")
    print(f"Engine   : {N_TRIALS} trials/match, 2-well event space")
    print()

    lambdas = rolling_lambdas(matches)
    elo_cache: dict = {}

    league_lc = float(np.mean([m["home_corners"] + m["away_corners"]
                                for m in matches]))
    league_ly = float(np.mean([m["home_cards_for"] + m["away_cards_for"]
                                for m in matches]))
    btts_rate = float(np.mean([m["actual_btts"] for m in matches]))

    print(f"League means: corners={league_lc:.2f}, "
          f"cards={league_ly:.2f}, BTTS={btts_rate:.3f}")
    print()

    cols = ["uniform", "league_poisson", "team_rolling", "engine"]
    markets = ["corners", "cards", "btts"]
    score = {c: {m: [] for m in markets} for c in cols}
    hits = {c: {m: 0 for m in markets} for c in cols}

    for i, (m, lam) in enumerate(zip(matches, lambdas), 1):
        actuals = {
            "corners": m["actual_corners"] > CORNER_LINE,
            "cards":   m["actual_cards"] > CARD_LINE,
            "btts":    bool(m["actual_btts"]),
        }
        elo_h = e03.elo_for(m["home_team"], m["date"], elo_cache)
        elo_a = e03.elo_for(m["away_team"], m["date"], elo_cache)
        # Normalise Elo diff to roughly [-1, 1]; standard club Elo
        # differences rarely exceed ~400.
        elo_skew = float(np.clip((elo_h - elo_a) / 400.0, -1.0, 1.0))

        # Uniform
        u_corners = {"over": 0.5, "under": 0.5}
        u_cards = {"over": 0.5, "under": 0.5}
        u_btts = {"over": 0.5, "under": 0.5}

        # League poisson
        p_corners = p_over(league_lc, CORNER_LINE)
        p_cards = p_over(league_ly, CARD_LINE)
        l_corners = {"over": p_corners, "under": 1 - p_corners}
        l_cards = {"over": p_cards, "under": 1 - p_cards}
        l_btts = {"over": btts_rate, "under": 1 - btts_rate}

        # Team rolling Poisson
        lam_c = lam["corners_home"] + lam["corners_away"]
        lam_y = lam["cards_home"] + lam["cards_away"]
        # BTTS rolling: P(home > 0) * P(away > 0) under independent Poissons
        p_h_score = 1.0 - exp(-lam["goals_home"])
        p_a_score = 1.0 - exp(-lam["goals_away"])
        p_btts = p_h_score * p_a_score
        t_corners = {"over": p_over(lam_c, CORNER_LINE),
                     "under": 1 - p_over(lam_c, CORNER_LINE)}
        t_cards = {"over": p_over(lam_y, CARD_LINE),
                   "under": 1 - p_over(lam_y, CARD_LINE)}
        t_btts = {"over": p_btts, "under": 1 - p_btts}

        # Engine — two-well with rolling-Poisson prior + Elo skew
        e_corners = two_well_engine(t_corners["over"], elo_skew)
        e_cards = two_well_engine(t_cards["over"], 0.0)  # no Elo skew on cards
        e_btts = two_well_engine(t_btts["over"], elo_skew)

        bundle = {
            "uniform": {"corners": u_corners, "cards": u_cards, "btts": u_btts},
            "league_poisson": {"corners": l_corners, "cards": l_cards, "btts": l_btts},
            "team_rolling": {"corners": t_corners, "cards": t_cards, "btts": t_btts},
            "engine": {"corners": e_corners, "cards": e_cards, "btts": e_btts},
        }
        for c in cols:
            for mk in markets:
                score[c][mk].append(brier(bundle[c][mk], actuals[mk]))
                modal = max(bundle[c][mk], key=bundle[c][mk].get)
                if (modal == "over") == actuals[mk]:
                    hits[c][mk] += 1

        if i % 50 == 0:
            print(f"  [{i:>3d}/{len(matches)}] "
                  f"corners.team={np.mean(score['team_rolling']['corners']):.4f} "
                  f"corners.eng={np.mean(score['engine']['corners']):.4f}",
                  flush=True)

    n = len(matches)
    print()
    print("=== aggregate Brier (lower better) ===")
    print(f"{'config':<16s}  " +
          "  ".join(f"{m+' Brier':>14s}" for m in markets))
    for c in cols:
        print(f"{c:<16s}  " +
              "  ".join(f"{np.mean(score[c][mk]):>14.4f}" for mk in markets))
    print()
    print("=== aggregate modal hit-rate ===")
    print(f"{'config':<16s}  " +
          "  ".join(f"{m+' hit':>12s}" for m in markets))
    for c in cols:
        print(f"{c:<16s}  " +
              "  ".join(f"{hits[c][mk] / n:>12.1%}" for mk in markets))
    print()
    print("=== bootstrap 90% CI on engine - team_rolling per-market Brier ===")
    for mk in markets:
        d = np.array(score["engine"][mk]) - np.array(score["team_rolling"][mk])
        mean, lo, hi = bootstrap_ci(d)
        verdict = ("engine BEATS team-rolling" if hi < 0 else
                   "team-rolling BEATS engine" if lo > 0 else
                   "tied (CI ∋ 0)")
        print(f"{mk:>10s}: delta={mean:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"{verdict}")


if __name__ == "__main__":
    main()

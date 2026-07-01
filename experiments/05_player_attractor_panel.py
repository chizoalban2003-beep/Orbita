"""05_player_attractor_panel.py — Option A on the 50-match EPL panel.

Scales the player-attractor prototype (experiment 04) up to the full
50-match EPL window (2024/25, Oct–Nov) used in experiment 03.

Three configurations are compared per match:

  Config A: BASELINE         — 3 H/D/L wells, no players (current model).
  Config B: MULTI-MARKET     — 6 joint wells (H/D/L × O/U 2.5 goals).
  Config C: MULTI + PLAYERS  — 6 wells + 22 player attractors (synthetic
                               4-3-3 lineups, ratings from ClubElo).

Each is scored on TWO markets:

  • H/D/L Brier vs Bet365 baseline.
  • Over/Under 2.5 goals Brier vs Bet365 baseline. (Configs B and C
    produce this directly via the joint posterior; Config A predicts
    only H/D/L and is not scored on O/U.)

Player attractors exert force on the body but are excluded from the
posterior — they perturb the orbit without competing for probability
mass (the load-bearing invariant from experiment 04).

Run with::

    PYTHONPATH=src python3 experiments/05_player_attractor_panel.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from math import log
from pathlib import Path

import numpy as np


def _stable_seed(s: str) -> int:
    """Deterministic 32-bit seed from a string. ``hash()`` in CPython is
    salted per-process by PYTHONHASHSEED, which makes synthetic lineups
    non-reproducible across runs."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4],
                          byteorder="little")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import (  # noqa: E402
    Attractor,
    Body,
    EventSpace,
    final_well_posterior,
    simulate,
)
from orbita.forces import SOFTENING  # noqa: E402


# ----- Vectorised final-state integrator --------------------------------
#
# The packaged :func:`orbita.simulate` loops over attractors per step.
# Once the force field has 25+ bodies that loop dominates runtime — a 50-
# match panel with player attractors would take hours. The version below
# stacks attractor state into numpy arrays so the per-step force costs
# one numpy ufunc instead of N Python iterations. Same velocity-Verlet
# update rule, same Plummer softening, same drag; identical math.

def fast_final_q(
    positions: np.ndarray,
    masses: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    duration: float,
    dt: float,
    C_d,
    m_body: float = 1.0,
    soft: float = SOFTENING,
) -> np.ndarray:
    """``C_d`` may be a scalar (isotropic), a length-2 array (anisotropic,
    per-axis), or a callable ``C_d(t) -> scalar | length-2``. Matches
    :func:`orbita.forces.drag_force`."""
    n_steps = int(duration / dt)
    soft2 = soft * soft
    q = q0.astype(float).copy()
    p = p0.astype(float).copy()
    mass_col = masses[:, None]
    is_callable = callable(C_d)
    cd_static = None if is_callable else np.asarray(C_d, dtype=float)

    def grav(qv: np.ndarray) -> np.ndarray:
        r = positions - qv
        d2 = np.einsum("ij,ij->i", r, r) + soft2
        return (mass_col * r / d2[:, None] ** 1.5).sum(axis=0)

    def cd_at(t: float) -> np.ndarray:
        if is_callable:
            return np.asarray(C_d(t), dtype=float)
        return cd_static

    t = 0.0
    F = m_body * grav(q) - cd_at(t) * (p / m_body)
    for _ in range(n_steps):
        p_half = p + 0.5 * dt * F
        q = q + dt * p_half / m_body
        t += dt
        F = m_body * grav(q) - cd_at(t) * (p_half / m_body)
        p = p_half + 0.5 * dt * F
    return q


def fast_posterior(
    q_end: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    labels: list,
    alpha: float = 2.0,
    soft: float = SOFTENING,
) -> dict:
    diff = positions - q_end
    d2 = np.einsum("ij,ij->i", diff, diff) + soft * soft
    w = masses / np.sqrt(d2) ** alpha
    s = w.sum()
    if s <= 0:
        n = len(labels)
        return {l: 1.0 / n for l in labels}
    w = w / s
    return {l: float(p) for l, p in zip(labels, w)}

# Reuse loaders from experiment 03 (football-data CSV + ClubElo).
spec = importlib.util.spec_from_file_location(
    "e03", ROOT / "experiments" / "03_footballdata_backtest.py")
e03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e03)

warnings.filterwarnings("ignore", message=r".*Renormalizing.*")


# ----- Knobs ------------------------------------------------------------

N_TRIALS = 30
SEED = 20260629
DT = 0.1
DURATION = 300.0
C_D = 0.04
IC_SCALE = 2.5
SOFT_ALPHA = 2.0

# 2D event-space geometry (matches experiment 04).
WIN_X = 5.0
OVER_Y = 3.0
UNDER_Y = -3.0
DRAW_OVER_Y = 4.0
DRAW_UNDER_Y = -2.5


# ----- Player synthesis -------------------------------------------------

ROLES = [
    ("GK",  1, (-2.5, -2.0)),
    ("DEF", 4, (-1.5, -0.5)),
    ("MID", 3, (+0.3, +1.0)),
    ("FWD", 3, (+2.2, +3.0)),
]
PLAYER_MASS_NORM = 600.0


def synth_lineup(team: str, x_sign: int, team_rating: float) -> list:
    """Deterministic-by-team synthetic 4-3-3. seeded by team name so the
    same lineup is reproduced every run."""
    rng = np.random.default_rng(seed=_stable_seed(team))
    out = []
    for role, count, (y_lo, y_hi) in ROLES:
        for _ in range(count):
            rating = float(rng.normal(loc=team_rating, scale=3.0))
            y = float(rng.uniform(y_lo, y_hi))
            x = x_sign * (WIN_X - rng.uniform(0.5, 2.0))
            out.append((role, x, y, rating))
    return out


def player_attractors(lineup: list, prefix: str) -> list:
    return [
        Attractor(f"{prefix}_{role}_{i}", [x, y], rating / PLAYER_MASS_NORM)
        for i, (role, x, y, rating) in enumerate(lineup)
    ]


# ----- Wells ------------------------------------------------------------

def baseline_wells(home_label, away_label, p_h, p_d, p_a):
    return [
        Attractor(home_label, [WIN_X, 0.0],   p_h),
        Attractor("draw",      [0.0, 4.0],     p_d),
        Attractor(away_label, [-WIN_X, 0.0], p_a),
    ]


def multimarket_wells(home_label, away_label, p_h, p_d, p_a, p_o, p_u):
    wells = []
    for x, name, p_match in [
        (WIN_X,   home_label, p_h),
        (0.0,     "draw",     p_d),
        (-WIN_X,  away_label, p_a),
    ]:
        for y, suffix, p_ou in [
            (DRAW_OVER_Y  if name == "draw" else OVER_Y,  "over",  p_o),
            (DRAW_UNDER_Y if name == "draw" else UNDER_Y, "under", p_u),
        ]:
            wells.append(Attractor(f"{name}_{suffix}", [x, y],
                                   p_match * p_ou))
    return wells


# ----- Monte Carlo ------------------------------------------------------

def _stack(space: EventSpace):
    positions = np.array([a.position for a in space.attractors], dtype=float)
    masses = np.array([a.mass for a in space.attractors], dtype=float)
    masses = masses / masses.sum()  # match EventSpace renormalisation
    labels = [a.label for a in space.attractors]
    return positions, masses, labels


def run(force_space: EventSpace, outcome_space: EventSpace) -> dict:
    rng = np.random.default_rng(seed=SEED)
    q_scale = np.array([0.3, 0.2]) * IC_SCALE
    p_scale = np.array([0.15, 0.15]) * IC_SCALE
    f_pos, f_mass, _ = _stack(force_space)
    o_pos, o_mass, o_labels = _stack(outcome_space)
    acc = {l: 0.0 for l in o_labels}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=q_scale)
        p0 = rng.normal(scale=p_scale)
        q_end = fast_final_q(f_pos, f_mass, q0, p0,
                             duration=DURATION, dt=DT, C_d=C_D)
        probs = fast_posterior(q_end, o_pos, o_mass, o_labels,
                               alpha=SOFT_ALPHA)
        for label, p in probs.items():
            acc[label] += p
    total = sum(acc.values())
    return {label: v / total for label, v in acc.items()}


# ----- Posterior reductions ---------------------------------------------

def marginal_hda(joint, home_label, away_label):
    out = {home_label: 0.0, "draw": 0.0, away_label: 0.0}
    for label, p in joint.items():
        if label.startswith(home_label):
            out[home_label] += p
        elif label.startswith("draw"):
            out["draw"] += p
        elif label.startswith(away_label):
            out[away_label] += p
    return out


def marginal_ou(joint):
    out = {"over": 0.0, "under": 0.0}
    for label, p in joint.items():
        if label.endswith("_over"):
            out["over"] += p
        elif label.endswith("_under"):
            out["under"] += p
    return out


def brier(probs, actual):
    return sum((p - (1.0 if l == actual else 0.0)) ** 2
               for l, p in probs.items())


def modal(probs):
    return max(probs, key=probs.get)


# ----- Load matches with O/U odds ---------------------------------------

def load_matches_with_ou() -> list:
    """Like experiment 03's load_matches() but also pulls B365 O/U 2.5
    closing odds and the actual goal total."""
    csv_path = e03.fetch_csv()
    out = []
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                date = e03.parse_date(row["Date"])
            except (KeyError, ValueError):
                continue
            if not (e03.DATE_FROM <= date <= e03.DATE_TO):
                continue
            try:
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
            # Two-way (over/under) needs its own normalisation; e03.devig
            # is hard-wired to the three-way H/D/L case.
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
                "home_team": home,
                "away_team": away,
                "home_label": home_label,
                "away_label": away_label,
                "p_h": ph, "p_d": pd, "p_a": pa,
                "p_over": po, "p_under": pu,
                "actual_hda": actual_hda,
                "actual_ou": actual_ou,
            })
    out.sort(key=lambda m: m["date"])
    return out


def main() -> None:
    matches = load_matches_with_ou()
    elo_cache: dict = {}

    print("=== Orbita: Option A on the 50-match EPL panel ===")
    print(f"Season       : 2024/25 EPL")
    print(f"Window       : {e03.DATE_FROM:%Y-%m-%d} → {e03.DATE_TO:%Y-%m-%d}")
    print(f"Panel size   : {len(matches)} matches")
    print(f"Monte Carlo  : N={N_TRIALS}  (seed={SEED})  | trials are seeded "
          f"per match, results are deterministic.")
    print()

    overall = defaultdict(lambda: defaultdict(list))
    hits = defaultdict(lambda: defaultdict(int))

    for i, m in enumerate(matches, 1):
        priors_hda = {m["home_label"]: m["p_h"],
                      "draw":            m["p_d"],
                      m["away_label"]:  m["p_a"]}
        priors_ou = {"over": m["p_over"], "under": m["p_under"]}

        # Score Bet365 baseline.
        overall["bookmaker"]["hda"].append(brier(priors_hda, m["actual_hda"]))
        overall["bookmaker"]["ou"].append(brier(priors_ou, m["actual_ou"]))
        if modal(priors_hda) == m["actual_hda"]:
            hits["bookmaker"]["hda"] += 1
        if modal(priors_ou) == m["actual_ou"]:
            hits["bookmaker"]["ou"] += 1

        # Config A: H/D/L baseline (no players).
        wells_b = baseline_wells(m["home_label"], m["away_label"],
                                  m["p_h"], m["p_d"], m["p_a"])
        space_b = EventSpace(wells_b)
        probs_a = run(space_b, space_b)
        overall["A_baseline"]["hda"].append(brier(probs_a, m["actual_hda"]))
        if modal(probs_a) == m["actual_hda"]:
            hits["A_baseline"]["hda"] += 1

        # Config B: multi-market (no players).
        wells_m = multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"],
            m["p_over"], m["p_under"],
        )
        space_m = EventSpace(wells_m)
        joint_b = run(space_m, space_m)
        hda_b = marginal_hda(joint_b, m["home_label"], m["away_label"])
        ou_b = marginal_ou(joint_b)
        overall["B_multimarket"]["hda"].append(brier(hda_b, m["actual_hda"]))
        overall["B_multimarket"]["ou"].append(brier(ou_b, m["actual_ou"]))
        if modal(hda_b) == m["actual_hda"]:
            hits["B_multimarket"]["hda"] += 1
        if modal(ou_b) == m["actual_ou"]:
            hits["B_multimarket"]["ou"] += 1

        # Config C: multi-market + 22 player attractors.
        r_h = e03.elo_for(m["home_team"], m["date"], elo_cache)
        r_a = e03.elo_for(m["away_team"], m["date"], elo_cache)
        home_xi = synth_lineup(m["home_team"], +1, r_h)
        away_xi = synth_lineup(m["away_team"], -1, r_a)
        players = (player_attractors(home_xi, m["home_label"]) +
                   player_attractors(away_xi, m["away_label"]))
        force_space = EventSpace(list(space_m.attractors) + players)
        joint_c = run(force_space, space_m)
        hda_c = marginal_hda(joint_c, m["home_label"], m["away_label"])
        ou_c = marginal_ou(joint_c)
        overall["C_multi_players"]["hda"].append(brier(hda_c, m["actual_hda"]))
        overall["C_multi_players"]["ou"].append(brier(ou_c, m["actual_ou"]))
        if modal(hda_c) == m["actual_hda"]:
            hits["C_multi_players"]["hda"] += 1
        if modal(ou_c) == m["actual_ou"]:
            hits["C_multi_players"]["ou"] += 1

        print(f"[{i:>2d}/{len(matches)}] {m['date']} {m['event']}  "
              f"actual={m['actual_hda']}/{m['actual_ou']}  "
              f"|  A.brier={overall['A_baseline']['hda'][-1]:.2f}  "
              f"B.brier={overall['B_multimarket']['hda'][-1]:.2f}  "
              f"C.brier={overall['C_multi_players']['hda'][-1]:.2f}",
              flush=True)

    n = len(matches)
    print()
    print("=== aggregate (mean Brier, lower is better) ===")
    print(f"{'config':<24s}  {'H/D/L Brier':>11s}  {'H/D/L hit':>10s}  "
          f"{'O/U Brier':>10s}  {'O/U hit':>8s}")
    for c in ("bookmaker", "A_baseline", "B_multimarket", "C_multi_players"):
        hda_b = float(np.mean(overall[c]["hda"])) if overall[c]["hda"] else float("nan")
        hda_h = hits[c]["hda"] / n if overall[c]["hda"] else float("nan")
        ou_b = float(np.mean(overall[c]["ou"])) if overall[c]["ou"] else float("nan")
        ou_h = hits[c]["ou"] / n if overall[c]["ou"] else float("nan")
        ou_b_s = f"{ou_b:>10.3f}" if not np.isnan(ou_b) else f"{'—':>10s}"
        ou_h_s = f"{ou_h:>7.1%}" if not np.isnan(ou_h) else f"{'—':>8s}"
        print(f"{c:<24s}  {hda_b:>11.3f}  {hda_h:>9.1%}  {ou_b_s}  {ou_h_s}")
    print()

    mb = float(np.mean(overall["bookmaker"]["hda"]))
    ma = float(np.mean(overall["A_baseline"]["hda"]))
    mm = float(np.mean(overall["B_multimarket"]["hda"]))
    mc = float(np.mean(overall["C_multi_players"]["hda"]))
    print("=== H/D/L verdicts (delta = engine - bookmaker) ===")
    print(f"A (baseline)         : {ma - mb:+.3f}  "
          f"({'engine wins' if ma < mb else 'market wins'})")
    print(f"B (multi-market)     : {mm - mb:+.3f}  "
          f"({'engine wins' if mm < mb else 'market wins'})")
    print(f"C (multi + players)  : {mc - mb:+.3f}  "
          f"({'engine wins' if mc < mb else 'market wins'})")
    print()
    mb_ou = float(np.mean(overall["bookmaker"]["ou"]))
    mm_ou = float(np.mean(overall["B_multimarket"]["ou"]))
    mc_ou = float(np.mean(overall["C_multi_players"]["ou"]))
    print("=== O/U 2.5 verdicts (delta = engine - bookmaker) ===")
    print(f"B (multi-market)     : {mm_ou - mb_ou:+.3f}  "
          f"({'engine wins' if mm_ou < mb_ou else 'market wins'})")
    print(f"C (multi + players)  : {mc_ou - mb_ou:+.3f}  "
          f"({'engine wins' if mc_ou < mb_ou else 'market wins'})")


if __name__ == "__main__":
    main()

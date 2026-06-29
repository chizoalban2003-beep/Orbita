"""04_player_attractor_prototype.py — Option A + multi-market wells.

Prototype of the architecture proposed in the v0.3.4 design discussion:

  1. Per-player attractors. Each starting-XI player is a small attractor
     in the event-space. Position is set by role (FWD pulls toward their
     team's high-scoring win, GK toward their team's clean-sheet win),
     mass scales with rating, sized to perturb the field rather than
     dominate the outcome wells.

  2. Wells beyond H/D/L. The event-space carries 6 joint outcome wells
     covering H/D/L × Over/Under 2.5 goals. Same machinery extends to
     cards, shots, BTTS — anything the bookmaker prices as a separate
     market — by adding wells at the corresponding joint outcomes.

Key invariant: player attractors exert force on the body but do NOT
compete for posterior mass. The posterior is computed only over the
outcome wells, so player attractors perturb the orbit without "stealing"
probability into themselves.

We run the prototype on one match — Newcastle vs Arsenal, 2024-11-02
(EPL matchday 10, actual result 1-0 to Newcastle, i.e. home win + under
2.5). Real Bet365 closing odds from football-data, synthetic 4-3-3
lineups with ratings derived from ClubElo team strength.

Three configurations are compared:

  Config 1: BASELINE         — 3 H/D/L wells. Current Orbita model.
  Config 2: MULTI-MARKET     — 6 joint wells (H/D/L × O/U 2.5).
  Config 3: MULTI + PLAYERS  — 6 wells + 22 player attractors.

Run with::

    PYTHONPATH=src python3 experiments/04_player_attractor_prototype.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import (  # noqa: E402
    Attractor,
    Body,
    EventSpace,
    final_well_posterior,
    simulate,
)

warnings.filterwarnings("ignore", message=r".*Renormalizing.*")


# ----- Match facts ------------------------------------------------------

HOME = "newcastle"
AWAY = "arsenal"
# Bet365 closing odds (football-data.co.uk, 2024/25 EPL).
B365_HDA = (4.20, 3.75, 1.83)
B365_OU25 = (1.91, 1.99)  # (over, under)
ACTUAL_RESULT = "H"
ACTUAL_GOALS = (1, 0)
ACTUAL_OVER = sum(ACTUAL_GOALS) > 2.5  # False → under

# Team strengths from ClubElo on 2024-11-02 (cached in experiment 03).
ELO_HOME = 1817.0
ELO_AWAY = 1981.0


def elo_to_rating(elo: float) -> float:
    return max(0.0, min(100.0, 50.0 + (elo - 1500.0) / 12.0))


# ----- 2D event-space geometry ------------------------------------------
# x = WIN axis  (positive = home win, negative = away win, 0 = draw)
# y = GOALS axis (positive = over 2.5, negative = under 2.5)

WIN_X = 5.0
OVER_Y = 3.0
UNDER_Y = -3.0
DRAW_OVER_Y = 4.0    # draws push slightly higher in the y because
DRAW_UNDER_Y = -2.5  # the "draw" prior tends to overlap with under.


def devig(odds: tuple) -> list:
    p = [1.0 / o for o in odds]
    s = sum(p)
    return [x / s for x in p]


# ----- Outcome wells ----------------------------------------------------

def baseline_wells(p_h: float, p_d: float, p_a: float) -> list:
    """3-well H/D/L."""
    return [
        Attractor(f"{HOME}_win", [WIN_X, 0.0],   p_h),
        Attractor("draw",         [0.0, 4.0],     p_d),
        Attractor(f"{AWAY}_win", [-WIN_X, 0.0], p_a),
    ]


def multimarket_wells(p_h: float, p_d: float, p_a: float,
                      p_o: float, p_u: float) -> list:
    """6 joint wells H/D/L × O/U 2.5.

    Joint masses assume independence between match result and goals
    market. Real bookmakers price the correlation (high-scoring games
    skew toward home wins in EPL) but the qualitative shape of the
    posterior is unaffected for the prototype.
    """
    wells = []
    for x, name, p_match in [
        (WIN_X,   f"{HOME}_win", p_h),
        (0.0,     "draw",        p_d),
        (-WIN_X,  f"{AWAY}_win", p_a),
    ]:
        for y_over, y_under in [(OVER_Y, UNDER_Y)]:
            # Draw is naturally lower-scoring so we offset its y.
            yo = DRAW_OVER_Y if name == "draw" else y_over
            yu = DRAW_UNDER_Y if name == "draw" else y_under
            wells.append(Attractor(f"{name}_over",  [x, yo], p_match * p_o))
            wells.append(Attractor(f"{name}_under", [x, yu], p_match * p_u))
    return wells


# ----- Player attractors ------------------------------------------------
# Role → (count in 4-3-3, y-range biasing toward over/under-scoring).
# GK and DEF push the body toward "clean-sheet win" (low y on their
# team's side). FWD pushes toward "high-scoring win" (high y). MID is
# neutral, slightly above the win-axis.
ROLES = [
    ("GK",  1, (-2.5, -2.0)),
    ("DEF", 4, (-1.5, -0.5)),
    ("MID", 3, (+0.3, +1.0)),
    ("FWD", 3, (+2.2, +3.0)),
]
PLAYER_MASS_NORM = 600.0  # per-player mass ≈ rating / 600, small by design


def synth_lineup(team: str, x_sign: int, team_rating: float) -> list:
    """Deterministic-by-team synthetic 4-3-3. ratings ~ N(team_rating, 3)."""
    rng = np.random.default_rng(seed=hash(team) & 0xFFFFFFFF)
    players = []
    for role, count, (y_lo, y_hi) in ROLES:
        for _ in range(count):
            rating = float(rng.normal(loc=team_rating, scale=3.0))
            y = float(rng.uniform(y_lo, y_hi))
            # x jitter so players spread across the team's half rather
            # than stacking at the touchline.
            x = x_sign * (WIN_X - rng.uniform(0.5, 2.0))
            players.append((role, x, y, rating))
    return players


def player_attractors(lineup: list, prefix: str) -> list:
    return [
        Attractor(f"{prefix}_{role}_{i}", [x, y], rating / PLAYER_MASS_NORM)
        for i, (role, x, y, rating) in enumerate(lineup)
    ]


# ----- Monte Carlo ------------------------------------------------------

N_TRIALS = 100
SEED = 20260629
DT = 0.1
DURATION = 600.0
C_D = 0.04
IC_SCALE = 2.5   # soccer template
SOFT_ALPHA = 2.0


def run(force_space: EventSpace, outcome_space: EventSpace) -> dict:
    """MC the body through ``force_space`` (which may include player
    attractors), but compute the posterior over ``outcome_space`` only.

    This is the load-bearing invariant: players exert force on the orbit
    but do not classify the final state. They are *part of the field*,
    not part of the prediction space.
    """
    rng = np.random.default_rng(seed=SEED)
    q_scale = np.array([0.3, 0.2]) * IC_SCALE
    p_scale = np.array([0.15, 0.15]) * IC_SCALE
    acc = {a.label: 0.0 for a in outcome_space.attractors}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=q_scale)
        p0 = rng.normal(scale=p_scale)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(force_space, body=body, duration=DURATION, dt=DT,
                       C_d=C_D, ic_scale=IC_SCALE)
        probs = final_well_posterior(sol, outcome_space, alpha=SOFT_ALPHA)
        for label, p in probs.items():
            acc[label] += p
    total = sum(acc.values())
    return {label: v / total for label, v in acc.items()}


# ----- Posterior reductions ---------------------------------------------

def marginal_hda(joint: dict) -> dict:
    out = {f"{HOME}_win": 0.0, "draw": 0.0, f"{AWAY}_win": 0.0}
    for label, p in joint.items():
        if label.startswith(f"{HOME}_win"):
            out[f"{HOME}_win"] += p
        elif label.startswith("draw"):
            out["draw"] += p
        elif label.startswith(f"{AWAY}_win"):
            out[f"{AWAY}_win"] += p
    return out


def marginal_ou(joint: dict) -> dict:
    out = {"over_2.5": 0.0, "under_2.5": 0.0}
    for label, p in joint.items():
        if label.endswith("_over"):
            out["over_2.5"] += p
        elif label.endswith("_under"):
            out["under_2.5"] += p
    return out


def fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.0%}" for k, v in d.items())


def main() -> None:
    ph, pd, pa = devig(B365_HDA)
    po, pu = devig(B365_OU25)
    r_h = elo_to_rating(ELO_HOME)
    r_a = elo_to_rating(ELO_AWAY)

    print("=== Orbita: per-player attractor prototype (Option A) ===")
    print(f"Match     : {HOME.capitalize()} vs {AWAY.capitalize()} 2024-11-02 (EPL)")
    print(f"Result    : {ACTUAL_GOALS[0]}-{ACTUAL_GOALS[1]}  → "
          f"{HOME}_win + {'over' if ACTUAL_OVER else 'under'}_2.5")
    print(f"Bet365    : H={ph:.0%}  D={pd:.0%}  A={pa:.0%}  | "
          f"O2.5={po:.0%}  U2.5={pu:.0%}")
    print(f"ClubElo   : {HOME}={r_h:.1f}  {AWAY}={r_a:.1f}")
    print(f"MC        : N={N_TRIALS}  seed={SEED}  ic_scale={IC_SCALE}")
    print()

    # Config 1: BASELINE (H/D/L wells only)
    wells_b = baseline_wells(ph, pd, pa)
    space_b = EventSpace(wells_b)
    probs_b = run(space_b, space_b)
    print("--- Config 1: BASELINE (3 H/D/L wells) ---")
    print(f"  posterior          : {fmt(probs_b)}")
    print()

    # Config 2: MULTI-MARKET (6 joint wells)
    wells_m = multimarket_wells(ph, pd, pa, po, pu)
    space_m = EventSpace(wells_m)
    probs_m = run(space_m, space_m)
    print("--- Config 2: MULTI-MARKET (6 joint wells, H/D/L × O/U 2.5) ---")
    print(f"  joint posterior    : {fmt(probs_m)}")
    print(f"  → H/D/L marginal   : {fmt(marginal_hda(probs_m))}")
    print(f"  → O/U marginal     : {fmt(marginal_ou(probs_m))}")
    print()

    # Config 3: MULTI + PLAYERS
    home_lineup = synth_lineup(HOME, +1, r_h)
    away_lineup = synth_lineup(AWAY, -1, r_a)
    players = (player_attractors(home_lineup, HOME) +
               player_attractors(away_lineup, AWAY))
    force_space = EventSpace(list(space_m.attractors) + players)
    probs_f = run(force_space, space_m)
    print("--- Config 3: MULTI + PLAYERS (6 wells + 22 player attractors) ---")
    print(f"  field size         : {len(force_space.attractors)} entities")
    print(f"  player mass range  : "
          f"{min(p.mass for p in players):.3f} – {max(p.mass for p in players):.3f}")
    print(f"  outcome-well masses: {min(w.mass for w in wells_m):.3f} – "
          f"{max(w.mass for w in wells_m):.3f}")
    print(f"  joint posterior    : {fmt(probs_f)}")
    print(f"  → H/D/L marginal   : {fmt(marginal_hda(probs_f))}")
    print(f"  → O/U marginal     : {fmt(marginal_ou(probs_f))}")
    print()

    # Diff: how much did the player layer move the H/D/L marginal?
    m_m = marginal_hda(probs_m)
    m_f = marginal_hda(probs_f)
    print("--- ΔH/D/L (Config 3 − Config 2): effect of player attractors ---")
    for k in m_m:
        delta_pp = (m_f[k] - m_m[k]) * 100
        marker = "← actual" if k == f"{HOME}_win" else ""
        print(f"  {k:<20s}  {m_m[k]:.1%} → {m_f[k]:.1%}  "
              f"({delta_pp:+.1f} pp)  {marker}")
    print()

    ou_m = marginal_ou(probs_m)
    ou_f = marginal_ou(probs_f)
    print("--- ΔO/U (Config 3 − Config 2): effect of player attractors ---")
    for k in ou_m:
        delta_pp = (ou_f[k] - ou_m[k]) * 100
        marker = "← actual" if (k == "over_2.5") == ACTUAL_OVER else ""
        print(f"  {k:<20s}  {ou_m[k]:.1%} → {ou_f[k]:.1%}  "
              f"({delta_pp:+.1f} pp)  {marker}")


if __name__ == "__main__":
    main()

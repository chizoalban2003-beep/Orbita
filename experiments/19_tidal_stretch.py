"""19_tidal_stretch.py — does game-state tidal deformation beat the engine
without it, on live (half-time) Over/Under 2.5 prediction?

The hypothesis (tidal stretching / game-state desperation): when a match
is heading decisively toward a result late on, the losing side abandons
shape — the goals axis stretches, deepening whichever Over/Under well the
state already leans toward. A pre-match closing line can't price the exact
non-linear tipping point of that desperation.

We test it *in play*, which is the only honest test:

  1. Seed the orbiting body at each match's REAL half-time state
     (from HTHG/HTAG in football-data.co.uk's cached EPL 2024/25 CSV):
       x0 ← goal difference   (leaning toward the home/away win well)
       y0 ← goals banked so far vs the 2.5 line (leaning over/under)
     Well MASSES stay the pre-match de-vigged B365 priors — the field is
     the market's belief; the body is the live state moving through it.
  2. Simulate the second half in the joint 6-well H/D/A × O/U space, with
     the tidal force OFF (baseline) and ON (treatment). Identical seeds.
  3. Score the actual full-time Over/Under 2.5 outcome (Brier).

TRAIN the tidal coefficient kappa on the first 60% of the season by date,
EVALUATE the winner out-of-sample on the last 40%. Beating the tidal-OFF
engine on the held-out set (not the pre-match line, which lacks the
half-time score) is the honest bar.

Run:  PYTHONPATH=src python3 experiments/19_tidal_stretch.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita.forces import SOFTENING  # noqa: E402

# reuse football-data loaders (cached CSV path, de-vig, slug)
spec = importlib.util.spec_from_file_location(
    "e03", ROOT / "experiments" / "03_footballdata_backtest.py")
e03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e03)

# ---- geometry (matches experiment 05) + knobs --------------------------
WIN_X, OVER_Y, UNDER_Y = 5.0, 3.0, -3.0
DRAW_OVER_Y, DRAW_UNDER_Y = 4.0, -2.5
DURATION_FULL, T_HALF = 300.0, 150.0     # 2nd half = [150, 300]
C_D, DT, ALPHA = 0.04, 0.15, 2.0
IC_SCALE = 2.5
N_TRIALS = 40
SEED = 20260708
LAM = 3.0                                # desperation ramp steepness
KAPPA_GRID = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]
TRAIN_FRAC = 0.60
MODE = "directional"                     # "symmetric" | "directional"

# half-time state -> seed position
X_PER_GOAL = 1.8      # each goal of HT lead pulls this far toward the win well
Y_PER_GOAL = 1.4      # each goal vs the ~1.5 HT-goals baseline tilts O/U
HT_GOAL_BASELINE = 1.5


def joint_wells(p_h, p_d, p_a, p_o, p_u):
    """6-well H/D/A x O/U joint space. Returns (positions[6,2], masses[6])."""
    rows = [
        (WIN_X, p_h, OVER_Y, UNDER_Y),
        (0.0,   p_d, DRAW_OVER_Y, DRAW_UNDER_Y),
        (-WIN_X, p_a, OVER_Y, UNDER_Y),
    ]
    pos, mass, lab = [], [], []
    for x, pm, oy, uy in rows:
        pos.append([x, oy]);  mass.append(pm * p_o); lab.append("over")
        pos.append([x, uy]);  mass.append(pm * p_u); lab.append("under")
    pos = np.array(pos, float)
    mass = np.array(mass, float)
    mass = mass / mass.sum()
    return pos, mass, lab


def seed_from_halftime(ht_h, ht_a):
    gd = ht_h - ht_a
    x0 = float(np.clip(gd * X_PER_GOAL, -WIN_X, WIN_X))
    tot = ht_h + ht_a
    y0 = float(np.clip((tot - HT_GOAL_BASELINE) * Y_PER_GOAL, UNDER_Y, OVER_Y))
    return np.array([x0, y0])


def second_half_batch(pos, mass, seed_q, kappa, lam, n_trials, rng):
    """Vectorised velocity-Verlet over the 2nd half for a batch of MC bodies.

    Bodies share the half-time seed with per-trial IC noise. Tidal force is
    the y-only stretch from orbita.forces.tidal_force, inlined here across
    the whole batch. Returns final positions (n_trials, 2)."""
    q = seed_q[None, :] + rng.normal(scale=np.array([0.3, 0.2]) * IC_SCALE,
                                     size=(n_trials, 2))
    p = rng.normal(scale=np.array([0.15, 0.15]) * IC_SCALE, size=(n_trials, 2))
    soft2 = SOFTENING ** 2
    m_col = mass[None, :, None]          # (1, W, 1)
    n_steps = int(T_HALF / DT)
    t_final = DURATION_FULL

    def forces(qb, pb, t):
        r = pos[None, :, :] - qb[:, None, :]           # (T, W, 2)
        d2 = np.einsum("twk,twk->tw", r, r) + soft2     # (T, W)
        g = (m_col * r / d2[:, :, None] ** 1.5).sum(axis=1)   # (T, 2)
        F = g - C_D * pb                                # drag (m_body=1)
        if kappa != 0.0:
            ramp = np.exp(lam * (t / t_final - 1.0))
            F = F.copy()
            if MODE == "symmetric":
                # amplify whichever O/U lean the state already has, scaled
                # by result decisiveness (x-pull magnitude)
                strain = np.abs(g[:, 0])
                F[:, 1] += kappa * strain * ramp * np.sign(qb[:, 1])
            else:
                # directional: a trailing team creates goals. Over-pressure
                # (+y) that is bump-shaped in the goal margin |x| — ~0 at
                # level (nobody desperate), peaks near a one-goal margin
                # (chasing team all-in but result still reachable), fades to
                # ~0 at a blowout (game dead).
                margin = np.abs(qb[:, 0])
                bump = margin * np.exp(-margin / X_PER_GOAL)
                F[:, 1] += kappa * bump * ramp
        return g, F

    t = T_HALF
    _, F = forces(q, p, t)
    for _ in range(n_steps):
        p_half = p + 0.5 * DT * F
        q = q + DT * p_half
        t += DT
        _, F = forces(q, p_half, t)
        p = p_half + 0.5 * DT * F
    return q


def ou_prob_over(q_end, pos, mass, over_mask):
    """Soft-assign each body to wells, marginalise to P(over)."""
    diff = pos[None, :, :] - q_end[:, None, :]
    d2 = np.einsum("twk,twk->tw", diff, diff) + SOFTENING ** 2
    w = mass[None, :] / np.sqrt(d2) ** ALPHA           # (T, W)
    w = w / w.sum(axis=1, keepdims=True)
    p_over = w[:, over_mask].sum(axis=1).mean()
    return float(p_over)


def load_matches_ht():
    """All EPL 2024/25 matches with HT score, FT total, B365 H/D/A + O/U 2.5."""
    csv_path = e03.fetch_csv()
    out = []
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                date = e03.parse_date(row["Date"])
                h, d, a = float(row["B365H"]), float(row["B365D"]), float(row["B365A"])
                o, u = float(row["B365>2.5"]), float(row["B365<2.5"])
                fthg, ftag = int(row["FTHG"]), int(row["FTAG"])
                hthg, htag = int(row["HTHG"]), int(row["HTAG"])
            except (KeyError, ValueError):
                continue
            ph, pd, pa = e03.devig(h, d, a)
            io, iu = 1.0 / o, 1.0 / u
            po, pu = io / (io + iu), iu / (io + iu)
            out.append({
                "date": date,
                "p_h": ph, "p_d": pd, "p_a": pa, "p_over": po, "p_under": pu,
                "ht_h": hthg, "ht_a": htag,
                "actual_over": (fthg + ftag) > 2.5,
            })
    out.sort(key=lambda m: m["date"])
    return out


def brier_over(p_over, actual_over):
    y = 1.0 if actual_over else 0.0
    return (p_over - y) ** 2 + ((1 - p_over) - (1 - y)) ** 2


def engine_ou_briers(matches, kappa, lam=LAM):
    """Per-match O/U Brier for the half-time-seeded engine at this kappa."""
    rng = np.random.default_rng(SEED)
    briers = []
    for m in matches:
        pos, mass, lab = joint_wells(m["p_h"], m["p_d"], m["p_a"],
                                     m["p_over"], m["p_under"])
        over_mask = np.array([l == "over" for l in lab])
        seed_q = seed_from_halftime(m["ht_h"], m["ht_a"])
        q_end = second_half_batch(pos, mass, seed_q, kappa, lam, N_TRIALS, rng)
        p_over = ou_prob_over(q_end, pos, mass, over_mask)
        briers.append(brier_over(p_over, m["actual_over"]))
    return np.array(briers)


def bootstrap_ci(deltas, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    means = [rng.choice(deltas, size=len(deltas), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def main():
    matches = load_matches_ht()
    n = len(matches)
    cut = int(n * TRAIN_FRAC)
    train, evalset = matches[:cut], matches[cut:]
    print("=" * 68)
    print("Tidal-stretch in-play O/U 2.5 backtest — EPL 2024/25")
    print(f"matches={n}  train={len(train)} (to {train[-1]['date']:%Y-%m-%d})"
          f"  eval={len(evalset)}  N_trials={N_TRIALS}  lam={LAM}")
    print("=" * 68)

    # book reference (pre-match, no HT info) on eval
    book = np.array([brier_over(m["p_over"], m["actual_over"]) for m in evalset])

    # TRAIN: sweep kappa
    print("\nTRAIN (first 60%): O/U Brier by tidal kappa")
    best_k, best_b = 0.0, np.inf
    for k in KAPPA_GRID:
        b = engine_ou_briers(train, k).mean()
        flag = ""
        if b < best_b:
            best_b, best_k = b, k
            flag = "  <= best"
        print(f"  kappa={k:<5}  mean Brier={b:.4f}{flag}")

    # EVAL out-of-sample: baseline (kappa=0) vs trained kappa
    base_e = engine_ou_briers(evalset, 0.0)
    tid_e = engine_ou_briers(evalset, best_k)
    delta = base_e - tid_e            # positive = tidal improves
    lo, hi = bootstrap_ci(delta)

    print(f"\nEVAL (held-out last 40%, n={len(evalset)}), trained kappa={best_k}")
    print(f"  pre-match book (no HT info) : {book.mean():.4f}")
    print(f"  engine tidal OFF (kappa=0)  : {base_e.mean():.4f}")
    print(f"  engine tidal ON  (kappa={best_k}) : {tid_e.mean():.4f}")
    print(f"  improvement (OFF - ON)      : {delta.mean():+.4f}  "
          f"90% CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = ("TIDAL HELPS (CI excludes 0)" if lo > 0 else
               "TIDAL HURTS (CI excludes 0)" if hi < 0 else
               "INCONCLUSIVE (CI spans 0)")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()

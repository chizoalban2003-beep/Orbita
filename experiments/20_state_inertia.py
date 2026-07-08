"""20_state_inertia.py — does per-match "inertial mass" (state dispersion)
beat the market on pre-match Over/Under 2.5?

The hypothesis: a bookmaker's Poisson/NegBin pricing assumes a fixed
variance-to-mean relationship. Real matches differ in *structural
rigidity* — two controlled possession sides produce a low-variance total;
two chaotic transitional sides produce an overdispersed one. If Orbita
modulates the volatility of the match state per match by a structural
proxy, it could price dispersion the market underprices.

HONEST PHYSICS NOTE. In Orbita gravitational *acceleration* is
mass-independent (F_grav = m·g ⇒ a = g; the equivalence principle), so a
literal body mass does not make the state "resist gravity". Body mass only
touches drag (a_drag = −(C_d/m)v) and q̇ = p/m. We therefore test the idea
two ways:

  (A) literal body mass m — global sweep (does the mass knob move O/U
      Brier at all?).
  (B) per-match dispersion — scale the Monte-Carlo IC spread per match by
      a leakage-free rigidity proxy (the engine's genuine variance knob).
      Volatile teams → wider IC → more dispersed final state.

Rigidity proxy (leakage-free): each team's EXPANDING variance of match
total goals, using only matches BEFORE the current one; the two teams'
values are summed. Low variance = rigid; high = volatile. z-scored using
TRAIN-set statistics only. Matches with < MIN_GAMES history for either
team get a neutral proxy (0).

Everything is PRE-MATCH (de-vigged B365 priors as the field; body from
neutral ICs). Train the knob on the first 60% of the season by date,
evaluate out-of-sample on the last 40%. Beating the closing line's O/U
Brier on the held-out set is the bar.

Run:  PYTHONPATH=src python3 experiments/20_state_inertia.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita.forces import SOFTENING  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e03", ROOT / "experiments" / "03_footballdata_backtest.py")
e03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e03)

WIN_X, OVER_Y, UNDER_Y = 5.0, 3.0, -3.0
DRAW_OVER_Y, DRAW_UNDER_Y = 4.0, -2.5
DURATION, DT, ALPHA = 300.0, 0.2, 2.0
IC_SCALE = 2.5
N_TRIALS = 32
SEED = 20260709
TRAIN_FRAC = 0.60
MIN_GAMES = 4
MASS_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]        # (A) literal body mass
S_GRID = [-0.30, -0.15, 0.0, 0.15, 0.30, 0.50]     # (B) dispersion coupling


def joint_wells(p_h, p_d, p_a, p_o, p_u):
    rows = [(WIN_X, p_h, OVER_Y, UNDER_Y),
            (0.0, p_d, DRAW_OVER_Y, DRAW_UNDER_Y),
            (-WIN_X, p_a, OVER_Y, UNDER_Y)]
    pos, mass, lab = [], [], []
    for x, pm, oy, uy in rows:
        pos.append([x, oy]); mass.append(pm * p_o); lab.append("over")
        pos.append([x, uy]); mass.append(pm * p_u); lab.append("under")
    m = np.array(mass, float)
    return np.array(pos, float), m / m.sum(), lab


def sim_batch(pos, mass, m_body, disp, n_trials, rng):
    """Vectorised full-match Verlet for a batch of MC bodies from neutral
    ICs. ``m_body`` = inertial mass (canonical F=m·g − C_d·p/m). ``disp``
    scales the IC spread (the dispersion knob). Returns final q (T,2)."""
    q = rng.normal(scale=np.array([0.3, 0.2]) * IC_SCALE * disp, size=(n_trials, 2))
    p = rng.normal(scale=np.array([0.15, 0.15]) * IC_SCALE * disp, size=(n_trials, 2))
    soft2 = SOFTENING ** 2
    m_col = mass[None, :, None]
    n_steps = int(DURATION / DT)

    def force(qb, pb):
        r = pos[None, :, :] - qb[:, None, :]
        d2 = np.einsum("twk,twk->tw", r, r) + soft2
        g = (m_col * r / d2[:, :, None] ** 1.5).sum(axis=1)
        return m_body * g - C_D_drag * (pb / m_body)

    F = force(q, p)
    for _ in range(n_steps):
        p_half = p + 0.5 * DT * F
        q = q + DT * p_half / m_body
        F = force(q, p_half)
        p = p_half + 0.5 * DT * F
    return q


C_D_drag = 0.04


def p_over(q_end, pos, mass, over_mask):
    diff = pos[None, :, :] - q_end[:, None, :]
    d2 = np.einsum("twk,twk->tw", diff, diff) + SOFTENING ** 2
    w = mass[None, :] / np.sqrt(d2) ** ALPHA
    w = w / w.sum(axis=1, keepdims=True)
    return float(w[:, over_mask].sum(axis=1).mean())


def brier_over(po, actual_over):
    y = 1.0 if actual_over else 0.0
    return (po - y) ** 2 + ((1 - po) - (1 - y)) ** 2


def load_prematch():
    """All EPL 2024/25 matches (date order) with pre-match B365 priors,
    actual O/U 2.5, and a leakage-free rigidity proxy from each team's
    expanding total-goals variance."""
    csv_path = e03.fetch_csv()
    rows = []
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                date = e03.parse_date(row["Date"])
                h, d, a = float(row["B365H"]), float(row["B365D"]), float(row["B365A"])
                o, u = float(row["B365>2.5"]), float(row["B365<2.5"])
                fthg, ftag = int(row["FTHG"]), int(row["FTAG"])
            except (KeyError, ValueError):
                continue
            rows.append((date, row["HomeTeam"], row["AwayTeam"], h, d, a, o, u, fthg, ftag))
    rows.sort(key=lambda r: r[0])

    history = defaultdict(list)   # team -> list of prior match total goals
    out = []
    for date, home, away, h, d, a, o, u, fthg, ftag in rows:
        ph, pd, pa = e03.devig(h, d, a)
        io, iu = 1.0 / o, 1.0 / u
        po_, pu_ = io / (io + iu), iu / (io + iu)
        tot = fthg + ftag
        hh, ah = history[home], history[away]
        if len(hh) >= MIN_GAMES and len(ah) >= MIN_GAMES:
            rigidity = float(np.var(hh) + np.var(ah))   # high = volatile
        else:
            rigidity = np.nan
        out.append({
            "date": date, "p_h": ph, "p_d": pd, "p_a": pa,
            "p_over": po_, "p_under": pu_,
            "actual_over": tot > 2.5, "rigidity": rigidity,
        })
        history[home].append(tot)
        history[away].append(tot)
    return out


def eval_config(matches, m_body_fn, disp_fn):
    rng = np.random.default_rng(SEED)
    briers = []
    for m in matches:
        pos, mass, lab = joint_wells(m["p_h"], m["p_d"], m["p_a"],
                                     m["p_over"], m["p_under"])
        over_mask = np.array([l == "over" for l in lab])
        q_end = sim_batch(pos, mass, m_body_fn(m), disp_fn(m), N_TRIALS, rng)
        briers.append(brier_over(p_over(q_end, pos, mass, over_mask), m["actual_over"]))
    return np.array(briers)


def bootstrap_ci(deltas, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    ms = [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(ms, 5)), float(np.percentile(ms, 95))


def main():
    matches = load_prematch()
    n = len(matches)
    cut = int(n * TRAIN_FRAC)
    train, ev = matches[:cut], matches[cut:]
    # z-score rigidity using TRAIN stats only
    rig_train = np.array([m["rigidity"] for m in train if not np.isnan(m["rigidity"])])
    mu, sd = rig_train.mean(), rig_train.std() + 1e-9

    def zr(m):
        return 0.0 if np.isnan(m["rigidity"]) else (m["rigidity"] - mu) / sd

    print("=" * 68)
    print("State-inertia pre-match O/U 2.5 backtest — EPL 2024/25")
    print(f"matches={n} train={len(train)} eval={len(ev)} "
          f"N={N_TRIALS} rigidity coverage(train)={len(rig_train)}/{len(train)}")
    print("=" * 68)

    book = np.array([brier_over(m["p_over"], m["actual_over"]) for m in ev])
    base = eval_config(ev, lambda m: 1.0, lambda m: 1.0)
    print(f"\nREFERENCE (held-out): market={book.mean():.4f}  "
          f"engine m=1,disp=1={base.mean():.4f}")

    # (A) literal body-mass global sweep on TRAIN
    print("\n(A) literal body mass — TRAIN O/U Brier:")
    bestm, bestmb = 1.0, np.inf
    for mb in MASS_GRID:
        b = eval_config(train, lambda m, _mb=mb: _mb, lambda m: 1.0).mean()
        flag = "  <=" if b < bestmb else ""
        if b < bestmb:
            bestmb, bestm = b, mb
        print(f"    m={mb:<4} Brier={b:.4f}{flag}")

    # (B) per-match dispersion coupling on TRAIN
    print("\n(B) per-match dispersion  disp=exp(s·z_rigidity) — TRAIN O/U Brier:")
    bests, bestsb = 0.0, np.inf
    for s in S_GRID:
        b = eval_config(train, lambda m: 1.0,
                        lambda m, _s=s: float(np.exp(_s * zr(m)))).mean()
        flag = "  <=" if b < bestsb else ""
        if b < bestsb:
            bestsb, bests = b, s
        print(f"    s={s:<+5} Brier={b:.4f}{flag}")

    # EVAL out-of-sample
    a_ev = eval_config(ev, lambda m: bestm, lambda m: 1.0)
    b_ev = eval_config(ev, lambda m: 1.0, lambda m: float(np.exp(bests * zr(m))))
    dA, dB = base - a_ev, base - b_ev
    loA, hiA = bootstrap_ci(dA)
    loB, hiB = bootstrap_ci(dB)
    print(f"\nEVAL (held-out n={len(ev)}):")
    print(f"  market                         : {book.mean():.4f}")
    print(f"  engine baseline (m=1,disp=1)   : {base.mean():.4f}")
    print(f"  (A) trained mass m={bestm:<4}        : {a_ev.mean():.4f}  "
          f"Δvs baseline {dA.mean():+.4f} CI[{loA:+.4f},{hiA:+.4f}]")
    print(f"  (B) trained dispersion s={bests:<+5}   : {b_ev.mean():.4f}  "
          f"Δvs baseline {dB.mean():+.4f} CI[{loB:+.4f},{hiB:+.4f}]")

    def verdict(lo, hi):
        return ("HELPS" if lo > 0 else "HURTS" if hi < 0 else "INCONCLUSIVE")
    print(f"\n  VERDICT (A) mass       : {verdict(loA, hiA)}")
    print(f"  VERDICT (B) dispersion : {verdict(loB, hiB)}")
    print(f"  engine beats market?   : "
          f"{'YES' if min(a_ev.mean(), b_ev.mean()) < book.mean() else 'NO'}")


if __name__ == "__main__":
    main()

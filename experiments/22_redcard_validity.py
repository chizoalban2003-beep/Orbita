"""22_redcard_validity.py — is Orbita's red-card intervention CALIBRATED to
reality? The first interventional-validity test.

A white-box forecast is only worth its legibility if its counterfactuals are
TRUE. So we hold the red-card lever to an objective, falsifiable standard:

  1. Ground truth. Across 10 EPL seasons, take the clean natural experiment
     — matches with exactly one red card, to exactly one team. Their actual
     H/D/A outcome frequency vs the pre-match (pre-card) market gives the
     empirical probability shift a red card causes.
  2. Orbita's claim. Re-price each match with the red_card lever — a
     momentum push toward the opponent's well — and read the shift the
     engine predicts.
  3. Verdict. Tune ONE momentum scalar on train seasons so the engine's
     mean shift matches the empirical mean, then test out-of-sample: does
     the intervention move the forecast toward the actual results (lower
     H/D/A Brier) better than the no-intervention baseline?

Design question answered empirically: STATIC scalar vs MASS-DEPENDENT
(momentum ∝ carded team's pre-match strength). A fixed push already yields
a state-dependent probability shift via the geometry (a nudge moves a tight
game near the saddle far more than a decided one), so parsimony says static
first — and we test the mass term head-to-head OOS rather than assume it.

DATA LIMIT: football-data has red-card COUNTS (HR/AR) but not the minute,
so this validates the AVERAGE-over-timing effect, not a minute-anchored
simulate_from_state. Honest, and still a real calibration test.

Run:  PYTHONPATH=src python3 experiments/22_redcard_validity.py
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita.forces import SOFTENING  # noqa: E402
from orbita.interventions import _POS  # noqa: E402

CACHE = Path.home() / ".cache" / "orbita" / "footballdata"
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
TRAIN_SEASONS = set(SEASONS[:7])
# Backtest settings: coarser than the interactive module (dt 0.1, dur 600,
# 200 trials) for thousands of forecasts. Same geometry + physics.
# dt/n_trials/sweep are env-tunable so the fine dt=0.1 calibration run can
# reuse this file without edits (ORBITA_DT=0.1 ORBITA_FINE=1 ...).
N_TRIALS = int(os.environ.get("ORBITA_NTRIALS", 120))
DT = float(os.environ.get("ORBITA_DT", 0.25))
DURATION = 600.0
IC_SCALE = 2.5
C_D = 0.04
if os.environ.get("ORBITA_FINE"):
    K_GRID = [0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.6]   # finer near optimum
else:
    K_GRID = [0.0, 0.4, 0.8, 1.2, 1.8, 2.4]
TRAIN_CAP = int(os.environ.get("ORBITA_TRAINCAP", 150))   # subsample for speed
OUTCOMES = ("home", "draw", "away")
_POSARR = np.array([_POS[k] for k in OUTCOMES], float)


def fast_forecast(priors, momentum, n_trials=N_TRIALS, seed=42):
    """Vectorised batch H/D/A forecast (same math as orbita.interventions,
    coarser dt for backtest volume). momentum = length-2 push added to p0."""
    mass = np.array([priors[k] for k in OUTCOMES], float)
    mass = mass / mass.sum()
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=np.array([0.3, 0.2]) * IC_SCALE, size=(n_trials, 2))
    p = rng.normal(scale=np.array([0.15, 0.15]) * IC_SCALE, size=(n_trials, 2)) + momentum
    soft2 = SOFTENING ** 2
    m_col = mass[None, :, None]
    n_steps = int(DURATION / DT)

    def force(qb, pb):
        r = _POSARR[None, :, :] - qb[:, None, :]
        d2 = np.einsum("twk,twk->tw", r, r) + soft2
        g = (m_col * r / d2[:, :, None] ** 1.5).sum(axis=1)
        return g - C_D * pb

    F = force(q, p)
    for _ in range(n_steps):
        p_half = p + 0.5 * DT * F
        q = q + DT * p_half
        F = force(q, p_half)
        p = p_half + 0.5 * DT * F
    diff = _POSARR[None, :, :] - q[:, None, :]
    d2 = np.einsum("twk,twk->tw", diff, diff) + soft2
    w = mass[None, :] / np.sqrt(d2) ** 2.0
    w = w / w.sum(axis=1, keepdims=True)
    pr = w.mean(axis=0)
    return {k: float(pr[i]) for i, k in enumerate(OUTCOMES)}


def parse_date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(s)


def devig3(h, d, a):
    ih, idr, ia = 1 / h, 1 / d, 1 / a
    s = ih + idr + ia
    return {"home": ih / s, "draw": idr / s, "away": ia / s}


def load():
    out = []
    for s in SEASONS:
        f = CACHE / f"E0_{s}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(f.open(encoding="utf-8-sig")):
            try:
                h, d, a = float(r["B365H"]), float(r["B365D"]), float(r["B365A"])
                hr, ar = int(r["HR"]), int(r["AR"])
                ftr = r["FTR"]
            except (KeyError, ValueError):
                continue
            if not ((hr == 1 and ar == 0) or (ar == 1 and hr == 0)):
                continue                       # clean single-card only
            carded = "home" if hr == 1 else "away"
            actual = {"H": "home", "D": "draw", "A": "away"}[ftr]
            out.append({"season": s, "priors": devig3(h, d, a),
                        "carded": carded, "actual": actual})
    return out


def onehot(o):
    return np.array([1.0 if k == o else 0.0 for k in OUTCOMES])


def prob_vec(d):
    return np.array([d[k] for k in OUTCOMES])


def brier(pred, actual):
    return float(np.sum((prob_vec(pred) - onehot(actual)) ** 2))


def redcard_momentum(carded, k):
    """Momentum vector toward the opponent's well; magnitude k."""
    opp = "away" if carded == "home" else "home"
    direction = _POS[opp] / np.linalg.norm(_POS[opp])
    return k * direction


def mean_shift(subset):
    """Empirical shift = mean(actual onehot) - mean(market prob)."""
    act = np.mean([onehot(m["actual"]) for m in subset], axis=0)
    mkt = np.mean([prob_vec(m["priors"]) for m in subset], axis=0)
    return mkt, act, act - mkt


def engine_forecast(m, k, dynamic=False):
    kk = k
    if dynamic:
        # momentum proportional to carded team's pre-match strength
        kk = k * (m["priors"][m["carded"]] / 0.38)   # 0.38 ~ mean carded p
    mom = np.zeros(2) if kk == 0.0 else redcard_momentum(m["carded"], kk)
    return fast_forecast(m["priors"], mom)


def briers(matches, k, dynamic=False):
    return np.array([brier(engine_forecast(m, k, dynamic), m["actual"])
                     for m in matches])


def bootstrap_ci(d, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def main():
    data = load()
    train = [m for m in data if m["season"] in TRAIN_SEASONS]
    ev = [m for m in data if m["season"] not in TRAIN_SEASONS]
    print("=" * 70)
    print("Red-card interventional validity — 10 EPL seasons")
    print(f"clean single-card matches: {len(data)}  "
          f"(home {sum(m['carded']=='home' for m in data)}, "
          f"away {sum(m['carded']=='away' for m in data)})  "
          f"train={len(train)} eval={len(ev)}")
    print("=" * 70)

    # ---- ground truth: the real shift a red card causes ----
    print("\nGROUND TRUTH — actual outcomes vs pre-match market:")
    for carded in ("away", "home"):
        sub = [m for m in data if m["carded"] == carded]
        mkt, act, sh = mean_shift(sub)
        print(f"  {carded} carded (n={len(sub)}):")
        print(f"    market : home {mkt[0]:.0%}  draw {mkt[1]:.0%}  away {mkt[2]:.0%}")
        print(f"    actual : home {act[0]:.0%}  draw {act[1]:.0%}  away {act[2]:.0%}")
        print(f"    SHIFT  : home {sh[0]:+.0%}  draw {sh[1]:+.0%}  away {sh[2]:+.0%}")

    # market Brier on card matches (reference)
    mkt_b = np.array([brier(m["priors"], m["actual"]) for m in ev])
    base_b = briers(ev, 0.0)
    print(f"\nEVAL references: market Brier={mkt_b.mean():.4f}  "
          f"engine baseline (no lever)={base_b.mean():.4f}")

    # ---- tune static scalar on train (subsampled for speed) ----
    tr = train[:TRAIN_CAP]
    print(f"\nSTATIC momentum — TRAIN H/D/A Brier by k (n={len(tr)}):")
    bestk, bestb = 0.0, np.inf
    for k in K_GRID:
        b = briers(tr, k).mean()
        flag = "  <=" if b < bestb else ""
        if b < bestb:
            bestb, bestk = b, k
        print(f"    k={k:<4} Brier={b:.4f}{flag}")

    # ---- eval static + dynamic ----
    stat_b = briers(ev, bestk)
    dyn_b = briers(ev, bestk, dynamic=True)
    d_stat = base_b - stat_b
    d_dyn = base_b - dyn_b
    lo_s, hi_s = bootstrap_ci(d_stat)
    lo_d, hi_d = bootstrap_ci(d_dyn)

    # what shift does the tuned engine actually predict? (away-card eval)
    away_ev = [m for m in ev if m["carded"] == "away"]
    base_pred = np.mean([prob_vec(engine_forecast(m, 0.0)) for m in away_ev], axis=0)
    tuned_pred = np.mean([prob_vec(engine_forecast(m, bestk)) for m in away_ev], axis=0)
    _, _, emp_sh = mean_shift(away_ev)

    print(f"\nEVAL (held-out last 3 seasons, n={len(ev)}), tuned k={bestk}:")
    print(f"  engine baseline (no lever) Brier : {base_b.mean():.4f}")
    print(f"  + red-card STATIC  momentum      : {stat_b.mean():.4f}  "
          f"Δ {d_stat.mean():+.4f} CI[{lo_s:+.4f},{hi_s:+.4f}]")
    print(f"  + red-card DYNAMIC (∝ mass)       : {dyn_b.mean():.4f}  "
          f"Δ {d_dyn.mean():+.4f} CI[{lo_d:+.4f},{hi_d:+.4f}]")
    print(f"\n  away-card shift toward HOME  — empirical {emp_sh[0]:+.0%}  "
          f"| engine tuned {tuned_pred[0]-base_pred[0]:+.0%}")

    def verdict(lo, hi):
        return "VALIDATED" if lo > 0 else "HURTS" if hi < 0 else "INCONCLUSIVE"
    print(f"\n  VERDICT static  : {verdict(lo_s, hi_s)}")
    print(f"  VERDICT dynamic : {verdict(lo_d, hi_d)}")
    print(f"  beats market on card matches? "
          f"{'YES' if stat_b.mean() < mkt_b.mean() else 'NO'}")


if __name__ == "__main__":
    main()

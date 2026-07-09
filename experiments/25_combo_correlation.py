"""25_combo_correlation.py — can Orbita's shared trajectory out-predict the
market's independent-leg multiplication on the Result x Over/Under 2.5 joint?

The single-outcome campaign proved the market efficient on marginals (exp 22-24).
The one structural blind spot left is CORRELATION: a bookmaker prices a combo as
P(result) x P(over) x (a blunt global correction), because it multiplies
spreadsheets. Orbita does not have marginals — it integrates a single kinetic
trajectory through a joint 6-well plane, so its result and total axes are
coupled by construction, per-match.

Baselines (the true combo price is unsourceable offline, so both are built from
single-leg CLOSING odds — see docs):
  tier-1  independence   P(r)*P(o)                         — the naive product
  tier-2  empirical copula  tier-1 * lift[r,o]             — lift fit on TRAIN,
          one GLOBAL correlation matrix = "the bookmaker's blunt matrix"
Orbita earns the edge claim only by beating BOTH out-of-sample on realized
6-cell Brier. Beating tier-1 not tier-2 = the correlation is real but a global
matrix captures it; beating tier-2 = the PER-MATCH kinetic coupling captures
dependency a static matrix cannot. (Predictive test on fair probs, not post-vig
profit.)

Run:  PYTHONPATH=src python3 experiments/25_combo_correlation.py
      env: ORBITA_GAMMA (correlation-strength scale, default 1.0),
           ORBITA_NTRIALS, ORBITA_DT
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from orbita.forces import SOFTENING  # noqa: E402

CACHE = Path.home() / ".cache" / "orbita" / "footballdata"
DIVS = ["E0","E1","D1","SP1","I1","F1"]
SEASONS = ["1920","2021","2122","2223","2324","2425"]      # carry closing O/U
TRAIN = {"1920","2021","2122","2223"}
RES = ("home","draw","away"); OU = ("over","under")
CELLS = [(r,o) for r in RES for o in OU]

# joint geometry (inherited from exp05/19). GAMMA scales the correlation the
# geometry encodes: the y-spread and the draw column's offset toward under.
WIN_X = 5.0
GAMMA = float(os.environ.get("ORBITA_GAMMA", 1.0))
OVER_Y, UNDER_Y = 3.0, -3.0
DRAW_OVER_Y, DRAW_UNDER_Y = 4.0, -2.5
C_D, ALPHA, IC_SCALE = 0.04, 2.0, 2.5
DURATION = 300.0
DT = float(os.environ.get("ORBITA_DT", 0.15))
N_TRIALS = int(os.environ.get("ORBITA_NTRIALS", 80))
SEED = 20260709


def devig(odds):
    a = np.array(odds, float)
    if np.any(a <= 1.0) or np.any(~np.isfinite(a)):
        raise ValueError("bad odds")
    inv = 1.0/a; return inv/inv.sum()


def load():
    out = []
    for div in DIVS:
        for s in SEASONS:
            f = CACHE / f"{div}_{s}.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(f.open(encoding="utf-8-sig")):
                try:
                    pr = devig([r["PSCH"], r["PSCD"], r["PSCA"]])
                    po = devig([r["PC>2.5"], r["PC<2.5"]])
                    res = {"H":"home","D":"draw","A":"away"}[r["FTR"]]
                    tot = int(r["FTHG"]) + int(r["FTAG"])
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                out.append({"s": s, "pr": {"home":pr[0],"draw":pr[1],"away":pr[2]},
                            "po": {"over":po[0],"under":po[1]},
                            "res": res, "ou": "over" if tot > 2.5 else "under"})
    return out


def onehot(m):
    return np.array([1.0 if (r==m["res"] and o==m["ou"]) else 0.0 for r,o in CELLS])


def indep_vec(m):
    return np.array([m["pr"][r]*m["po"][o] for r,o in CELLS])


def brier6(pred, m):
    return float(np.sum((np.asarray(pred) - onehot(m))**2))


# ---- the joint geometry + Orbita forecaster ------------------------------

def joint_geometry(gamma=1.0):
    """positions[6,2] for CELLS order (home/away over further out; draw column
    shifted toward under; gamma scales the y-spread = correlation strength)."""
    oy, uy = OVER_Y*gamma, UNDER_Y*gamma
    doy, duy = DRAW_OVER_Y*gamma, DRAW_UNDER_Y*gamma
    pos = {("home","over"):[WIN_X,oy], ("home","under"):[WIN_X,uy],
           ("draw","over"):[0.0,doy], ("draw","under"):[0.0,duy],
           ("away","over"):[-WIN_X,oy], ("away","under"):[-WIN_X,uy]}
    return np.array([pos[c] for c in CELLS], float)


def orbita_joint(m, pos, n_trials=N_TRIALS, seed=SEED):
    """Pre-match joint forecast: seed at origin, integrate the shared trajectory
    through the 6-well plane, soft-assign to cells. Masses = independent product
    of the marginals, so any correlation is emergent geometry/dynamics."""
    mass = indep_vec(m); mass = mass / mass.sum()
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=np.array([0.3, 0.2]) * IC_SCALE, size=(n_trials, 2))
    p = rng.normal(scale=np.array([0.15, 0.15]) * IC_SCALE, size=(n_trials, 2))
    soft2 = SOFTENING ** 2
    m_col = mass[None, :, None]
    n_steps = int(DURATION / DT)

    def force(qb, pb):
        r = pos[None, :, :] - qb[:, None, :]
        d2 = np.einsum("twk,twk->tw", r, r) + soft2
        g = (m_col * r / d2[:, :, None] ** 1.5).sum(axis=1)
        return g - C_D * pb

    F = force(q, p)
    for _ in range(n_steps):
        p_half = p + 0.5 * DT * F
        q = q + DT * p_half
        F = force(q, p_half)
        p = p_half + 0.5 * DT * F
    diff = pos[None, :, :] - q[:, None, :]
    d2 = np.einsum("twk,twk->tw", diff, diff) + soft2
    w = mass[None, :] / np.sqrt(d2) ** ALPHA
    w = w / w.sum(axis=1, keepdims=True)
    return w.mean(axis=0)


def fit_lift(train):
    jr = {c: 0.0 for c in CELLS}; mr = {r:0.0 for r in RES}; mo = {o:0.0 for o in OU}
    for m in train:
        jr[(m["res"],m["ou"])] += 1; mr[m["res"]] += 1; mo[m["ou"]] += 1
    n = len(train)
    return {c: (jr[c]/n)/((mr[c[0]]/n)*(mo[c[1]]/n)) if mr[c[0]]*mo[c[1]] > 0 else 1.0
            for c in CELLS}


def implied_lift(preds, ms):
    """Lift matrix a set of joint predictions implies (avg joint / product of
    avg marginals) — to compare the sign/shape of Orbita's correlation to real."""
    P = np.mean(preds, axis=0)
    d = {c: P[i] for i, c in enumerate(CELLS)}
    mr = {r: sum(d[(r,o)] for o in OU) for r in RES}
    mo = {o: sum(d[(r,o)] for r in RES) for o in OU}
    return {c: d[c]/(mr[c[0]]*mo[c[1]]) if mr[c[0]]*mo[c[1]] > 0 else 1.0 for c in CELLS}


def bootstrap_ci(diff, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    ms = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)]
    return float(np.percentile(ms, 5)), float(np.percentile(ms, 95))


GAMMA_GRID = [1.0, 1.2, 1.4, 1.6, 1.8]
TRAIN_TUNE_CAP = int(os.environ.get("ORBITA_TUNECAP", 700))


def main():
    d = load()
    train = [m for m in d if m["s"] in TRAIN]
    ev = [m for m in d if m["s"] not in TRAIN]
    print(f"loaded {len(d)}  train={len(train)} eval={len(ev)}  dt={DT} N={N_TRIALS}")

    lift = fit_lift(train)
    copula = lambda m: (lambda j: j/j.sum())(indep_vec(m)*np.array([lift[c] for c in CELLS]))

    # tune the correlation-strength gamma on a train subsample (leakage-free)
    rng = np.random.default_rng(0)
    tune = [train[i] for i in rng.choice(len(train), min(TRAIN_TUNE_CAP, len(train)), replace=False)]
    print(f"\nGAMMA tune on train subsample (n={len(tune)}) — 6-cell Brier:")
    best_g, best_b = 1.0, np.inf
    for g in GAMMA_GRID:
        pos = joint_geometry(g)
        b = np.mean([brier6(orbita_joint(m, pos), m) for m in tune])
        flag = "  <=" if b < best_b else ""
        if b < best_b:
            best_b, best_g = b, g
        print(f"    gamma={g:<4} Brier={b:.4f}{flag}")

    b1 = np.array([brier6(indep_vec(m), m) for m in ev])
    b2 = np.array([brier6(copula(m), m) for m in ev])
    for tag, g in [("untuned gamma=1.0", 1.0), (f"train-tuned gamma={best_g}", best_g)]:
        pos = joint_geometry(g)
        orb = [orbita_joint(m, pos) for m in ev]
        bo = np.array([brier6(orb[i], ev[i]) for i in range(len(ev))])
        oimp = implied_lift(orb, ev)
        print(f"\n=== Orbita {tag} ===")
        print(f"  correlation lift (empirical | orbita): "
              + "  ".join(f"{c[0][0]}{c[1][0]}={lift[c]:.2f}|{oimp[c]:.2f}" for c in CELLS))
        lo1, hi1 = bootstrap_ci(b1 - bo)
        lo2, hi2 = bootstrap_ci(b2 - bo)
        print(f"  EVAL 6-cell Brier (n={len(ev)}): tier1={b1.mean():.4f} "
              f"tier2={b2.mean():.4f} Orbita={bo.mean():.4f}")
        print(f"    vs tier-1: Δ {(b1-bo).mean():+.4f} CI[{lo1:+.4f},{hi1:+.4f}]  "
              f"{'BEATS indep' if lo1>0 else 'no'}")
        print(f"    vs tier-2: Δ {(b2-bo).mean():+.4f} CI[{lo2:+.4f},{hi2:+.4f}]  "
              f"{'BEATS copula = REAL per-match edge' if lo2>0 else 'ties/loses copula'}")


if __name__ == "__main__":
    main()

"""23_drag_lowtempo_validity.py — is Orbita's drag (low-tempo) lever
calibrated to reality? The second interventional-validity test.

Red cards validated the MOMENTUM vector (a directional push). Drag is the
other primitive: a global friction that bleeds kinetic energy, shrinks the
orbit and traps the state where it sits. Its distinct physical signature is
NOT a directional shift but a variance collapse — the favourite/leader
locks in, upsets fall, and the draw firms up.

Proxy for "low tempo" (no possession data in football-data): total shots
HS+AS, the most direct openness measure present every season. Low-tempo =
bottom tercile of total shots. Used as a post-hoc natural-experiment
selector, exactly like the red-card counts in exp 22 — we don't know a
match will be low-tempo pre-match; we validate that WHEN it is, the drag
lever prices it right.

HONEST HAZARD: low shots is near-tautologically tied to few goals and
draws, so "add draw probability → wins on low-shot games" would be
circular. Guards: (1) the model-free DIAGNOSTIC first — does reality show
drag's *distinct* signature (fewer upsets / favourite holds), not just more
draws? (2) the lever must beat the MARKET (which already prices each team's
expected tempo), not merely the engine baseline.

Run:  PYTHONPATH=src python3 experiments/23_drag_lowtempo_validity.py
      (add DIAG=1 for the fast ground-truth diagnostic only, no forecasts)
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
from orbita.interventions import _POS  # noqa: E402

CACHE = Path.home() / ".cache" / "orbita" / "footballdata"
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
TRAIN_SEASONS = set(SEASONS[:7])
N_TRIALS = int(os.environ.get("ORBITA_NTRIALS", 120))
DT = float(os.environ.get("ORBITA_DT", 0.25))
DURATION, IC_SCALE, BASE_CD = 600.0, 2.5, 0.04
G_GRID = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0]     # drag multipliers to sweep
TRAIN_CAP = int(os.environ.get("ORBITA_TRAINCAP", 200))
OUTCOMES = ("home", "draw", "away")
_POSARR = np.array([_POS[k] for k in OUTCOMES], float)


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
                hs, as_ = int(r["HS"]), int(r["AS"])
                fthg, ftag = int(r["FTHG"]), int(r["FTAG"])
                ftr = r["FTR"]
            except (KeyError, ValueError):
                continue
            pri = devig3(h, d, a)
            out.append({"season": s, "priors": pri, "shots": hs + as_,
                        "actual": {"H": "home", "D": "draw", "A": "away"}[ftr],
                        "goals": fthg + ftag,
                        "fav": "home" if pri["home"] >= pri["away"] else "away"})
    return out


def onehot(o):
    return np.array([1.0 if k == o else 0.0 for k in OUTCOMES])


def prob_vec(d):
    return np.array([d[k] for k in OUTCOMES])


def brier(pred, actual):
    return float(np.sum((prob_vec(pred) - onehot(actual)) ** 2))


def fast_forecast(priors, C_d, n_trials=N_TRIALS, seed=42):
    mass = np.array([priors[k] for k in OUTCOMES], float)
    mass = mass / mass.sum()
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=np.array([0.3, 0.2]) * IC_SCALE, size=(n_trials, 2))
    p = rng.normal(scale=np.array([0.15, 0.15]) * IC_SCALE, size=(n_trials, 2))
    soft2 = SOFTENING ** 2
    m_col = mass[None, :, None]
    n_steps = int(DURATION / DT)

    def force(qb, pb):
        r = _POSARR[None, :, :] - qb[:, None, :]
        d2 = np.einsum("twk,twk->tw", r, r) + soft2
        g = (m_col * r / d2[:, :, None] ** 1.5).sum(axis=1)
        return g - C_d * pb

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


def briers(ms, g):
    return np.array([brier(fast_forecast(m["priors"], BASE_CD * g), m["actual"])
                     for m in ms])


def bootstrap_ci(d, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def diagnostic(data):
    """Model-free: does reality show drag's signature by shot tercile?"""
    shots = np.array([m["shots"] for m in data])
    q1, q2 = np.quantile(shots, [1 / 3, 2 / 3])
    print("\nDIAGNOSTIC — outcomes by total-shots tercile (drag signature = "
          "low-tempo has MORE draws AND fav holds MORE / fewer upsets):")
    print(f"  {'tempo':<10}{'n':>5}{'shots':>7}{'goals':>7}"
          f"{'draw%':>8}{'mkt_draw%':>10}{'fav_win%':>10}{'mkt_fav%':>10}")
    for lo, hi, name in [(-1, q1, "low"), (q1, q2, "mid"), (q2, 1e9, "high")]:
        b = [m for m in data if lo < m["shots"] <= hi]
        draw = np.mean([m["actual"] == "draw" for m in b])
        mkt_draw = np.mean([m["priors"]["draw"] for m in b])
        favwin = np.mean([m["actual"] == m["fav"] for m in b])
        mkt_fav = np.mean([max(m["priors"]["home"], m["priors"]["away"]) for m in b])
        print(f"  {name:<10}{len(b):>5}{np.mean([m['shots'] for m in b]):>7.1f}"
              f"{np.mean([m['goals'] for m in b]):>7.2f}{draw*100:>8.1f}"
              f"{mkt_draw*100:>10.1f}{favwin*100:>10.1f}{mkt_fav*100:>10.1f}")
    return q1


def main():
    data = load()
    q1 = diagnostic(data)
    if os.environ.get("DIAG"):
        return
    low = [m for m in data if m["shots"] <= q1]      # low-tempo natural experiment
    train = [m for m in low if m["season"] in TRAIN_SEASONS][:TRAIN_CAP]
    ev = [m for m in low if m["season"] not in TRAIN_SEASONS]
    print(f"\nLOW-TEMPO lever backtest: low-tempo matches={sum(m['shots']<=q1 for m in data)} "
          f"train(sweep)={len(train)} eval={len(ev)}  dt={DT} N={N_TRIALS}")

    mkt_b = np.array([brier(m["priors"], m["actual"]) for m in ev])
    base_b = briers(ev, 1.0)
    print(f"  references (eval): market={mkt_b.mean():.4f}  baseline(g=1)={base_b.mean():.4f}")

    print("\n  drag multiplier g — TRAIN H/D/A Brier:")
    bestg, bestb = 1.0, np.inf
    for g in G_GRID:
        b = briers(train, g).mean()
        flag = "  <=" if b < bestb else ""
        if b < bestb:
            bestb, bestg = b, g
        print(f"    g={g:<4} Brier={b:.4f}{flag}")

    tuned_b = briers(ev, bestg)
    d = base_b - tuned_b
    lo, hi = bootstrap_ci(d)
    dm = mkt_b - tuned_b
    lom, him = bootstrap_ci(dm)
    print(f"\n  EVAL (held-out, n={len(ev)}), tuned g={bestg}:")
    print(f"    drag lever Brier = {tuned_b.mean():.4f}")
    print(f"    Δ vs baseline = {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]")
    print(f"    Δ vs market   = {dm.mean():+.4f} CI[{lom:+.4f},{him:+.4f}]")
    v = lambda l, h: "VALIDATED" if l > 0 else "HURTS" if h < 0 else "INCONCLUSIVE"
    print(f"    VERDICT vs baseline: {v(lo,hi)}   vs market: {v(lom,him)}")


if __name__ == "__main__":
    main()

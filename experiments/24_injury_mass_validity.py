"""24_injury_mass_validity.py — is Orbita's MASS primitive calibrated to
reality? The third interventional-validity test — and the one that isolates
mass from momentum.

Red cards validated MOMENTUM (a directional push); drag (low-tempo) was
REJECTED. This tests the remaining primitive: MASS. An injury / suspension /
rotation makes a team weaker — in Orbita that is a pure CUT to that team's
gravity-well mass (its likelihood), with NO directional vector. If a
mass-cut moves the forecast the way a real weakening does, the counterfactual
engine's three levers (mass, drag, momentum) are each independently
grounded.

SOURCING THE CONDITION. football-data has no lineups/injuries. But it carries
Pinnacle OPENING (PSH/PSD/PSA) and CLOSING (PSCH/PSCD/PSCA) odds in every
season. Pinnacle is the sharp book, so an open->close drift IS the market
re-rating a team's strength — and in Orbita, well mass *is* market-implied
strength, so a drift is a mass change in the engine's own units. We select
the natural experiment by ADVERSE DRIFT: matches where a team's Pinnacle
win-prob dropped >= THR open->close. (Premise-gated: at THR=0.05 the OPENING
price over-rates such teams by ~7.8pts vs actual, and CLOSING captures it.)

HONEST HAZARDS + GUARDS (this is a re-rating proxy, not "injury" narrowly,
and sizing a cut to each match's drift would just relabel the closing line):
  * The lever is ONE STATIC scalar tuned on train seasons — never per-match
    fitted to the drift (that would peek at closing).
  * Scored from OPENING masses against ACTUAL results, out-of-sample.
  * PLACEBO control: the same tuned cut applied to no-drift matches must NOT
    help.
  * WRONG-TEAM falsification: cutting the team that SHORTENED (strengthened)
    must HURT.
Baseline to beat = the OPENING market (news-naive), the fair analogue of the
pre-card market in exp22. The CLOSING market (news-aware) is the ceiling.

Run:  PYTHONPATH=src python3 experiments/24_injury_mass_validity.py
      (DIAG=1 prints the ground-truth table only; env: ORBITA_THR, ORBITA_DT,
       ORBITA_NTRIALS)
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
SEASONS = ["1516","1617","1718","1819","1920","2021","2122","2223","2324","2425"]
TRAIN_SEASONS = set(SEASONS[:7])
N_TRIALS = int(os.environ.get("ORBITA_NTRIALS", 150))
DT = float(os.environ.get("ORBITA_DT", 0.1))
THR = float(os.environ.get("ORBITA_THR", 0.05))
# TRANSFER=1 re-specs the lever: freed mass goes to the OPPONENT well (result
# axis), not renormalized symmetrically — the diagnostic showed a plain cut
# leaks ~half the mass into the central draw well, which reality does not do.
TRANSFER = bool(os.environ.get("ORBITA_TRANSFER"))
PLACEBO_CAP = int(os.environ.get("ORBITA_PLACEBO_CAP", 10_000))
DURATION, IC_SCALE, C_D = 600.0, 2.5, 0.04
C_GRID = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75]   # fraction to cut weak well mass
OUTCOMES = ("home", "draw", "away")
_POSARR = np.array([_POS[k] for k in OUTCOMES], float)
_IDX = {"home": 0, "away": 2}


def devig(h, d, a):
    ih, idr, ia = 1/h, 1/d, 1/a
    s = ih + idr + ia
    return np.array([ih/s, idr/s, ia/s])


def load():
    out = []
    for s in SEASONS:
        f = CACHE / f"E0_{s}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(f.open(encoding="utf-8-sig")):
            try:
                op = devig(float(r["PSH"]), float(r["PSD"]), float(r["PSA"]))
                cl = devig(float(r["PSCH"]), float(r["PSCD"]), float(r["PSCA"]))
                res = {"H":"home","D":"draw","A":"away"}[r["FTR"]]
            except (KeyError, ValueError, ZeroDivisionError):
                continue
            drift = cl - op
            weak = "home" if drift[0] <= drift[2] else "away"
            out.append({"season": s, "op": op, "cl": cl, "res": res,
                        "weak": weak, "strong": "away" if weak == "home" else "home",
                        "weak_drop": -min(drift[0], drift[2]),
                        "max_abs": max(abs(drift[0]), abs(drift[2]))})
    return out


def onehot(o):
    return np.array([1.0 if k == o else 0.0 for k in OUTCOMES])


def brier(p, res):
    return float(np.sum((np.asarray(p) - onehot(res))**2))


def cut_mass(op, side, c):
    """Mass vector with `side` well cut by fraction c. If TRANSFER, the freed
    mass is added to the OPPONENT well (result-axis transfer, draw untouched);
    otherwise it is redistributed symmetrically by renormalization."""
    m = op.astype(float).copy()
    removed = m[_IDX[side]] * c
    m[_IDX[side]] -= removed
    if TRANSFER:
        opp = "away" if side == "home" else "home"
        m[_IDX[opp]] += removed
    return m / m.sum()


def fast_forecast(mass, n_trials=N_TRIALS, seed=42):
    """Vectorised H/D/A forecast. `mass` drives BOTH the gravity field and the
    posterior weighting — cutting a well's mass is the faithful Mass lever.
    Same seed across baseline/lever so the paired Brier difference cancels MC
    noise."""
    mass = np.asarray(mass, float)
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
    return pr


def briers_cut(ms, side_key, c):
    return np.array([brier(fast_forecast(cut_mass(m["op"], m[side_key], c)), m["res"])
                     for m in ms])


def bootstrap_ci(d, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def ground_truth(data):
    print("\nGROUND TRUTH — weakened team (adverse Pinnacle drift) vs its OPENING price:")
    print(f"  {'thr':>5}{'n':>6}{'op_imp%':>9}{'cl_imp%':>9}{'actual%':>9}{'op_gap':>8}")
    for thr in (0.02, 0.03, 0.05, 0.08):
        sub = [m for m in data if m["weak_drop"] >= thr]
        if not sub:
            continue
        i = lambda m: _IDX[m["weak"]]
        op = np.mean([m["op"][i(m)] for m in sub])
        cl = np.mean([m["cl"][i(m)] for m in sub])
        ac = np.mean([1.0 if m["res"] == m["weak"] else 0.0 for m in sub])
        print(f"  {thr:>5}{len(sub):>6}{op*100:>9.1f}{cl*100:>9.1f}{ac*100:>9.1f}{(ac-op)*100:>+8.1f}")


def main():
    data = load()
    ground_truth(data)
    if os.environ.get("DIAG"):
        return

    drift = [m for m in data if m["weak_drop"] >= THR]
    stable = [m for m in data if m["max_abs"] < 0.01][:PLACEBO_CAP]   # placebo pool
    print(f"\n{'*** RESULT-AXIS TRANSFER lever ***' if TRANSFER else '*** symmetric mass-cut lever ***'}")
    train = [m for m in drift if m["season"] in TRAIN_SEASONS]
    ev = [m for m in drift if m["season"] not in TRAIN_SEASONS]
    print(f"\nMASS-CUT lever backtest  THR={THR} dt={DT} N={N_TRIALS}")
    print(f"  adverse-drift matches={len(drift)}  train={len(train)} eval={len(ev)}"
          f"  placebo(stable)={len(stable)}")

    op_b = np.array([brier(m["op"], m["res"]) for m in ev])
    cl_b = np.array([brier(m["cl"], m["res"]) for m in ev])
    base_b = briers_cut(ev, "weak", 0.0)      # engine, no cut
    print(f"  references (eval): market OPEN={op_b.mean():.4f}  "
          f"market CLOSE(ceiling)={cl_b.mean():.4f}  engine baseline={base_b.mean():.4f}")

    print(f"\n  mass-cut fraction c — TRAIN weakened-team Brier (n={len(train)}):")
    bestc, bestb = 0.0, np.inf
    for c in C_GRID:
        b = briers_cut(train, "weak", c).mean()
        flag = "  <=" if b < bestb else ""
        if b < bestb:
            bestb, bestc = b, c
        print(f"    c={c:<5} Brier={b:.4f}{flag}")

    # ---- EVAL: validity, placebo, wrong-team ----
    tuned_b = briers_cut(ev, "weak", bestc)
    d_base = base_b - tuned_b
    lo, hi = bootstrap_ci(d_base)

    # placebo: same tuned cut applied to the would-be-weak side of STABLE
    # (no-drift) matches — a genuine weakening signal should not live here.
    plac_base = np.array([brier(fast_forecast(m["op"]), m["res"]) for m in stable])
    plac_cut = np.array([brier(fast_forecast(cut_mass(m["op"], m["weak"], bestc)), m["res"])
                         for m in stable])
    d_plac = plac_base - plac_cut
    lop, hip = bootstrap_ci(d_plac)

    # wrong-team: cut the STRENGTHENED team on the drift eval matches (should hurt)
    wrong_b = briers_cut(ev, "strong", bestc)
    d_wrong = base_b - wrong_b
    low, hiw = bootstrap_ci(d_wrong)

    print(f"\n  EVAL (held-out, n={len(ev)}), tuned c={bestc}:")
    print(f"    mass-cut lever Brier = {tuned_b.mean():.4f}")
    print(f"    Δ vs engine baseline = {d_base.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]")
    print(f"    lever vs market OPEN  = {op_b.mean()-tuned_b.mean():+.4f}  "
          f"(reached {(op_b.mean()-tuned_b.mean())/(op_b.mean()-cl_b.mean()+1e-9)*100:.0f}% "
          f"of the open->close ceiling gap)")
    print(f"\n  CONTROLS:")
    print(f"    PLACEBO  (cut on stable matches, n={len(stable)}): "
          f"Δ {d_plac.mean():+.4f} CI[{lop:+.4f},{hip:+.4f}]  (want ~0 / not helpful)")
    print(f"    WRONG-TEAM (cut the strengthened side): "
          f"Δ {d_wrong.mean():+.4f} CI[{low:+.4f},{hiw:+.4f}]  (want < 0 / hurts)")

    def verdict(l, h):
        return "VALIDATED" if l > 0 else "HURTS" if h < 0 else "INCONCLUSIVE"
    print(f"\n  VERDICT (lever vs baseline): {verdict(lo, hi)}")
    print(f"  placebo differential clean? "
          f"{'YES' if d_base.mean() > d_plac.mean() + 0.002 else 'NO/weak'}")
    print(f"  wrong-team falsification hurts? {'YES' if hiw < 0 else 'NO'}")


if __name__ == "__main__":
    main()

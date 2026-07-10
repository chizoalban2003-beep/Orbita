"""26_precalibrate.py — train the lever constants on PAST data, before a single
live read exists. Answers "can it learn before the season starts?": yes — the
engine's red-card magnitude is a physical constant we can fit on 10 seasons of
natural experiments right now, producing the Bayesian prior the live ledger then
refines.

Natural experiment: clean single-red-card matches (HR/AR), Pinnacle closing 1X2
as the pre-match belief, actual result as ground truth. We fold the Bayesian
likelihood update (orbita.calibrate) over these matches → a posterior over the
red-card lever magnitude.

Run:  PYTHONPATH=src python3 experiments/26_precalibrate.py
      env: ORBITA_DIVS, ORBITA_NCAL (subsample), ORBITA_NTRIALS
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from orbita.calibrate import calibrate_from_matches, historical_prior  # noqa: E402

CACHE = Path.home() / ".cache" / "orbita" / "footballdata"
DIVS = os.environ.get("ORBITA_DIVS", "E0,E1,D1,SP1,I1,F1").split(",")
SEASONS = ["1516","1617","1718","1819","1920","2021","2122","2223","2324","2425"]
NCAL = int(os.environ.get("ORBITA_NCAL", 60))
NTR = int(os.environ.get("ORBITA_NTRIALS", 60))
LEVER = os.environ.get("ORBITA_LEVER", "red_card")   # red_card | injury
THR = float(os.environ.get("ORBITA_THR", 0.05))      # injury: adverse-drift cut


def devig(h, d, a):
    inv = np.array([1/h, 1/d, 1/a]); inv /= inv.sum()
    return {"home": inv[0], "draw": inv[1], "away": inv[2]}


def load_redcards():
    out = []
    for div in DIVS:
        for s in SEASONS:
            f = CACHE / f"{div}_{s}.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(f.open(encoding="utf-8-sig")):
                try:
                    hr, ar = int(r["HR"]), int(r["AR"])
                    if not ((hr == 1 and ar == 0) or (ar == 1 and hr == 0)):
                        continue
                    pri = devig(float(r["PSCH"]), float(r["PSCD"]), float(r["PSCA"]))
                    res = {"H":"home","D":"draw","A":"away"}[r["FTR"]]
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                out.append({"priors": pri, "side": "home" if hr == 1 else "away",
                            "result": res})
    return out


def load_injury_drift():
    """Natural experiment for the injury/re-rating lever: the team whose
    Pinnacle win-prob dropped >= THR open->close is the weakened side (exp24)."""
    out = []
    for div in DIVS:
        for s in SEASONS:
            f = CACHE / f"{div}_{s}.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(f.open(encoding="utf-8-sig")):
                try:
                    op = devig(float(r["PSH"]), float(r["PSD"]), float(r["PSA"]))
                    cl = devig(float(r["PSCH"]), float(r["PSCD"]), float(r["PSCA"]))
                    res = {"H":"home","D":"draw","A":"away"}[r["FTR"]]
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                dh, da = cl["home"] - op["home"], cl["away"] - op["away"]
                weak = "home" if dh <= da else "away"
                if -min(dh, da) < THR:
                    continue
                out.append({"priors": op, "side": weak, "result": res})
    return out


def main():
    m = load_injury_drift() if LEVER == "injury" else load_redcards()
    rng = np.random.default_rng(0)
    if len(m) > NCAL:
        m = [m[i] for i in rng.choice(len(m), NCAL, replace=False)]
    print(f"{LEVER} natural experiments: {len(m)} matches  (N={NTR}/forecast)")

    prior = historical_prior(LEVER)
    print("\nPRIOR  (from the campaign, pre-data):")
    print("  " + prior.summary())

    post = calibrate_from_matches(LEVER, m, n_trials=NTR)
    print("\nPOSTERIOR  (trained on past data):")
    print("  " + post.summary())

    p = post.probs()
    print("\n  magnitude grid | posterior mass")
    for t, pm in zip(post.grid, p):
        bar = "█" * int(round(pm / p.max() * 22))
        print(f"    {t:.3f}  {pm:5.3f}  {bar}")
    print(f"\n  → the live ledger will refine this posterior one settled read at a time.")


if __name__ == "__main__":
    main()

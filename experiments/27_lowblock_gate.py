"""27_lowblock_gate.py — model-free premise gate for the "heavy-favourite
low-block draw" hypothesis. NO engine, NO lever: just asks whether the *closing*
line misprices draws (and favourites/underdogs) in the regime the hypothesis
names, using the real tradeable price. If a bucket's flat-stake ROI at the
closing odds is positive with a bootstrap CI clearing zero, there is an unpriced
edge worth turning into a pre-match scenario; otherwise the market prices it and
we leave the engine frozen.

Hypothesis (user): an extreme AWAY favourite meeting a low-scoring / compact
HOME underdog yields a draw-heavy distribution the closing line fails to price.

Method:
  * closing devig (PSCH/PSCD/PSCA) = the market's fair belief per match.
  * flat-stake ROI of backing outcome O in a bucket = mean over matches of
    (O happened ? odds_O - 1 : -1). This is the honest money number: >0 means
    backing O at the close was +EV in that bucket. Bootstrap 95% CI.
  * PRIMARY gate: bucket by favourite strength × favourite side → draw ROI.
  * REFINEMENT: within extreme away-favourite matches, split the HOME underdog
    by a LEAKAGE-FREE expanding "low-scoring/compact" proxy (mean total goals in
    the team's prior fixtures) and re-test the draw ROI.

Run:  PYTHONPATH=src python3 experiments/27_lowblock_gate.py
      env: ORBITA_DIVS, ORBITA_FAVCUT (extreme-fav implied cut, def 0.55),
           ORBITA_MINGAMES (proxy warmup, def 5), ORBITA_BOOT (def 3000)
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "orbita" / "footballdata"
DIVS = os.environ.get("ORBITA_DIVS", "E0,E1,D1,SP1,I1,F1").split(",")
SEASONS = ["1516","1617","1718","1819","1920","2021","2122","2223","2324","2425"]
FAVCUT = float(os.environ.get("ORBITA_FAVCUT", 0.55))
MINGAMES = int(os.environ.get("ORBITA_MINGAMES", 5))
BOOT = int(os.environ.get("ORBITA_BOOT", 3000))
RNG = np.random.default_rng(0)


def devig(h, d, a):
    inv = np.array([1/h, 1/d, 1/a]); inv /= inv.sum()
    return inv  # [home, draw, away]


def _date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def load():
    """Return matches with closing implied probs + a leakage-free 'compact'
    proxy for each team (expanding mean total goals over prior fixtures)."""
    rows = []
    for div in DIVS:
        for s in SEASONS:
            f = CACHE / f"{div}_{s}.csv"
            if not f.exists():
                continue
            recs = []
            for r in csv.DictReader(f.open(encoding="utf-8-sig")):
                try:
                    h, d, a = float(r["PSCH"]), float(r["PSCD"]), float(r["PSCA"])
                    fthg, ftag = int(r["FTHG"]), int(r["FTAG"])
                    res = r["FTR"]
                    if min(h, d, a) <= 1.0 or res not in ("H", "D", "A"):
                        continue
                    dt = _date(r["Date"])
                except (KeyError, ValueError):
                    continue
                recs.append({"div": div, "season": s, "dt": dt,
                             "home": r["HomeTeam"], "away": r["AwayTeam"],
                             "odds": (h, d, a), "imp": devig(h, d, a),
                             "res": res, "tot": fthg + ftag})
            # order by date within season, build expanding per-team total-goals mean
            recs.sort(key=lambda x: (x["dt"] or datetime.min))
            hist = defaultdict(list)          # team -> list of prior total goals
            for m in recs:
                for who in ("home", "away"):
                    t = m[who]
                    pri = hist[t]
                    m[f"proxy_{who}"] = (float(np.mean(pri)) if len(pri) >= MINGAMES
                                         else None)
                for t in (m["home"], m["away"]):
                    hist[t].append(m["tot"])
                rows.append(m)
    return rows


def roi(matches, leg):
    """Flat-stake ROI of backing `leg` (0=home,1=draw,2=away) at closing odds,
    with a bootstrap 95% CI. Returns (mean, lo, hi, n, hit_rate, mean_implied)."""
    if not matches:
        return None
    idx = {"H": 0, "D": 1, "A": 2}
    pay = np.array([(m["odds"][leg] - 1.0) if idx[m["res"]] == leg else -1.0
                    for m in matches])
    imp = np.array([m["imp"][leg] for m in matches])
    hit = np.array([1.0 if idx[m["res"]] == leg else 0.0 for m in matches])
    n = len(pay)
    boot = np.array([pay[RNG.integers(0, n, n)].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (pay.mean(), lo, hi, n, hit.mean(), imp.mean())


def show(label, matches, leg=1):
    r = roi(matches, leg)
    if r is None or r[3] == 0:
        print(f"  {label:<34}     n=0")
        return
    mean, lo, hi, n, hit, imp = r
    edge = "  <== EDGE" if lo > 0 else ""
    print(f"  {label:<34} n={n:>5}  hit {hit*100:4.1f}%  impl {imp*100:4.1f}%  "
          f"ROI {mean*100:+5.1f}% [{lo*100:+5.1f},{hi*100:+5.1f}]{edge}")


def main():
    ms = load()
    print(f"loaded {len(ms)} matches ({','.join(DIVS)} x {len(SEASONS)} seasons)\n")

    # baseline: blindly backing the draw everywhere (the vig floor)
    print("BASELINE — back the DRAW indiscriminately:")
    show("all matches", ms, 1)

    print("\nPRIMARY GATE — draw ROI by favourite strength x side:")
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.85), (0.85, 1.01)]
    for side, si in (("HOME fav", 0), ("AWAY fav", 2)):
        for lo, hi in bins:
            sub = [m for m in ms
                   if m["imp"][si] == max(m["imp"]) and lo <= m["imp"][si] < hi]
            show(f"{side}  impl[{lo:.2f},{hi:.2f})", sub, 1)
        print()

    # user's exact hypothesis: extreme AWAY favourite (home = underdog)
    extreme = [m for m in ms if m["imp"][2] == max(m["imp"]) and m["imp"][2] >= FAVCUT]
    print(f"REFINEMENT — extreme AWAY favourite (away impl >= {FAVCUT}), "
          f"n={len(extreme)}:")
    show("  all extreme away-fav", extreme, 1)
    # split the HOME underdog by the leakage-free compact/low-scoring proxy
    have = [m for m in extreme if m.get("proxy_home") is not None]
    if have:
        med = float(np.median([m["proxy_home"] for m in have]))
        low = [m for m in have if m["proxy_home"] <= med]   # compact / low-event home
        hig = [m for m in have if m["proxy_home"] > med]
        print(f"  (home-underdog prior avg total goals; median split @ {med:.2f})")
        show("  LOW-scoring/compact home", low, 1)
        show("  higher-scoring home", hig, 1)

    print("\nCONTROL — favourite-longshot bias (back the FAVOURITE / UNDERDOG):")
    # favourite / underdog ROI need a per-match leg; compute directly
    for name, pick in (("back the FAVOURITE", "fav"), ("back the UNDERDOG", "dog")):
        sub = [m for m in ms if max(m["imp"]) >= 0.70]
        pay = []
        for m in sub:
            leg = int(np.argmax(m["imp"])) if pick == "fav" else int(np.argmin(m["imp"]))
            idx = {"H": 0, "D": 1, "A": 2}[m["res"]]
            pay.append((m["odds"][leg] - 1.0) if idx == leg else -1.0)
        pay = np.array(pay); n = len(pay)
        boot = np.array([pay[RNG.integers(0, n, n)].mean() for _ in range(BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        edge = "  <== EDGE" if lo > 0 else ""
        print(f"  {name:<34} n={n:>5}  ROI {pay.mean()*100:+5.1f}% "
              f"[{lo*100:+5.1f},{hi*100:+5.1f}]{edge}")


if __name__ == "__main__":
    main()

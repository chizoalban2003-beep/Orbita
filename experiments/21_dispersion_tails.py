"""21_dispersion_tails.py — does a team-structure proxy predict exploitable
overdispersion in the TAILS, across 10 EPL seasons?

Premise gate for the Orbita "state inertia" idea. Exp 20 showed a
dispersion knob can't beat the market on O/U 2.5 — but 2.5 is the
median-line, the least dispersion-sensitive market ("the median trap").
Dispersion mispricing lives in the tails (Over 3.5, Under 1.5) and BTTS.

Before porting a goals-ladder into Orbita's geometry, test the PREMISE
with a minimal model: is there a leakage-free team-structure proxy such
that modelling per-match dispersion beats a Poisson extrapolation on
realised tail outcomes, out-of-sample across seasons? If not, no engine
geometry can exploit it.

Baseline (the "market assumes Poisson" null): from the de-vigged B365 O/U
2.5 line, invert the implied mean total goals lambda under Poisson, then
read the tail probabilities Poisson(lambda) implies. football-data has no
1.5/3.5 odds, so this Poisson extrapolation is the honest baseline.

Treatment: same mean lambda, but a Negative-Binomial dispersion whose size
r is modulated per match by a proxy — volatile matchups get fatter tails.
    r = r0 * exp(-s * z_proxy)          (r -> inf recovers Poisson)

Two proxies (the question raised: raw goals vs an xG surrogate). No xG in
multi-season football-data, so the "deeper" metric is a shots-on-target
xG surrogate (0.3 * SoT):
    goalvar : Var(prior total goals)      of the two teams, summed
    sotxgvar: Var(prior 0.3*SoT totals)   of the two teams, summed (less noisy)
Both expanding (prior matches only, across the full 10-season timeline),
z-scored on TRAIN seasons. Train on the first 7 seasons, evaluate on the
last 3.

Run:  PYTHONPATH=src python3 experiments/21_dispersion_tails.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import nbinom, poisson

CACHE = Path.home() / ".cache" / "orbita" / "footballdata"
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
TRAIN_SEASONS = set(SEASONS[:7])          # first 7 train, last 3 eval
MIN_GAMES = 6
XG_PER_SOT = 0.30


def parse_date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(s)


def devig3(h, d, a):
    ih, id_, ia = 1 / h, 1 / d, 1 / a
    s = ih + id_ + ia
    return ih / s, id_ / s, ia / s


def lambda_from_under25(p_under):
    """Invert Poisson so P(total <= 2) == p_under (monotone decreasing in lam)."""
    lo, hi = 0.05, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if poisson.cdf(2, mid) > p_under:   # too few goals -> raise lambda
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nb_pr(lam, r):
    """(P(X>=4), P(X<=1)) for NB with mean lam, size r."""
    p = r / (r + lam)
    return float(1 - nbinom.cdf(3, r, p)), float(nbinom.cdf(1, r, p))


def pois_pr(lam):
    return float(1 - poisson.cdf(3, lam)), float(poisson.cdf(1, lam))


def load():
    hist_g = defaultdict(list)          # team -> prior total goals
    hist_x = defaultdict(list)          # team -> prior total SoT-xG
    rows = []
    for s in SEASONS:
        f = CACHE / f"E0_{s}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(f.open(encoding="utf-8-sig")):
            try:
                date = parse_date(r["Date"])
                h, d, a = float(r["B365H"]), float(r["B365D"]), float(r["B365A"])
                o, u = float(r["B365>2.5"]), float(r["B365<2.5"])
                fthg, ftag = int(r["FTHG"]), int(r["FTAG"])
                hst, ast = int(r["HST"]), int(r["AST"])
            except (KeyError, ValueError):
                continue
            rows.append((s, date, r["HomeTeam"], r["AwayTeam"], h, d, a, o, u,
                         fthg, ftag, hst, ast))
    rows.sort(key=lambda x: x[1])
    out = []
    for s, date, home, away, h, d, a, o, u, fthg, ftag, hst, ast in rows:
        io, iu = 1 / o, 1 / u
        p_under = iu / (io + iu)
        lam = lambda_from_under25(p_under)
        tot = fthg + ftag
        gh, ga = hist_g[home], hist_g[away]
        xh, xa = hist_x[home], hist_x[away]
        if min(len(gh), len(ga)) >= MIN_GAMES:
            goalvar = float(np.var(gh) + np.var(ga))
            sotxgvar = float(np.var(xh) + np.var(xa))
            # Fano factor (Var/Mean): dispersion ISOLATED from scoring level.
            # Poisson => 1; >1 genuine overdispersion. The de-confounded proxy.
            fano = float(np.var(gh) / (np.mean(gh) + 1e-9)
                         + np.var(ga) / (np.mean(ga) + 1e-9))
        else:
            goalvar = sotxgvar = fano = np.nan
        out.append({
            "season": s, "lam": lam, "total": tot,
            "over35": tot > 3.5, "under15": tot < 1.5,
            "btts": (fthg > 0 and ftag > 0),
            "goalvar": goalvar, "sotxgvar": sotxgvar, "fano": fano,
        })
        hist_g[home].append(tot);  hist_g[away].append(tot)
        xg = XG_PER_SOT * (hst + ast)
        hist_x[home].append(xg);   hist_x[away].append(xg)
    return out


def brier(p, y):
    return (p - (1.0 if y else 0.0)) ** 2 + ((1 - p) - (0.0 if y else 1.0)) ** 2


def bootstrap_ci(d, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def zscorer(train_vals):
    v = np.array([x for x in train_vals if not np.isnan(x)])
    mu, sd = v.mean(), v.std() + 1e-9
    return lambda x: 0.0 if np.isnan(x) else (x - mu) / sd


def main():
    data = load()
    train = [m for m in data if m["season"] in TRAIN_SEASONS]
    ev = [m for m in data if m["season"] not in TRAIN_SEASONS]
    print("=" * 70)
    print("Dispersion in the TAILS — 10 EPL seasons")
    print(f"matches={len(data)} train={len(train)} eval={len(ev)}  "
          f"(train {sorted(TRAIN_SEASONS)})")
    base_rate = np.mean([m["total"] for m in data])
    print(f"mean total goals={base_rate:.3f}  over3.5={np.mean([m['over35'] for m in data]):.3f}  "
          f"under1.5={np.mean([m['under15'] for m in data]):.3f}  "
          f"btts={np.mean([m['btts'] for m in data]):.3f}")
    print("=" * 70)

    # ---- model-free diagnostic: do high-proxy matches have fatter tails? ----
    cov = [m for m in data if not np.isnan(m["goalvar"])]
    for pname in ("goalvar", "fano"):
        q = np.quantile([m[pname] for m in cov], [0.25, 0.5, 0.75])
        print(f"\nDIAGNOSTIC — realised tail rates by {pname} quartile:")
        print("  quartile   n     over3.5   under1.5   base_rate  "
              "(overdispersion = BOTH tails fatter in Q4)")
        for lo, hi, name in [(-1e9, q[0], "Q1 low"), (q[0], q[1], "Q2"),
                             (q[1], q[2], "Q3"), (q[2], 1e18, "Q4 high")]:
            b = [m for m in cov if lo < m[pname] <= hi]
            print(f"  {name:<9} {len(b):<5} {np.mean([m['over35'] for m in b]):.3f}     "
                  f"{np.mean([m['under15'] for m in b]):.3f}      "
                  f"{np.mean([m['total'] for m in b]):.2f}")

    # ---- Poisson baseline tail Brier on eval ----
    def pois_tailbrier(ms):
        b35 = np.array([brier(pois_pr(m["lam"])[0], m["over35"]) for m in ms])
        b15 = np.array([brier(pois_pr(m["lam"])[1], m["under15"]) for m in ms])
        return b35, b15

    pb35, pb15 = pois_tailbrier(ev)
    print(f"\nPOISSON baseline (eval): over3.5 Brier={pb35.mean():.4f}  "
          f"under1.5 Brier={pb15.mean():.4f}  sum={pb35.mean()+pb15.mean():.4f}")

    # ---- train NB dispersion per proxy, eval out-of-sample ----
    R0_GRID = [3.0, 5.0, 8.0, 12.0, 20.0]
    S_GRID = [0.0, 0.15, 0.30, 0.50, 0.75]
    for proxy in ("goalvar", "sotxgvar", "fano"):
        z = zscorer([m[proxy] for m in train])

        def nb_tailbrier(ms, r0, s):
            b35, b15 = [], []
            for m in ms:
                r = r0 * np.exp(-s * z(m[proxy]))
                pr35, pr15 = nb_pr(m["lam"], r)
                b35.append(brier(pr35, m["over35"]))
                b15.append(brier(pr15, m["under15"]))
            return np.array(b35), np.array(b15)

        best, bestk = np.inf, (12.0, 0.0)
        for r0 in R0_GRID:
            for s in S_GRID:
                a, b = nb_tailbrier(train, r0, s)
                tot = a.mean() + b.mean()
                if tot < best:
                    best, bestk = tot, (r0, s)
        r0, s = bestk
        eb35, eb15 = nb_tailbrier(ev, r0, s)
        d = (pb35 + pb15) - (eb35 + eb15)      # positive = NB better
        lo, hi = bootstrap_ci(d)
        verdict = ("HELPS" if lo > 0 else "HURTS" if hi < 0 else "INCONCLUSIVE")
        print(f"\nNB dispersion proxy={proxy}  trained r0={r0} s={s}")
        print(f"  eval over3.5 Brier={eb35.mean():.4f}  under1.5 Brier={eb15.mean():.4f}"
              f"  sum={eb35.mean()+eb15.mean():.4f}")
        print(f"  improvement vs Poisson (sum) = {d.mean():+.4f}  "
              f"90% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")


if __name__ == "__main__":
    main()

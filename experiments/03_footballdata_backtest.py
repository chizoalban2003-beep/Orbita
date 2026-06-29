"""03_footballdata_backtest.py — v0.3.3 vs real bookmaker closing odds.

The 02_understat_backtest experiment turned up a critical finding: the
``h_w/h_d/h_l`` fields Understat embeds in its match pages are NOT
pre-match bookmaker odds. They are *post-match xG-derived outcome
probabilities*. Two diagnostics from that panel:

  - Bournemouth vs Arsenal (2024-10-19): Understat gives Bournemouth
    p=0.76 at home. Pre-match Arsenal was a heavy favourite. Bournemouth
    won 2-0 on the day, and the xG simulation reflects that — leaking
    the outcome into the "baseline."
  - Brighton vs Man City (2024-11-09): Understat gives City p=0.33 away.
    No bookmaker priced City below ~65% pre-match. Brighton won 2-1.

Using xG-derived probabilities as the bookmaker baseline makes the
comparison meaningless because the baseline is already conditioned on
what happened. So this experiment switches to football-data.co.uk's
free historical CSVs, which carry Bet365 *closing* odds for every EPL
fixture. Closing odds are the cleanest pre-match consensus available
without a paid feed.

Pipeline:

  1. Read the 2024/25 EPL CSV
  2. De-vig the (B365H, B365D, B365A) triplet → pre-match priors
  3. Filter to the date window matching the Understat panel
  4. Pull ClubElo for each match-date → roster strengths
  5. Run four forecasts (bookmaker / engine+priors / engine+roster /
     engine+calibrated) and score Brier against actual outcomes

Run with::

    PYTHONPATH=src python3 experiments/03_footballdata_backtest.py
"""
from __future__ import annotations

import csv
import sys
import urllib.request
import warnings
from collections import defaultdict
from datetime import datetime
from math import log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import (  # noqa: E402
    Body,
    Player,
    Roster,
    blend,
    build_space,
    final_well_posterior,
    fit_alpha,
    loocv_alpha,
    simulate,
)

warnings.filterwarnings("ignore", message=r".*Renormalizing.*")

CSV_URL = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
CACHE_FILE = Path.home() / ".cache" / "orbita" / "footballdata" / "E0_2425.csv"

# ClubElo for the roster layer. http://api.clubelo.com/<YYYY-MM-DD> returns
# a CSV of all clubs' Elo on that date. We cache one CSV per match-date.
CLUBELO_URL = "http://api.clubelo.com/{date}"
CLUBELO_CACHE_DIR = Path.home() / ".cache" / "orbita" / "clubelo"

# football-data short names → ClubElo names. Only entries that differ.
ELO_NAME_FIX = {
    "Nott'm Forest": "Forest",
}

# Match the Understat panel window for direct comparison.
DATE_FROM = datetime(2024, 10, 19)
DATE_TO = datetime(2024, 11, 25)

# Reduced from 200 → 50. The 13-event panel showed N=40 was already
# stable; 200 was over-engineered. 50 keeps Monte Carlo noise <1% per
# probability and the panel runs in under two minutes.
N_TRIALS = 50
SEED = 20260627
DT = 0.1
SOFT_ALPHA = 2.0


def fetch_csv() -> Path:
    """Download the season CSV and cache it on disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists():
        print(f"Fetching {CSV_URL} ...")
        urllib.request.urlretrieve(CSV_URL, CACHE_FILE)
    return CACHE_FILE


def fetch_clubelo(date_str: str) -> dict:
    """Return {club_name: elo} for ``date_str`` (YYYY-MM-DD).

    Caches one CSV per date in CLUBELO_CACHE_DIR so re-runs are offline.
    """
    CLUBELO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CLUBELO_CACHE_DIR / f"{date_str}.csv"
    if not cache_path.exists():
        urllib.request.urlretrieve(
            CLUBELO_URL.format(date=date_str), cache_path
        )
    elos = {}
    with cache_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                elos[row["Club"]] = float(row["Elo"])
            except (KeyError, ValueError):
                continue
    return elos


def elo_for(team: str, date_str: str, cache: dict) -> float:
    """Look up ``team``'s Elo on ``date_str``, normalising team names
    between football-data and ClubElo. Returns a rating in the 0-100
    range expected by Player by scaling raw Elo (typically 1500-2100)."""
    if date_str not in cache:
        cache[date_str] = fetch_clubelo(date_str)
    elos = cache[date_str]
    name = ELO_NAME_FIX.get(team, team)
    if name not in elos:
        raise KeyError(f"{name!r} not found in ClubElo for {date_str}")
    raw = elos[name]
    # Map Elo (≈1500-2100) → 0-100 roster scale. Anchor 1500 → 50,
    # 2100 → 100 so a top-of-Europe club rounds to ~100 and a relegation
    # candidate sits near 50. The relative ordering is what the roster
    # layer cares about, not the absolute scale.
    return max(0.0, min(100.0, 50.0 + (raw - 1500.0) / 12.0))


def _slug(team: str) -> str:
    """football-data uses short names; reuse the same slug rules as
    Understat so the labels are comparable."""
    s = team.lower().replace("&", "and")
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def devig(h: float, d: float, a: float) -> tuple:
    """Convert (home, draw, away) decimal odds to a normalised probability
    triplet. Strips the bookmaker overround proportionally — the
    'standard' method that football-data and the betting research
    literature default to."""
    ph = 1.0 / h
    pd = 1.0 / d
    pa = 1.0 / a
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def parse_date(s: str) -> datetime:
    """football-data CSVs use DD/MM/YYYY."""
    return datetime.strptime(s, "%d/%m/%Y")


def load_matches() -> list:
    """Return list of backtest-shaped dicts for matches in the window."""
    csv_path = fetch_csv()
    matches = []
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                date = parse_date(row["Date"])
            except (KeyError, ValueError):
                continue
            if not (DATE_FROM <= date <= DATE_TO):
                continue
            try:
                h = float(row["B365H"]); d = float(row["B365D"]); a = float(row["B365A"])
            except (KeyError, ValueError):
                continue  # missing odds row — skip
            ph, pd, pa = devig(h, d, a)
            home = row["HomeTeam"]; away = row["AwayTeam"]
            ftr = row["FTR"]  # 'H', 'D', 'A'
            side_a = f"{_slug(home)}_win"
            side_b = f"{_slug(away)}_win"
            draw_label = "draw"
            actual = {
                "H": side_a, "D": draw_label, "A": side_b,
            }[ftr]
            matches.append({
                "date": date.strftime("%Y-%m-%d"),
                "event": f"EPL {home} vs {away}",
                "sport": "soccer",
                "home_team": home,
                "away_team": away,
                "side_a_label": side_a,
                "side_b_label": side_b,
                "draw_label": draw_label,
                "prior_a": ph,
                "prior_b": pa,
                "prior_draw": pd,
                "actual": actual,
            })
    matches.sort(key=lambda m: m["date"])
    return matches


def brier(probs: dict, actual: str) -> float:
    return sum((p - (1.0 if l == actual else 0.0)) ** 2
               for l, p in probs.items())


def log_loss(probs: dict, actual: str, eps: float = 1e-9) -> float:
    return -log(max(eps, probs[actual]))


def modal(probs: dict) -> str:
    return max(probs, key=probs.get)


def run_engine(space, sim_kwargs) -> dict:
    rng = np.random.default_rng(seed=SEED)
    ic_scale = sim_kwargs.get("ic_scale", 1.0)
    q_scale = np.array([0.3, 0.2]) * ic_scale
    p_scale = np.array([0.15, 0.15]) * ic_scale
    acc = {a.label: 0.0 for a in space.attractors}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=q_scale)
        p0 = rng.normal(scale=p_scale)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(space, body=body, dt=DT, **sim_kwargs)
        probs = final_well_posterior(sol, space, alpha=SOFT_ALPHA)
        for label, p in probs.items():
            acc[label] += p
    total = sum(acc.values())
    return {label: v / total for label, v in acc.items()}


def score_one(m: dict, elo_cache: dict) -> tuple:
    """Returns priors, engine_priors_probs, engine_roster_probs, actual,
    plus the (rating_a, rating_b) the roster layer was given."""
    priors = {
        m["side_a_label"]: m["prior_a"],
        m["draw_label"]:   m["prior_draw"],
        m["side_b_label"]: m["prior_b"],
    }

    space_priors, sim_kwargs = build_space(
        sport="soccer",
        side_a_label=m["side_a_label"],
        side_b_label=m["side_b_label"],
        prior_a=m["prior_a"],
        prior_b=m["prior_b"],
        prior_draw=m["prior_draw"],
        draw_label=m["draw_label"],
    )
    engine_priors_probs = run_engine(space_priors, sim_kwargs)

    rating_a = elo_for(m["home_team"], m["date"], elo_cache)
    rating_b = elo_for(m["away_team"], m["date"], elo_cache)
    roster = Roster(players=[
        Player(name=f"{m['side_a_label']}-agg", team=m["side_a_label"],
               rating=rating_a),
        Player(name=f"{m['side_b_label']}-agg", team=m["side_b_label"],
               rating=rating_b),
    ])
    space_roster, _ = build_space(
        sport="soccer",
        side_a_label=m["side_a_label"],
        side_b_label=m["side_b_label"],
        prior_a=m["prior_a"],
        prior_b=m["prior_b"],
        prior_draw=m["prior_draw"],
        draw_label=m["draw_label"],
        roster=roster,
        roster_share=1.0,
    )
    engine_roster_probs = run_engine(space_roster, sim_kwargs)

    return priors, engine_priors_probs, engine_roster_probs, m["actual"], (rating_a, rating_b)


def main() -> None:
    matches = load_matches()
    print("=== Orbita: football-data.co.uk panel backtest (v0.3.3) ===")
    print(f"Season       : 2024/25 EPL")
    print(f"Window       : {DATE_FROM:%Y-%m-%d} → {DATE_TO:%Y-%m-%d}")
    print(f"Panel size   : {len(matches)} matches")
    print(f"Monte Carlo  : N={N_TRIALS}  (seed={SEED})")
    print()

    elo_cache: dict = {}
    records, full = [], []
    for i, m in enumerate(matches, 1):
        print(f"[{i}/{len(matches)}] {m['date']} {m['event']}", flush=True)
        priors, engine_p, engine_r, actual, ratings = score_one(m, elo_cache)
        records.append((priors, engine_p, actual))
        full.append((m, priors, engine_p, engine_r, actual, ratings))

    alpha_is, brier_is = fit_alpha(records)
    loocv_alphas, loocv_brier = loocv_alpha(records)
    print()
    print(f"Fitted alpha (in-sample) : {alpha_is:.3f}  "
          f"(mean Brier @ alpha = {brier_is:.3f})")
    print(f"LOOCV alpha (per-fold)   : "
          f"min={min(loocv_alphas):.3f}  "
          f"median={float(np.median(loocv_alphas)):.3f}  "
          f"max={max(loocv_alphas):.3f}")
    print(f"LOOCV mean Brier         : {loocv_brier:.3f}")
    print()

    overall = defaultdict(list)
    hits = defaultdict(int)
    for m, priors, engine_p, engine_r, actual, ratings in full:
        calibrated = blend(priors, engine_p, alpha_is)
        overall["bookmaker"].append(brier(priors, actual))
        overall["orbita_priors"].append(brier(engine_p, actual))
        overall["orbita_roster"].append(brier(engine_r, actual))
        overall["orbita_calibrated"].append(brier(calibrated, actual))
        for col, probs in (("bookmaker", priors),
                           ("orbita_priors", engine_p),
                           ("orbita_roster", engine_r),
                           ("orbita_calibrated", calibrated)):
            if modal(probs) == actual:
                hits[col] += 1

    n = len(full)
    print("=== aggregate Brier (mean, lower is better) ===")
    print(f"{'model':<22s}  {'N':>3s}  {'mean Brier':>12s}  {'modal hit-rate':>16s}")
    for c in ("bookmaker", "orbita_priors", "orbita_roster",
              "orbita_calibrated"):
        print(f"{c:<22s}  {n:>3d}  {float(np.mean(overall[c])):>12.3f}  "
              f"{hits[c] / n:>15.1%}")
    print()

    mb = float(np.mean(overall["bookmaker"]))
    mp = float(np.mean(overall["orbita_priors"]))
    mr = float(np.mean(overall["orbita_roster"]))
    mc = float(np.mean(overall["orbita_calibrated"]))
    print("=== verdicts (delta = engine - bookmaker, negative = engine wins) ===")
    print(f"priors-only vs market : {mp - mb:+.3f}  "
          f"({'engine wins' if mp < mb else 'market wins'})")
    print(f"roster vs market      : {mr - mb:+.3f}  "
          f"({'engine+roster wins' if mr < mb else 'market wins'})")
    print(f"calibrated vs market  : {mc - mb:+.3f}  "
          f"({'engine wins' if mc < mb else 'market wins'})")
    print(f"LOOCV vs market       : {loocv_brier - mb:+.3f}  "
          f"({'engine wins' if loocv_brier < mb else 'market wins'})")


if __name__ == "__main__":
    main()

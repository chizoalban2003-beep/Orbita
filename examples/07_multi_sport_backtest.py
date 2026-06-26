"""
07_multi_sport_backtest.py — v0.2

Multi-sport backtest of the Orbita engine with sport-specific
event-space templates (issue #4) and alpha sharpening calibration
(issue #3).

For each match we run four forecasts and score each against the actual
outcome with Brier and log-loss:

    1. BOOKMAKER          — believe the market exactly
    2. ORBITA_PRIORS      — engine + bookmaker priors, no roster
                            (using the SPORT-SPECIFIC template)
    3. ORBITA_ROSTER      — engine + bookmaker priors + roster mass-
                            multiplier (sport-specific template)
    4. ORBITA_CALIBRATED  — alpha-blend of bookmaker priors and the
                            engine's priors-only output. Alpha fit by
                            (a) in-sample grid search and (b) leave-
                            one-out cross-validation.

Run with::

    PYTHONPATH=src python3 examples/07_multi_sport_backtest.py
"""
from __future__ import annotations

import sys
import tomllib
import warnings
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np

from orbita import (
    Body,
    Player,
    Roster,
    aggregate_brier,
    blend,
    build_space,
    final_well,
    fit_alpha,
    loocv_alpha,
    simulate,
)


warnings.filterwarnings("ignore", message=r".*Renormalizing.*")

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "backtest_matches.toml"

N_TRIALS = 40
SEED = 20260626
DT = 0.1


def brier(probs: dict, actual: str) -> float:
    return sum((p - (1.0 if l == actual else 0.0)) ** 2
               for l, p in probs.items())


def log_loss(probs: dict, actual: str, eps: float = 1e-9) -> float:
    return -log(max(eps, probs[actual]))


def modal(probs: dict) -> str:
    return max(probs, key=probs.get)


def run_engine(space, sim_kwargs):
    """Monte-Carlo the engine and return label -> probability."""
    rng = np.random.default_rng(seed=SEED)
    counts = {a.label: 0 for a in space.attractors}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=[0.3, 0.2], size=2)
        p0 = rng.normal(scale=[0.15, 0.15], size=2)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(space, body=body, dt=DT, **sim_kwargs)
        counts[final_well(sol, space)] += 1
    total = sum(counts.values())
    return {label: c / total for label, c in counts.items()}


def score_match(match: dict):
    """Run all forecasts on one match. Returns (bookmaker_probs,
    engine_priors_probs, engine_roster_probs, priors_dict, actual)."""
    sport = match["sport"]
    side_a = match["side_a_label"]
    side_b = match["side_b_label"]
    draw_label = match.get("draw_label")
    actual = match["actual"]

    if "prior_draw" in match:
        priors = {
            side_a:     match["prior_a"],
            draw_label: match["prior_draw"],
            side_b:     match["prior_b"],
        }
    else:
        priors = {side_a: match["prior_a"], side_b: match["prior_b"]}

    space_priors, sim_kwargs = build_space(
        sport=sport,
        side_a_label=side_a,
        side_b_label=side_b,
        prior_a=match["prior_a"],
        prior_b=match["prior_b"],
        prior_draw=match.get("prior_draw"),
        draw_label=draw_label,
    )
    engine_priors_probs = run_engine(space_priors, sim_kwargs)

    roster = Roster(players=[
        Player(name=f"{side_a}-agg", team=side_a,
               rating=float(match["roster_strength_a"])),
        Player(name=f"{side_b}-agg", team=side_b,
               rating=float(match["roster_strength_b"])),
    ])
    space_roster, _ = build_space(
        sport=sport,
        side_a_label=side_a,
        side_b_label=side_b,
        prior_a=match["prior_a"],
        prior_b=match["prior_b"],
        prior_draw=match.get("prior_draw"),
        draw_label=draw_label,
        roster=roster,
        roster_share=1.0,
    )
    engine_roster_probs = run_engine(space_roster, sim_kwargs)

    return priors, engine_priors_probs, engine_roster_probs, actual


def fmt_probs(probs: dict) -> str:
    return "  ".join(f"{k}={v:.0%}" for k, v in probs.items())


def fmt_row(label: str, probs: dict, actual: str) -> str:
    b = brier(probs, actual)
    ll = log_loss(probs, actual)
    hit = "✓" if modal(probs) == actual else "✗"
    return (f"    {label:<18s} brier={b:.3f}  logloss={ll:.2f}  "
            f"modal{hit}  [{fmt_probs(probs)}]")


def main() -> None:
    if not DATA_FILE.exists():
        print(f"ERROR: data file not found at {DATA_FILE}", flush=True)
        sys.exit(1)

    with DATA_FILE.open("rb") as fh:
        data = tomllib.load(fh)
    matches = data.get("match", [])

    print("=== Orbita: multi-sport backtest (v0.2 templates + alpha) ===")
    print(f"Data file       : {DATA_FILE.name}  ({len(matches)} matches)")
    print(f"Monte Carlo N   : {N_TRIALS}  (seed={SEED}, deterministic)")
    print()

    # PASS 1 — run priors-only and roster forecasts for every match.
    records = []   # (priors_dict, engine_priors_probs, actual) for alpha fit
    full_results = []  # for printing
    for i, m in enumerate(matches, 1):
        print(f"[{i}/{len(matches)}] scoring {m['sport']} {m.get('date', '?')}...",
              flush=True)
        priors, engine_p, engine_r, actual = score_match(m)
        records.append((priors, engine_p, actual))
        full_results.append((m, priors, engine_p, engine_r, actual))

    # PASS 2 — fit alpha (in-sample) and report.
    alpha_in_sample, brier_in_sample = fit_alpha(records)
    loocv_alphas, loocv_mean_brier = loocv_alpha(records)
    print()
    print(f"Fitted alpha (in-sample)    : {alpha_in_sample:.3f}  "
          f"(mean Brier @ alpha = {brier_in_sample:.3f})")
    print(f"Fitted alpha (LOOCV per fold): "
          f"min={min(loocv_alphas):.3f}  max={max(loocv_alphas):.3f}  "
          f"median={float(np.median(loocv_alphas)):.3f}")
    print(f"LOOCV mean Brier            : {loocv_mean_brier:.3f}")
    print()

    # PASS 3 — print per-match table.
    per_sport = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)

    for m, priors, engine_p, engine_r, actual in full_results:
        sport = m["sport"]
        print(f"--- {sport:<8s} {m.get('date', '?')}  "
              f"{m.get('event', '?')}")
        print(f"    actual : {actual}")
        # bookmaker = priors as-is
        print(fmt_row("bookmaker", priors, actual))
        print(fmt_row("orbita_priors", engine_p, actual))
        print(fmt_row("orbita_roster", engine_r, actual))
        # calibrated = blend at in-sample alpha
        calibrated = blend(priors, engine_p, alpha_in_sample)
        print(fmt_row("orbita_calibrated", calibrated, actual))

        per_sport[sport]["bookmaker"].append(brier(priors, actual))
        per_sport[sport]["orbita_priors"].append(brier(engine_p, actual))
        per_sport[sport]["orbita_roster"].append(brier(engine_r, actual))
        per_sport[sport]["orbita_calibrated"].append(brier(calibrated, actual))
        for col in ("bookmaker", "orbita_priors", "orbita_roster",
                    "orbita_calibrated"):
            overall[col].append(per_sport[sport][col][-1])
        print()

    # PASS 4 — aggregate report.
    print("=== aggregate Brier (mean — lower is better) ===")
    cols = ("bookmaker", "orbita_priors", "orbita_roster", "orbita_calibrated")
    header = f"{'sport':<10s} {'N':>3s}  " + "  ".join(
        f"{c:>18s}" for c in cols)
    print(header)
    for sport, models in per_sport.items():
        n = len(models["bookmaker"])
        row = f"{sport:<10s} {n:>3d}  " + "  ".join(
            f"{np.mean(models[c]):>18.3f}" for c in cols)
        print(row)
    n_all = len(overall["bookmaker"])
    row = f"{'ALL':<10s} {n_all:>3d}  " + "  ".join(
        f"{np.mean(overall[c]):>18.3f}" for c in cols)
    print(row)
    print()

    # PASS 5 — verdicts.
    mb = float(np.mean(overall["bookmaker"]))
    mp = float(np.mean(overall["orbita_priors"]))
    mr = float(np.mean(overall["orbita_roster"]))
    mc = float(np.mean(overall["orbita_calibrated"]))

    print("=== verdicts (delta = engine - bookmaker, negative = engine wins) ===")
    print(f"engine vs market (priors-only)      : {mp - mb:+.3f}  "
          f"({'engine wins' if mp < mb else 'market wins'})")
    print(f"engine+roster vs market              : {mr - mb:+.3f}  "
          f"({'engine+roster wins' if mr < mb else 'market wins'})")
    print(f"engine+alpha vs market (in-sample)   : {mc - mb:+.3f}  "
          f"({'calibrated engine wins' if mc < mb else 'market wins'})")
    print(f"engine+alpha vs market (LOOCV mean)  : {loocv_mean_brier - mb:+.3f}  "
          f"({'calibrated engine wins' if loocv_mean_brier < mb else 'market wins'})")
    print()

    # Diagnostic on alpha.
    if alpha_in_sample < 0.05:
        print("Diagnostic: alpha collapsed to ~0. The engine adds essentially")
        print("no signal beyond the prior on this panel. The gravity-well")
        print("physics is the bottleneck — calibration cannot save it.")
    elif alpha_in_sample > 0.95:
        print("Diagnostic: alpha → 1. The engine is well-calibrated; trust it.")
    else:
        print(f"Diagnostic: alpha ≈ {alpha_in_sample:.2f}. The engine adds")
        print("partial signal — useful blend with the prior.")


if __name__ == "__main__":
    main()

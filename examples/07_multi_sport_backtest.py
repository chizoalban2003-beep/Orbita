"""
07_multi_sport_backtest.py

Multi-sport backtest of the Orbita engine + minimal roster layer.

For each match in ``data/backtest_matches.toml`` we run three forecasts
and score each against the actual outcome:

    1. BOOKMAKER     — believe the market exactly
    2. ORBITA_PRIORS — engine + bookmaker priors, no roster (Path A from
                       the user's options)
    3. ORBITA_ROSTER — engine + bookmaker priors + roster mass-multiplier
                       (Path B; this is the player-up signal we actually
                       want to test)

The aggregate report tells us, per sport and overall:
    - Whether the engine on its own (no roster) beats, matches, or loses
      to the market when fed market priors. Anything other than "matches"
      is a structural engine bias and a calibration target.
    - Whether adding the roster layer improves on the priors-only run.
      This is the headline number: did "players are mass-modifiers"
      add signal in real matches?

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

# Roster multipliers intentionally rescale well masses; the resulting
# pre-normalisation sum is not 1.0, and EventSpace renormalises it back.
# That's the design — silence the noisy warning for the backtest output.
warnings.filterwarnings("ignore", message=r".*Renormalizing.*")

from orbita import (
    Body,
    Player,
    Roster,
    event_space_from_rosters,
    final_well,
    simulate,
)


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "backtest_matches.toml"

N_TRIALS = 40
SEED = 20260626
C_d = 0.04          # slightly stronger drag — most contests resolve faster than 90 min
DURATION = 600.0    # body falls into a well long before 600 sim-seconds
DT = 0.1

POS_THREE_WAY = {
    "side_a": [5.0, 0.0],
    "draw":   [0.0, 4.0],
    "side_b": [-5.0, 0.0],
}
POS_TWO_WAY = {
    "side_a": [5.0, 0.0],
    "side_b": [-5.0, 0.0],
}


def brier(probs: dict, actual: str) -> float:
    s = 0.0
    for label, p in probs.items():
        target = 1.0 if label == actual else 0.0
        s += (p - target) ** 2
    return s


def log_loss(probs: dict, actual: str, eps: float = 1e-9) -> float:
    return -log(max(eps, probs[actual]))


def modal(probs: dict) -> str:
    return max(probs, key=probs.get)


def run_engine(base_priors, positions, roster=None, head_to_head=None,
               roster_share=1.0):
    """Monte-Carlo the engine with the given priors / roster and return the
    outcome distribution as a dict of label -> probability."""
    space = event_space_from_rosters(
        base_priors=base_priors,
        positions=positions,
        roster=roster,
        head_to_head=head_to_head,
        roster_share=roster_share,
    )
    rng = np.random.default_rng(seed=SEED)
    counts = {a.label: 0 for a in space.attractors}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=[0.3, 0.2], size=2)
        p0 = rng.normal(scale=[0.15, 0.15], size=2)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(space, body=body, duration=DURATION, C_d=C_d, dt=DT)
        counts[final_well(sol, space)] += 1
    total = sum(counts.values())
    return {label: c / total for label, c in counts.items()}


def score_match(match: dict) -> dict:
    """Score a single match record under bookmaker / priors-only / roster
    forecasts. Returns a dict of Brier and logloss numbers per model."""
    has_draw = "prior_draw" in match

    side_a_label = match["side_a_label"]
    side_b_label = match["side_b_label"]
    draw_label = match.get("draw_label")
    actual = match["actual"]

    if has_draw:
        priors = {
            side_a_label: match["prior_a"],
            draw_label:   match["prior_draw"],
            side_b_label: match["prior_b"],
        }
        positions = {
            side_a_label: POS_THREE_WAY["side_a"],
            draw_label:   POS_THREE_WAY["draw"],
            side_b_label: POS_THREE_WAY["side_b"],
        }
    else:
        priors = {
            side_a_label: match["prior_a"],
            side_b_label: match["prior_b"],
        }
        positions = {
            side_a_label: POS_TWO_WAY["side_a"],
            side_b_label: POS_TWO_WAY["side_b"],
        }

    # 1. Bookmaker = priors as-is, no engine
    bookmaker_probs = dict(priors)

    # 2. Engine, no roster
    engine_priors_probs = run_engine(priors, positions)

    # 3. Engine + roster mass-multiplier
    roster = Roster(players=[
        Player(name=f"{side_a_label}-aggregate",
               team=side_a_label,
               rating=float(match["roster_strength_a"])),
        Player(name=f"{side_b_label}-aggregate",
               team=side_b_label,
               rating=float(match["roster_strength_b"])),
    ])
    engine_roster_probs = run_engine(
        priors,
        positions,
        roster=roster,
        head_to_head=(side_a_label, side_b_label),
        roster_share=1.0,
    )

    return {
        "bookmaker":     {"probs": bookmaker_probs,
                          "brier": brier(bookmaker_probs, actual),
                          "logloss": log_loss(bookmaker_probs, actual),
                          "modal_hit": modal(bookmaker_probs) == actual},
        "orbita_priors": {"probs": engine_priors_probs,
                          "brier": brier(engine_priors_probs, actual),
                          "logloss": log_loss(engine_priors_probs, actual),
                          "modal_hit": modal(engine_priors_probs) == actual},
        "orbita_roster": {"probs": engine_roster_probs,
                          "brier": brier(engine_roster_probs, actual),
                          "logloss": log_loss(engine_roster_probs, actual),
                          "modal_hit": modal(engine_roster_probs) == actual},
    }


def fmt_probs(probs: dict) -> str:
    return "  ".join(f"{k}={v:.0%}" for k, v in probs.items())


def main() -> None:
    if not DATA_FILE.exists():
        print(f"ERROR: data file not found at {DATA_FILE}")
        print("Run the multi-sport data subagent first.")
        sys.exit(1)

    with DATA_FILE.open("rb") as fh:
        data = tomllib.load(fh)
    matches = data.get("match", [])

    print("=== Orbita: multi-sport backtest ===")
    print(f"Data file       : {DATA_FILE.name}  ({len(matches)} matches)")
    print(f"Monte Carlo N   : {N_TRIALS}  (seed={SEED}, deterministic)")
    print()

    per_sport = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)

    for i, m in enumerate(matches, 1):
        sport = m["sport"]
        event = m.get("event", "?")
        date = m.get("date", "?")
        print(f"[{i}/{len(matches)}] scoring {sport} {date}...", flush=True)
        scores = score_match(m)
        print(f"--- {sport:<8s} {date}  {event}")
        print(f"    actual : {m['actual']}")
        for model in ("bookmaker", "orbita_priors", "orbita_roster"):
            s = scores[model]
            hit = "✓" if s["modal_hit"] else "✗"
            print(f"    {model:<14s} brier={s['brier']:.3f}  "
                  f"logloss={s['logloss']:.2f}  modal{hit}  "
                  f"[{fmt_probs(s['probs'])}]")
            per_sport[sport][model].append(s["brier"])
            overall[model].append(s["brier"])
        print()

    print("=== aggregate Brier (mean — lower is better) ===")
    print(f"{'sport':<10s} {'N':>3s}  {'bookmaker':>10s}  "
          f"{'orbita_priors':>14s}  {'orbita_roster':>14s}")
    for sport, models in per_sport.items():
        n = len(models["bookmaker"])
        mb = np.mean(models["bookmaker"])
        mp = np.mean(models["orbita_priors"])
        mr = np.mean(models["orbita_roster"])
        print(f"{sport:<10s} {n:>3d}  {mb:>10.3f}  {mp:>14.3f}  {mr:>14.3f}")
    n_all = len(overall["bookmaker"])
    mb = np.mean(overall["bookmaker"])
    mp = np.mean(overall["orbita_priors"])
    mr = np.mean(overall["orbita_roster"])
    print(f"{'ALL':<10s} {n_all:>3d}  {mb:>10.3f}  {mp:>14.3f}  {mr:>14.3f}")
    print()

    print("=== verdicts ===")
    delta_priors = mp - mb
    delta_roster = mr - mb
    delta_lift   = mr - mp
    print(f"engine vs market (priors-only): {delta_priors:+.3f} Brier  "
          f"({'engine wins' if delta_priors < 0 else 'market wins'})")
    print(f"engine+roster vs market       : {delta_roster:+.3f} Brier  "
          f"({'engine+roster wins' if delta_roster < 0 else 'market wins'})")
    print(f"roster lift over priors-only  : {delta_lift:+.3f} Brier  "
          f"({'roster helps' if delta_lift < 0 else 'roster hurts'})")


if __name__ == "__main__":
    main()

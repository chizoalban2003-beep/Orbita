"""
05b_norway_france_ablation.py

Prior-sensitivity ablation for the Norway vs France forecast.

The headline run (05_world_cup_norway_france.py) prints France 43.3% /
Draw 28.3% / Norway 28.3% — a big departure from the bookmaker
consensus of 60 / 21 / 19. Three explanations are possible:

    A. The priors we picked are doing all the work. If we feed the
       engine the bookmaker priors and a neutral body, it should land
       on the bookmaker numbers (or very close).
    B. The momentum encoding (body pushed toward Norway's side because
       "Norway must win") is dragging probability mass off France even
       when the priors agree with the market.
    C. Author's adjusted priors *plus* the momentum encoding combine
       to produce the headline forecast.

Running A, B, C with the same RNG seed isolates which factor is doing
what. This is the post-mortem instrument we'll point at the result
once the actual score is in.

Run with::

    PYTHONPATH=src python3 examples/05b_norway_france_ablation.py
"""
from __future__ import annotations

import numpy as np

from orbita import Attractor, Body, EventSpace, final_well, simulate


BOOKMAKER = {"france_win": 0.60, "draw": 0.21, "norway_win": 0.19}
N_TRIALS = 60
SEED = 20260626
C_d = 0.02


def run_scenario(name, priors, q0_mean, p0_mean, note):
    space = EventSpace([
        Attractor(label="france_win", position=[5.0, 0.0],  mass=priors["france_win"]),
        Attractor(label="draw",       position=[0.0, 4.0],  mass=priors["draw"]),
        Attractor(label="norway_win", position=[-5.0, 0.0], mass=priors["norway_win"]),
    ])

    rng = np.random.default_rng(seed=SEED)
    counts = {a.label: 0 for a in space.attractors}

    for _ in range(N_TRIALS):
        q0 = q0_mean + rng.normal(scale=[0.3, 0.2], size=2)
        p0 = p0_mean + rng.normal(scale=[0.15, 0.15], size=2)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(space, body=body, duration=2700.0, C_d=C_d, dt=0.1)
        counts[final_well(sol, space)] += 1

    total = sum(counts.values())
    probs = {label: c / total for label, c in counts.items()}
    return probs, note


SCENARIOS = [
    ("A — bookmaker priors, neutral body",
     dict(BOOKMAKER),
     np.array([0.0, 0.0]),
     np.array([0.0, 0.0]),
     "Tests whether priors alone explain the headline disagreement."),
    ("B — bookmaker priors, Norway-press momentum",
     dict(BOOKMAKER),
     np.array([0.5, 0.0]),
     np.array([-0.3, 0.4]),
     "Tests whether the momentum encoding bleeds France probability."),
    ("C — author priors, Norway-press momentum (=05)",
     {"france_win": 0.52, "draw": 0.20, "norway_win": 0.28},
     np.array([0.5, 0.0]),
     np.array([-0.3, 0.4]),
     "Reproduces the headline forecast for reference."),
]


print("=== Orbita: Norway vs France — prior-sensitivity ablation ===")
print(f"Date            : 2026-06-26  (World Cup, Group I, Foxborough)")
print(f"Monte Carlo N   : {N_TRIALS}  (seed={SEED}, deterministic)")
print(f"Reference (book): France {BOOKMAKER['france_win']:.0%}  "
      f"Draw {BOOKMAKER['draw']:.0%}  "
      f"Norway {BOOKMAKER['norway_win']:.0%}")
print()

results = []
for name, priors, q0, p0, note in SCENARIOS:
    probs, _ = run_scenario(name, priors, q0, p0, note)
    results.append((name, probs, note))
    print(f"--- {name}")
    print(f"    {note}")
    for label in ("france_win", "draw", "norway_win"):
        delta = probs[label] - BOOKMAKER[label]
        arrow = "↑" if delta >= 0 else "↓"
        print(f"      {label:<12s}: {probs[label]:>6.1%}   "
              f"(book {BOOKMAKER[label]:>5.1%}, {arrow}{abs(delta)*100:>4.1f} pp)")
    print()

# --- interpretation -------------------------------------------------------
a_probs = results[0][1]
b_probs = results[1][1]
c_probs = results[2][1]

a_book_gap = abs(a_probs["france_win"] - BOOKMAKER["france_win"])
b_vs_a_gap = abs(b_probs["france_win"] - a_probs["france_win"])
c_vs_b_gap = abs(c_probs["france_win"] - b_probs["france_win"])

print("--- decomposition (France-win probability) ---")
print(f"  bookmaker baseline                  : {BOOKMAKER['france_win']:>6.1%}")
print(f"  A (engine, bookmaker priors)        : {a_probs['france_win']:>6.1%}   "
      f"Δ vs book = {(a_probs['france_win']-BOOKMAKER['france_win'])*100:+.1f} pp  "
      f"(structural engine bias)")
print(f"  B (+ Norway-press momentum)         : {b_probs['france_win']:>6.1%}   "
      f"Δ vs A    = {(b_probs['france_win']-a_probs['france_win'])*100:+.1f} pp  "
      f"(momentum effect)")
print(f"  C (+ author prior shift)            : {c_probs['france_win']:>6.1%}   "
      f"Δ vs B    = {(c_probs['france_win']-b_probs['france_win'])*100:+.1f} pp  "
      f"(prior shift effect)")
print()

if a_book_gap < 0.05:
    verdict_a = "engine TRACKS market when fed market priors  ✓"
else:
    verdict_a = f"engine deviates {a_book_gap*100:.1f} pp from market on priors alone  ⚠"
print(f"Verdict on A : {verdict_a}")
print(f"Verdict on B : momentum encoding moves France by {b_vs_a_gap*100:.1f} pp")
print(f"Verdict on C : author prior shift adds another {c_vs_b_gap*100:.1f} pp on France")

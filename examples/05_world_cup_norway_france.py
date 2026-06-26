"""
05_world_cup_norway_france.py

Norway vs France — World Cup 2026, Group I, June 26 2026, Boston/Foxborough.

A pre-kickoff Orbita prediction published as a public benchmark. This file
is intentionally checked into the repo so the prediction can be scored
against the actual outcome afterward.

Match facts encoded as priors and intangibles:

    - France need only a draw to top Group I  →  market favorite (~60%)
    - Norway must win to top the group        →  high desperation thrust
    - France injuries: Saliba (CB) and Ryerson out, Lacroix in
    - Both teams' attacks firing: Mbappé (4 tournament goals), Haaland
      (2 braces in 2 games)
    - Norwegian-American population in New England → mildly pro-Norway crowd

Run with::

    python examples/05_world_cup_norway_france.py
"""
from __future__ import annotations

import numpy as np

from orbita import Attractor, Body, EventSpace, final_well, simulate


# Orbita's three outcome wells. Priors lean slightly toward Norway vs the
# bookmaker consensus (France ~60% / Draw ~21% / Norway ~20%) because the
# engine is told Norway *must* win — desperation reduces their anti-Win drag
# while France's injuries shallow the France well.
space = EventSpace([
    Attractor(label="france_win", position=[5.0, 0.0],   mass=0.52),
    Attractor(label="draw",       position=[0.0, 4.0],   mass=0.20),
    Attractor(label="norway_win", position=[-5.0, 0.0],  mass=0.28),
])

# Body starts slightly biased toward France (they're favored on paper),
# with momentum directed toward Norway side (Norway must press from minute 1).
body_q0_mean = np.array([0.5, 0.0])
body_p0_mean = np.array([-0.3, 0.4])

C_d = 0.02   # mid-match isotropic drag

# Monte Carlo: run 60 trials with randomized initial conditions to estimate
# the outcome distribution. (Phase 2 will replace this with a proper Bayesian
# posterior over initial state and parameters.)
N_TRIALS = 60
RNG = np.random.default_rng(seed=20260626)
counts = {a.label: 0 for a in space.attractors}

for trial in range(N_TRIALS):
    q0 = body_q0_mean + RNG.normal(scale=[0.3, 0.2], size=2)
    p0 = body_p0_mean + RNG.normal(scale=[0.15, 0.15], size=2)
    body = Body(mass=1.0, q0=q0, p0=p0)
    sol = simulate(space, body=body, duration=2700.0, C_d=C_d, dt=0.1)
    counts[final_well(sol, space)] += 1

total = sum(counts.values())
probs = {label: c / total for label, c in counts.items()}

print("=== Orbita: Norway vs France — pre-kickoff forecast ===")
print(f"Date            : 2026-06-26  (World Cup, Group I, Foxborough)")
print(f"Engine version  : Orbita v0.1 (mechanistic, no learned weights)")
print(f"Monte Carlo N   : {N_TRIALS}")
print()
print("Outcome distribution:")
for label in ("france_win", "draw", "norway_win"):
    print(f"  {label:<12s}: {probs[label]:>6.1%}")
print()

bookmaker = {"france_win": 0.60, "draw": 0.21, "norway_win": 0.19}
print("Compared to bookmaker consensus (Opta supercomputer ≈ same):")
for label in ("france_win", "draw", "norway_win"):
    delta = probs[label] - bookmaker[label]
    arrow = "↑" if delta > 0 else "↓"
    print(f"  {label:<12s}: Orbita {probs[label]:>5.1%}  vs  book {bookmaker[label]:>5.1%}   {arrow}{abs(delta)*100:>4.1f} pp")

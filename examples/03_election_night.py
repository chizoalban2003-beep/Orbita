"""
03_election_night.py

Cross-domain demo: same engine, different event. Two candidates as
attractors, polling momentum as the orbiting body. Drag = electorate
stickiness (the harder it is to change minds, the higher the drag).

This file demonstrates Orbita's core claim: the engine is domain-agnostic.
Nothing physical changed between soccer and elections — only the labels,
priors, and time scale.

Run with::

    python examples/03_election_night.py
"""
from __future__ import annotations

from orbita import Attractor, Body, EventSpace, final_well, simulate

space = EventSpace([
    Attractor(label="candidate_a", position=[6.0, 0.0],  mass=0.48),
    Attractor(label="candidate_b", position=[-6.0, 0.0], mass=0.52),
])

body = Body(mass=1.0, q0=[0.0, 0.0], p0=[-0.2, 0.1])

C_d = 0.05   # higher drag than soccer — electorate inertia exceeds a 90-min match

# Simulate a 30-day campaign (units are arbitrary; what matters is total time × drag)
duration = 30.0 * 24.0 * 60.0

print("=== Orbita: election night simulation ===")
print(f"Priors:    A={space.attractors[0].mass}  B={space.attractors[1].mass}")

sol = simulate(space, body=body, duration=duration, C_d=C_d)
qf = sol["q"][-1]
print(f"Final position    = ({qf[0]:.3f}, {qf[1]:.3f})")
print(f"Predicted winner  = {final_well(sol, space)}")

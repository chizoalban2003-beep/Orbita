"""
02_soccer_match.py

Soccer demo with drag enabled. Three outcomes (home / draw / away),
asymmetric priors reflecting a moderate home favorite. Drag bleeds the
body's energy so it eventually falls into one of the wells.

Run with::

    python examples/02_soccer_match.py
"""
from __future__ import annotations

from orbita import Attractor, Body, EventSpace, final_well, simulate

space = EventSpace([
    Attractor(label="home_win", position=[5.0, 0.0],  mass=0.55),
    Attractor(label="draw",     position=[0.0, 4.0],  mass=0.20),
    Attractor(label="away_win", position=[-5.0, 0.0], mass=0.25),
])

body = Body(mass=1.0, q0=[0.5, 0.0], p0=[0.4, 0.2])

C_d = 0.02   # mild isotropic drag — Phase 2 will derive this from the ontology

print("=== Orbita: soccer match simulation ===")
print(f"Priors:    home={space.attractors[0].mass}  draw={space.attractors[1].mass}  away={space.attractors[2].mass}")
print(f"Body:      q0={body.q0.tolist()}  p0={body.p0.tolist()}")
print(f"Drag:      C_d={C_d}")
print()

sol = simulate(space, body=body, duration=5400.0, C_d=C_d)

qf = sol["q"][-1]
print(f"Final position    = ({qf[0]:.3f}, {qf[1]:.3f})")
print(f"Predicted outcome = {final_well(sol, space)}")
print(f"Trajectory points = {len(sol['t'])}")

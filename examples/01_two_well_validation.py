"""
01_two_well_validation.py

The non-negotiable gate. Before any drag or intangibles, the symplectic
integrator must conserve total energy over a 90-minute simulation. If it
doesn't, every prediction downstream is polluted by numerical leakage
that looks just like real drag.

Run with::

    python examples/01_two_well_validation.py
"""
from __future__ import annotations

import sys

from orbita import Attractor, Body, EventSpace, final_well, hamiltonian, simulate

space = EventSpace([
    Attractor(label="home_win", position=[5.0, 0.0],  mass=0.33),
    Attractor(label="draw",     position=[0.0, 4.0],  mass=0.34),
    Attractor(label="away_win", position=[-5.0, 0.0], mass=0.33),
])

body = Body(mass=1.0, q0=[0.0, 0.0], p0=[0.5, 0.3])

# Drag OFF — this is the conservation test.
sol = simulate(space, body=body, duration=5400.0, C_d=0.0)

attractors = list(space.attractors)
H0 = hamiltonian(sol["q"][0],  sol["p"][0],  body.mass, attractors)
Hf = hamiltonian(sol["q"][-1], sol["p"][-1], body.mass, attractors)
drift = (Hf - H0) / abs(H0)

print(f"H(0)            = {H0:.6e}")
print(f"H(T)            = {Hf:.6e}")
print(f"Relative drift  = {drift:.3e}")
print(f"Final position  = ({sol['q'][-1][0]:.3f}, {sol['q'][-1][1]:.3f})")
print(f"Closest well    = {final_well(sol, space)}")

if abs(drift) < 1e-2:
    print("\n✓ Validation passed. Integrator conserves energy.")
    sys.exit(0)
else:
    print(f"\n✗ Validation FAILED. Drift = {drift:.3e} exceeds tolerance 1e-2.")
    sys.exit(1)

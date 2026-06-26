"""Ontology sanity check: higher drag → less remaining energy."""
from __future__ import annotations

from orbita import Attractor, Body, EventSpace, hamiltonian, simulate


def test_higher_drag_bleeds_more_energy():
    space = EventSpace([
        Attractor("a", [4.0, 0.0],  0.5),
        Attractor("b", [-4.0, 0.0], 0.5),
    ])
    body = Body(mass=1.0, q0=[0.0, 1.0], p0=[0.4, 0.0])
    attractors = list(space.attractors)

    def final_energy(C_d: float) -> float:
        sol = simulate(space, body=body, duration=600.0, C_d=C_d)
        return hamiltonian(sol["q"][-1], sol["p"][-1], body.mass, attractors)

    H_low = final_energy(0.01)
    H_high = final_energy(0.10)

    assert H_high < H_low, f"Higher drag did not bleed more energy: low={H_low}, high={H_high}"

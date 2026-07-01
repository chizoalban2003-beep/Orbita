"""Ontology sanity check: higher drag → less remaining energy."""
from __future__ import annotations

import numpy as np

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


def test_anisotropic_drag_force_components():
    """drag_force with C_d = [a, b] returns -[a, b] * p / m component-wise."""
    from orbita.forces import drag_force

    p = np.array([0.5, 0.5])
    m = 1.0
    iso = drag_force(p, m, 0.10)
    aniso = drag_force(p, m, np.array([0.10, 0.0]))

    assert np.allclose(iso, [-0.05, -0.05]), iso
    # x-component unchanged, y-component zero.
    assert np.allclose(aniso, [-0.05, 0.0]), aniso


def test_anisotropic_drag_sim_bleeds_only_active_axis():
    """A two-well event space on the y-axis: with drag = [0.10, 0.0]
    only the x-component bleeds. The body's y-energy survives, x-energy
    falls. Compares the H/D/L analogue (x = home/away, y = draw)."""
    space = EventSpace([
        Attractor("home", [4.0, 0.0],  0.5),
        Attractor("away", [-4.0, 0.0], 0.5),
    ])
    body = Body(mass=1.0, q0=[0.0, 0.0], p0=[0.5, 0.5])

    sol_iso = simulate(space, body=body, duration=200.0, C_d=0.05)
    sol_x_only = simulate(space, body=body, duration=200.0,
                          C_d=np.array([0.05, 0.0]))

    p_iso = sol_iso["p"][-1]
    p_x = sol_x_only["p"][-1]
    # With y-drag off, |p_y| should be larger at end than under isotropic.
    assert abs(p_x[1]) > abs(p_iso[1]), (
        f"Anisotropic (no y-drag) did not preserve y-momentum vs isotropic: "
        f"aniso |p_y|={abs(p_x[1]):.4f}, iso |p_y|={abs(p_iso[1]):.4f}"
    )

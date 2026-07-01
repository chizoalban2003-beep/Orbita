"""n-dimensional event-space support (v0.3.6)."""
from __future__ import annotations

import numpy as np

from orbita import Attractor, Body, EventSpace, hamiltonian, simulate


def test_3d_attractor_accepts_shape_3():
    a = Attractor("w", [1.0, 2.0, 3.0], 1.0)
    assert a.position.shape == (3,)


def test_2d_and_higher_positions_reject_mismatched_body():
    space = EventSpace([
        Attractor("a", [1.0, 0.0, 0.0], 0.5),
        Attractor("b", [-1.0, 0.0, 0.0], 0.5),
    ])
    body_2d = Body(mass=1.0, q0=[0.0, 0.0], p0=[0.1, 0.1])
    # 2D body vs 3D attractors: gravity_force loop mixes dims → error.
    import pytest

    with pytest.raises((ValueError, IndexError)):
        simulate(space, body=body_2d, duration=1.0, C_d=0.0)


def test_3d_simulate_conserves_energy_zero_drag():
    """Energy conservation is the ontology's ground truth. If it survives
    a jump to 3D, the refactor is sound."""
    space = EventSpace([
        Attractor("a", [2.0, 0.0, 0.0], 0.5),
        Attractor("b", [-2.0, 0.0, 0.0], 0.5),
    ])
    body = Body(mass=1.0, q0=[0.0, 1.0, 0.5], p0=[0.3, 0.0, 0.0])
    sol = simulate(space, body=body, duration=50.0, C_d=0.0, dt=0.02)
    attrs = list(space.attractors)
    H0 = hamiltonian(sol["q"][0], sol["p"][0], body.mass, attrs)
    HN = hamiltonian(sol["q"][-1], sol["p"][-1], body.mass, attrs)
    rel_drift = abs(HN - H0) / abs(H0)
    assert rel_drift < 1e-2, f"3D energy drift {rel_drift} too large"


def test_3d_anisotropic_drag():
    """A length-3 C_d bleeds x, y, z axes independently."""
    space = EventSpace([
        Attractor("a", [3.0, 0.0, 0.0], 0.5),
        Attractor("b", [-3.0, 0.0, 0.0], 0.5),
    ])
    body = Body(mass=1.0, q0=[0.0, 0.0, 0.0], p0=[0.5, 0.5, 0.5])
    sol = simulate(space, body=body, duration=100.0,
                   C_d=np.array([0.05, 0.0, 0.20]))
    p_end = sol["p"][-1]
    # z bleeds fastest (0.20), y not at all, x moderately.
    assert abs(p_end[2]) < abs(p_end[1]), \
        f"z-drag didn't bleed more than y: {p_end}"

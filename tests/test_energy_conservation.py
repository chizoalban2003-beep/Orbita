"""Energy conservation gate and dissipation sanity test."""
from __future__ import annotations

from orbita import Attractor, Body, EventSpace, hamiltonian, simulate


def _make_space():
    return EventSpace([
        Attractor("a", [4.0, 0.0],  0.5),
        Attractor("b", [-4.0, 0.0], 0.5),
    ])


def test_energy_conservation_drag_off():
    """With drag off, the symplectic integrator must conserve H."""
    space = _make_space()
    body = Body(mass=1.0, q0=[0.0, 1.0], p0=[0.3, 0.0])
    sol = simulate(space, body=body, duration=5400.0, C_d=0.0)

    attractors = list(space.attractors)
    H0 = hamiltonian(sol["q"][0],  sol["p"][0],  body.mass, attractors)
    Hf = hamiltonian(sol["q"][-1], sol["p"][-1], body.mass, attractors)
    drift = (Hf - H0) / abs(H0)

    assert abs(drift) < 1e-2, f"Energy drift {drift} exceeds 1e-2"


def test_energy_dissipation_drag_on():
    """With drag on, total energy must strictly decrease."""
    space = _make_space()
    body = Body(mass=1.0, q0=[0.0, 1.0], p0=[0.3, 0.0])
    sol = simulate(space, body=body, duration=5400.0, C_d=0.05)

    attractors = list(space.attractors)
    H0 = hamiltonian(sol["q"][0],  sol["p"][0],  body.mass, attractors)
    Hf = hamiltonian(sol["q"][-1], sol["p"][-1], body.mass, attractors)

    assert Hf < H0, f"Drag failed to dissipate energy: H0={H0}, Hf={Hf}"

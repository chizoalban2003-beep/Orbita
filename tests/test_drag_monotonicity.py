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


def test_callable_drag_schedule_evaluated_at_time():
    """drag_force accepts a callable C_d(t) and evaluates it at t."""
    from orbita.forces import drag_force

    p = np.array([1.0, 1.0])
    m = 1.0
    schedule = lambda t: 0.10 if t < 100.0 else 0.00
    early = drag_force(p, m, schedule, t=50.0)
    late = drag_force(p, m, schedule, t=200.0)
    assert np.allclose(early, [-0.10, -0.10])
    assert np.allclose(late, [0.0, 0.0])


def test_linear_ramp_schedule_endpoints():
    """linear_ramp_schedule interpolates from C_start to C_end."""
    from orbita import linear_ramp_schedule

    sched = linear_ramp_schedule([0.0, 0.0], [0.0, 0.20], duration=100.0)
    assert np.allclose(sched(0.0), [0.0, 0.0])
    assert np.allclose(sched(50.0), [0.0, 0.10])
    assert np.allclose(sched(100.0), [0.0, 0.20])
    # Clamps outside domain.
    assert np.allclose(sched(-10.0), [0.0, 0.0])
    assert np.allclose(sched(500.0), [0.0, 0.20])


def test_piecewise_schedule_stepping():
    """piecewise_constant_schedule holds each knot's value until the next."""
    from orbita import piecewise_constant_schedule

    sched = piecewise_constant_schedule([
        (0.0, np.array([0.0, 0.04])),
        (60.0, np.array([0.0, 0.16])),
        (120.0, np.array([0.0, 0.32])),
    ])
    assert np.allclose(sched(0.0), [0.0, 0.04])
    assert np.allclose(sched(59.99), [0.0, 0.04])
    assert np.allclose(sched(60.0), [0.0, 0.16])
    assert np.allclose(sched(119.9), [0.0, 0.16])
    assert np.allclose(sched(120.0), [0.0, 0.32])
    assert np.allclose(sched(9999.0), [0.0, 0.32])


def test_scheduled_drag_sim_bleeds_more_when_ramped_up():
    """A body simulated with a late-ramping schedule bleeds MORE energy
    than under a constant early-value drag — the ramp actually kicks in."""
    from orbita import linear_ramp_schedule

    space = EventSpace([
        Attractor("home", [4.0, 0.0],  0.5),
        Attractor("away", [-4.0, 0.0], 0.5),
    ])
    body = Body(mass=1.0, q0=[0.0, 0.0], p0=[0.5, 0.5])

    sol_flat = simulate(space, body=body, duration=200.0, C_d=0.02)
    ramp = linear_ramp_schedule(0.02, 0.30, duration=200.0)
    sol_ramp = simulate(space, body=body, duration=200.0, C_d=ramp)

    from orbita import hamiltonian
    attractors = list(space.attractors)
    H_flat = hamiltonian(sol_flat["q"][-1], sol_flat["p"][-1], body.mass, attractors)
    H_ramp = hamiltonian(sol_ramp["q"][-1], sol_ramp["p"][-1], body.mass, attractors)
    assert H_ramp < H_flat, (
        f"Ramping drag up did not bleed more energy: flat={H_flat}, ramp={H_ramp}"
    )


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

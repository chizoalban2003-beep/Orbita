"""Tidal-stretch force (game-state deformation)."""
import numpy as np

from orbita.domain import Attractor, EventSpace
from orbita.forces import tidal_force
from orbita.integrator import simulate_from_state


def test_off_when_kappa_zero():
    q = np.array([4.0, 1.0])
    g = np.array([-3.0, 0.0])
    assert np.allclose(tidal_force(q, g, t=1.0, duration=1.0, kappa=0.0, lam=2.0),
                       [0.0, 0.0])


def test_only_y_component():
    f = tidal_force(np.array([4.0, 1.0]), np.array([-3.0, 0.5]),
                    t=1.0, duration=1.0, kappa=0.1, lam=2.0)
    assert f[0] == 0.0 and f[1] != 0.0          # x untouched, y stretched


def test_direction_follows_y_sign():
    g = np.array([-3.0, 0.0])
    up = tidal_force(np.array([4.0, 1.0]), g, 1.0, 1.0, 0.1, 2.0)
    dn = tidal_force(np.array([4.0, -1.0]), g, 1.0, 1.0, 0.1, 2.0)
    assert up[1] > 0 and dn[1] < 0              # stretches outward from y=0


def test_scales_with_result_decisiveness():
    # bigger x-pull (more decisive result) => bigger stretch
    weak = tidal_force(np.array([1.0, 1.0]), np.array([-1.0, 0.0]), 1.0, 1.0, 0.1, 2.0)
    strong = tidal_force(np.array([1.0, 1.0]), np.array([-5.0, 0.0]), 1.0, 1.0, 0.1, 2.0)
    assert strong[1] > weak[1]


def test_time_ramp_grows_toward_whistle():
    g = np.array([-3.0, 0.0])
    early = tidal_force(np.array([4.0, 1.0]), g, t=0.1, duration=1.0, kappa=0.1, lam=3.0)
    late = tidal_force(np.array([4.0, 1.0]), g, t=1.0, duration=1.0, kappa=0.1, lam=3.0)
    assert late[1] > early[1]                   # desperation scales with time


def test_integrator_tidal_none_reproduces_baseline():
    space = EventSpace([
        Attractor("home_over", [5.0, 3.0], 0.25),
        Attractor("home_under", [5.0, -3.0], 0.25),
        Attractor("away_over", [-5.0, 3.0], 0.25),
        Attractor("away_under", [-5.0, -3.0], 0.25),
    ])
    q0, p0 = np.array([2.0, 0.5]), np.array([0.1, 0.1])
    base = simulate_from_state(space, q0, p0, duration=150.0, C_d=0.04, dt=0.1)
    same = simulate_from_state(space, q0, p0, duration=150.0, C_d=0.04, dt=0.1,
                               tidal=None)
    assert np.allclose(base["q"][-1], same["q"][-1])


def test_integrator_tidal_on_deepens_ou_excursion():
    space = EventSpace([
        Attractor("home_over", [5.0, 3.0], 0.25),
        Attractor("home_under", [5.0, -3.0], 0.25),
        Attractor("away_over", [-5.0, 3.0], 0.25),
        Attractor("away_under", [-5.0, -3.0], 0.25),
    ])
    q0, p0 = np.array([2.0, 0.5]), np.array([0.1, 0.1])   # leans home + over
    base = simulate_from_state(space, q0, p0, duration=150.0, C_d=0.04, dt=0.1)
    tid = simulate_from_state(space, q0, p0, duration=150.0, C_d=0.04, dt=0.1,
                              tidal=(0.05, 3.0))
    # tidal should push the final state further along the (positive) y-axis
    assert tid["q"][-1][1] > base["q"][-1][1]

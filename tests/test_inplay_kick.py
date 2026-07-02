"""In-play kick + simulate_from_state (v0.3.7)."""
from __future__ import annotations

import numpy as np

from orbita import (
    Attractor,
    EventSpace,
    final_well_posterior,
    kick,
    simulate_from_state,
)


def test_kick_is_pure_addition():
    p = np.array([0.1, 0.2])
    p2 = kick(p, np.array([+0.5, -0.1]))
    assert np.allclose(p2, [0.6, 0.1])
    assert np.allclose(p, [0.1, 0.2]), "kick must not mutate input"


def test_simulate_from_state_matches_zero_drag_energy():
    space = EventSpace([
        Attractor("a", [2.0, 0.0], 0.5),
        Attractor("b", [-2.0, 0.0], 0.5),
    ])
    q0 = np.array([0.0, 1.0])
    p0 = np.array([0.3, 0.0])
    sol = simulate_from_state(space, q0=q0, p0=p0, duration=100.0, C_d=0.0,
                              dt=0.02)
    # Energy conservation check.
    from orbita import hamiltonian
    attrs = list(space.attractors)
    H0 = hamiltonian(sol["q"][0], sol["p"][0], 1.0, attrs)
    HN = hamiltonian(sol["q"][-1], sol["p"][-1], 1.0, attrs)
    rel_drift = abs(HN - H0) / abs(H0)
    assert rel_drift < 1e-2, f"energy drift {rel_drift} too large"


def test_kick_toward_home_well_shifts_posterior_home():
    """A +x momentum kick should shift posterior mass toward the +x well."""
    space = EventSpace([
        Attractor("home", [+5.0, 0.0], 0.4),
        Attractor("draw", [ 0.0, 0.0], 0.3),
        Attractor("away", [-5.0, 0.0], 0.3),
    ])
    q0 = np.array([0.0, 0.0])
    p0_baseline = np.array([0.1, 0.05])
    p0_kicked = kick(p0_baseline, np.array([+2.0, 0.0]))

    # No drag — pure ballistic. The kicked body will end further +x
    # (or actually orbiting past) but the *closest* well and its Plummer
    # weighting must lean home vs the baseline.
    sol_base = simulate_from_state(space, q0=q0, p0=p0_baseline,
                                    duration=100.0, C_d=0.0, dt=0.05)
    sol_kick = simulate_from_state(space, q0=q0, p0=p0_kicked,
                                    duration=100.0, C_d=0.0, dt=0.05)
    base = final_well_posterior(sol_base, space)
    kicked = final_well_posterior(sol_kick, space)
    # Take the sign of the shift; magnitude depends on where the orbit
    # ends up. The kicked run should give home strictly more mass than
    # the baseline run.
    assert kicked["home"] > base["home"], (
        f"kick toward home well did not raise home posterior: "
        f"{base['home']:.3f} -> {kicked['home']:.3f}"
    )

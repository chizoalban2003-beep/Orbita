"""Symplectic velocity-Verlet integrator for the Hamiltonian system."""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from .domain import Body, EventSpace
from .forces import gravity_force, drag_force


def simulate(
    space: EventSpace,
    body: Optional[Body] = None,
    duration: float = 5400.0,
    C_d: float = 0.0,
    dt: float = 0.05,
    n_saves: int = 1000,
) -> Dict[str, np.ndarray]:
    """Integrate the Hamiltonian system forward using velocity-Verlet.

    Velocity-Verlet is a 2nd-order symplectic integrator. With ``C_d == 0``
    it conserves total energy to numerical precision over arbitrarily long
    runs (bounded oscillating drift, no secular leak).

    With ``C_d > 0`` the system becomes dissipative; total energy decreases
    monotonically and the body eventually falls into a well.

    Parameters
    ----------
    space : EventSpace
    body : Body, optional
        Initial state. Defaults to origin with small east-and-north momentum.
    duration : float
        Simulation length in seconds (default 5400 = 90 min).
    C_d : float
        Isotropic drag coefficient. Set to 0 for the conservation gate.
    dt : float
        Fixed timestep in seconds.
    n_saves : int
        Number of trajectory points to save (downsampled from dt-stepping).

    Returns
    -------
    dict
        ``'t'`` : (T,) time array
        ``'q'`` : (T, 2) positions
        ``'p'`` : (T, 2) momenta
    """
    if body is None:
        body = Body(p0=np.array([0.5, 0.3]))

    attractors = list(space.attractors)
    n_steps = int(duration / dt)
    save_every = max(1, n_steps // n_saves)
    n_out = (n_steps // save_every) + 1

    q = body.q0.copy()
    p = body.p0.copy()
    m = body.mass

    t_out = np.zeros(n_out)
    q_out = np.zeros((n_out, 2))
    p_out = np.zeros((n_out, 2))

    q_out[0] = q
    p_out[0] = p
    out_i = 1

    # Velocity-Verlet:
    #   p_{n+½} = p_n   + (dt/2) · F(q_n, p_n)
    #   q_{n+1} = q_n   + dt · p_{n+½} / m
    #   p_{n+1} = p_{n+½} + (dt/2) · F(q_{n+1}, p_{n+½})
    F = m * gravity_force(q, attractors) + drag_force(p, m, C_d)

    for step in range(1, n_steps + 1):
        p_half = p + 0.5 * dt * F
        q = q + dt * p_half / m
        F = m * gravity_force(q, attractors) + drag_force(p_half, m, C_d)
        p = p_half + 0.5 * dt * F

        if step % save_every == 0 and out_i < n_out:
            t_out[out_i] = step * dt
            q_out[out_i] = q
            p_out[out_i] = p
            out_i += 1

    return {
        "t": t_out[:out_i],
        "q": q_out[:out_i],
        "p": p_out[:out_i],
    }


def final_well(sol: Dict[str, np.ndarray], space: EventSpace) -> str:
    """Return the label of the attractor closest to the body's final position.

    Phase 2 will replace this with a soft Bayesian assignment based on local
    well depth and trajectory curvature.
    """
    q_end = sol["q"][-1]
    dists = [float(np.linalg.norm(q_end - a.position)) for a in space.attractors]
    return space.attractors[int(np.argmin(dists))].label

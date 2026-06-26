"""Symplectic velocity-Verlet integrator for the Hamiltonian system."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .domain import Body, EventSpace, Observation, Sensor
from .forces import gravity_force, drag_force


def simulate(
    space: EventSpace,
    body: Optional[Body] = None,
    duration: float = 5400.0,
    C_d: float = 0.0,
    dt: float = 0.05,
    n_saves: int = 1000,
    sensors: Optional[Sequence[Sensor]] = None,
    observations: Optional[Sequence[Observation]] = None,
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

    # Sensor layer (issue #2): sort observations by time, walk them in
    # lockstep with the integrator. Empty if not provided — zero overhead.
    sensors_by_name = {s.name: s for s in (sensors or [])}
    obs_sorted: List[Observation] = sorted(observations or [], key=lambda o: o.t)
    obs_idx = 0

    # Velocity-Verlet:
    #   p_{n+½} = p_n   + (dt/2) · F(q_n, p_n)
    #   q_{n+1} = q_n   + dt · p_{n+½} / m
    #   p_{n+1} = p_{n+½} + (dt/2) · F(q_{n+1}, p_{n+½})
    F = m * gravity_force(q, attractors) + drag_force(p, m, C_d)

    for step in range(1, n_steps + 1):
        p_half = p + 0.5 * dt * F
        q = q + dt * p_half / m

        # Apply any observations whose timestamp falls in (t-dt, t].
        # This mutates the well field — the gravity field on the next
        # step reflects the new posterior masses.
        t_now = step * dt
        while obs_idx < len(obs_sorted) and obs_sorted[obs_idx].t <= t_now:
            obs = obs_sorted[obs_idx]
            sensor = sensors_by_name.get(obs.sensor)
            if sensor is None:
                raise KeyError(
                    f"Observation references sensor {obs.sensor!r} which "
                    f"is not in the sensors list."
                )
            space.apply_observation(obs, sensor)
            obs_idx += 1
        attractors = list(space.attractors)

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

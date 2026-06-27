"""Symplectic velocity-Verlet integrator for the Hamiltonian system."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .domain import Body, EventSpace, Observation, Sensor
from .forces import SOFTENING, gravity_force, drag_force


def simulate(
    space: EventSpace,
    body: Optional[Body] = None,
    duration: float = 5400.0,
    C_d: float = 0.0,
    dt: float = 0.05,
    n_saves: int = 1000,
    sensors: Optional[Sequence[Sensor]] = None,
    observations: Optional[Sequence[Observation]] = None,
    ic_scale: float = 1.0,
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
    ic_scale : float
        Carried through from sport templates; consumed by Monte Carlo
        harnesses that sample initial conditions. Not used by the
        integrator itself — accepted so ``simulate(**sim_kwargs)`` works
        when a template-driven harness includes it in the kwargs dict.

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

    confidence = space.confidence(q_out[out_i - 1])
    return {
        "t": t_out[:out_i],
        "q": q_out[:out_i],
        "p": p_out[:out_i],
        "confidence": float(confidence),
    }


def final_well(sol: Dict[str, np.ndarray], space: EventSpace) -> str:
    """Return the label of the attractor closest to the body's final position.

    Hard nearest-well classification — useful for visualisation and the
    legacy demos. For probability accumulation across many MC trials,
    prefer :func:`final_well_posterior`, which spreads each trial's
    contribution across wells in proportion to mass-weighted Plummer
    attraction at the final state. Hard-classify is the v0.2 default
    and the source of the v0.2 over-sharpening on multi-outcome sports
    (see ``experiments/01_sharpening_triage.py``).
    """
    q_end = sol["q"][-1]
    dists = [float(np.linalg.norm(q_end - a.position)) for a in space.attractors]
    return space.attractors[int(np.argmin(dists))].label


def final_well_posterior(
    sol: Dict[str, np.ndarray],
    space: EventSpace,
    alpha: float = 2.0,
) -> Dict[str, float]:
    """Soft assignment of the final state to wells (v0.3.3).

    Each well receives probability proportional to
    ``mass_k / r_k**alpha``, where ``r_k`` is the Plummer-softened
    distance from the body's final position to well ``k``. ``alpha``
    controls sharpness:

    * ``alpha = 0`` — pure mass prior (the final state contributes
      no information).
    * ``alpha = 1`` — proportional to gravitational potential magnitude.
    * ``alpha = 2`` — proportional to gravitational force magnitude
      (default; the physically natural choice).
    * ``alpha -> inf`` — collapses to :func:`final_well` (hard
      nearest-well classification).

    Returns a dict ``label -> probability`` summing to 1.0.
    """
    q_end = sol["q"][-1]
    weights: Dict[str, float] = {}
    for a in space.attractors:
        d2 = float(np.sum((q_end - a.position) ** 2)) + SOFTENING ** 2
        d = float(np.sqrt(d2))
        weights[a.label] = a.mass / (d ** alpha)
    total = sum(weights.values())
    if total <= 0:
        # Degenerate (alpha=0 + zero mass) — fall back to uniform.
        n = len(weights)
        return {label: 1.0 / n for label in weights}
    return {label: w / total for label, w in weights.items()}

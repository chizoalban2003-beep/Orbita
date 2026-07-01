"""Gravitational and drag forces."""
from __future__ import annotations

from typing import Iterable

import numpy as np

G: float = 1.0           # normalized gravitational constant
SOFTENING: float = 0.5   # avoid singularity at the well bottom — sized so
                         # close approaches stay resolvable at default dt


def gravity_force(q: np.ndarray, attractors: Iterable) -> np.ndarray:
    """Sum of Newtonian forces from every attractor on a body at position ``q``.

    Uses Plummer softening so the force stays finite as the body approaches
    a well centre.
    """
    F = np.zeros(2)
    for a in attractors:
        r = a.position - q
        d2 = float(r @ r) + SOFTENING ** 2
        F += G * a.mass * r / d2 ** 1.5
    return F


def potential_energy(q: np.ndarray, attractors: Iterable) -> float:
    """Per-unit-mass gravitational potential at ``q``."""
    U = 0.0
    for a in attractors:
        d = float(np.sqrt(np.sum((q - a.position) ** 2) + SOFTENING ** 2))
        U -= G * a.mass / d
    return U


def drag_force(p: np.ndarray, body_mass: float, C_d) -> np.ndarray:
    """Linear drag — the simplest non-conservative force.

    ``C_d`` may be:

    * a scalar — isotropic drag (the v0.1 default).
    * a length-2 array — anisotropic drag with separate coefficients
      along the x- and y-axes of event space. Useful when the event
      geometry has a directional asymmetry (e.g. soccer's home/away
      axis carries directional momentum, the draw axis does not).

    Note: enabling drag turns the system from conservative to dissipative.
    The energy-conservation validation gate must be run with ``C_d == 0``.
    """
    coef = np.asarray(C_d, dtype=float)
    if coef.ndim == 0:
        return -float(coef) * (p / body_mass)
    return -coef * (p / body_mass)


def hamiltonian(
    q: np.ndarray, p: np.ndarray, body_mass: float, attractors: Iterable
) -> float:
    """Total energy of the system.

    Conserved (to numerical precision) by symplectic integrators when drag
    is off. The validation gate uses this.
    """
    KE = float(p @ p) / (2 * body_mass)
    PE = body_mass * potential_energy(q, attractors)
    return KE + PE

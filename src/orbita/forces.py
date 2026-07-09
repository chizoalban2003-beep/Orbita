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
    a well centre. Works in any dimension >= 2 — the loop is coordinate-free.
    """
    F = np.zeros_like(q)
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


def drag_force(p: np.ndarray, body_mass: float, C_d, t: float = 0.0) -> np.ndarray:
    """Linear drag — the simplest non-conservative force.

    ``C_d`` may be:

    * a scalar — isotropic drag (the v0.1 default).
    * a length-2 array — anisotropic drag with separate coefficients
      along the x- and y-axes of event space. Useful when the event
      geometry has a directional asymmetry (e.g. soccer's home/away
      axis carries directional momentum, the draw axis does not).
    * a callable ``C_d(t) -> scalar | length-2 array`` — time-varying
      drag schedule. Physically meaningful for events with non-uniform
      time structure: fatigue accumulates late, desperation reverses
      near the whistle, etc. The integrator evaluates ``C_d`` at the
      appropriate half-step time — leapfrog remains 2nd-order accurate.

    Note: enabling drag turns the system from conservative to dissipative.
    The energy-conservation validation gate must be run with ``C_d == 0``.
    """
    if callable(C_d):
        C_d = C_d(t)
    coef = np.asarray(C_d, dtype=float)
    if coef.ndim == 0:
        return -float(coef) * (p / body_mass)
    return -coef * (p / body_mass)


def favourite_lock_force(q, v, fav_pos, strength: float) -> np.ndarray:
    """Directional 'game-management' drag — the leading side holds a lead.

    A favourite protecting a lead does not damp *all* motion (that is scalar
    drag, which bleeds the state into the central draw well — see
    ``experiments/23_drag_lowtempo_validity.py``). It selectively kills velocity
    that carries the match state *away* from its own winning well — the
    opponent's counter-attack — while leaving safe, possession motion toward the
    well untouched.

    That asymmetry is intrinsically **nonlinear** in ``v``. A linear drag tensor
    ``F = −C·v`` is symmetric under ``v → −v``: it damps advancing toward the
    well and retreating from it by the same magnitude, so it can *freeze* motion
    along an axis but never *bias* it — it cannot hold a lead. Hence a sign-gated
    radial projection, not a tensor::

        r̂        = (fav_pos − q) / |fav_pos − q|      # toward the favourite's well
        v_radial  = v · r̂                              # < 0  ⇒ drifting away (losing the lead)
        F         = −strength · min(v_radial, 0) · r̂   # push back only while retreating

    Vanishes when advancing toward the well (``v_radial ≥ 0``) and when
    ``strength == 0`` (the identity that keeps un-locked dynamics reproducible).
    Broadcasts over a leading batch axis, so it serves both the single-body
    integrator and the vectorised forecaster.
    """
    if strength == 0.0:
        return np.zeros_like(np.asarray(q, dtype=float))
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    r = np.asarray(fav_pos, dtype=float) - q
    dist = np.sqrt((r * r).sum(axis=-1, keepdims=True) + SOFTENING ** 2)
    r_hat = r / dist
    v_radial = (v * r_hat).sum(axis=-1, keepdims=True)
    gate = np.minimum(v_radial, 0.0)                    # only the retreating part
    return -strength * gate * r_hat


def tidal_force(
    q: np.ndarray,
    grav_F: np.ndarray,
    t: float,
    duration: float,
    kappa: float,
    lam: float,
) -> np.ndarray:
    """Game-state tidal stretch along the goals (y) axis.

    A body approaching a *decisive* result well (a home/away win, on the
    x-axis) late in the event experiences a tidal force: the structural
    integrity of the match state stretches along the orthogonal
    goals-axis, deepening whichever over/under well the state already
    leans toward. This is the mechanism a static closing line struggles
    to price — the non-linear tipping point of a chasing team abandoning
    shape as time runs out.

    Magnitude couples two things:

    * **result decisiveness** — ``|grav_F[0]|``, the x-component of the
      gravitational pull. Large when the body is committed toward a
      home/away well, ~0 near a still-open (central) state. This is the
      discrete analogue of ``∂U_wdl/∂r`` in the hypothesis: the harder
      the result is being decided, the stronger the tidal strain.
    * **time pressure** — ``exp(lam·(t/duration − 1))``, an exponential
      desperation ramp: ≈ ``e^{-lam}`` at kickoff, exactly 1 at the final
      whistle. Desperation scales exponentially as time runs out.

    Direction is ``sign(q_y)`` — it stretches the body *further* along the
    axis it already occupies, so the reached over/under well is deepened
    and widened rather than shifted. Vanishes at ``q_y = 0`` (an undecided
    goals state) and at ``kappa = 0`` (force off — the identity that keeps
    every pre-tidal result reproducible).

    Returns a force vector with only a y-component; x is untouched so the
    result-axis dynamics are unchanged.
    """
    out = np.zeros_like(q)
    if kappa == 0.0:
        return out
    ramp = np.exp(lam * (t / duration - 1.0))
    strain = abs(float(grav_F[0]))
    out[1] = kappa * strain * ramp * np.sign(q[1])
    return out


def linear_ramp_schedule(C_start, C_end, duration: float):
    """Piecewise-linear drag schedule from ``C_start`` at t=0 to ``C_end`` at
    t=duration.

    Both endpoints may be scalar or length-2 array. Returned closure is
    passed as ``C_d=`` to :func:`orbita.simulate` or the fast Verlet kernel.
    """
    C_start = np.asarray(C_start, dtype=float)
    C_end = np.asarray(C_end, dtype=float)
    inv_d = 1.0 / duration

    def schedule(t: float):
        u = min(max(t * inv_d, 0.0), 1.0)
        return C_start * (1.0 - u) + C_end * u
    return schedule


def piecewise_constant_schedule(knots):
    """Piecewise-constant drag schedule.

    ``knots`` is an iterable of ``(t_start, C_d)`` pairs sorted by
    ``t_start``; the last pair applies for all ``t >= t_start``. E.g.::

        piecewise_constant_schedule([
            (0.0,    np.array([0.0, 0.08])),   # first hour: light y-drag
            (3600.0, np.array([0.0, 0.24])),   # last 30 min: heavy y-drag
        ])
    """
    knots = sorted(knots, key=lambda kv: kv[0])
    ts = np.array([k[0] for k in knots])
    vals = [np.asarray(k[1], dtype=float) for k in knots]

    def schedule(t: float):
        # rightmost knot with t_start <= t
        i = int(np.searchsorted(ts, t, side="right") - 1)
        i = max(0, min(i, len(vals) - 1))
        return vals[i]
    return schedule


def ornstein_uhlenbeck_schedule(
    C_mean,
    theta: float,
    sigma: float,
    dt: float,
    seed: int | None = None,
    clip_min: float = 0.0,
):
    """Stochastic drag: Ornstein-Uhlenbeck process around a mean coefficient.

    ``C(t+dt) = C(t) + theta * (C_mean - C(t)) * dt + sigma * sqrt(dt) * eta``

    where ``eta ~ N(0, I)``. Returned closure caches the last-evaluated
    ``(t, C)`` and steps forward only when ``t`` advances — leapfrog calls
    ``C_d(t)`` twice per step (once at t_now, once at t_now again for the
    half-step momentum update) and we want both calls to see the same
    stochastic realisation.

    Breaks symplecticity (drag is non-conservative *and* now noisy).
    Solver becomes Euler-Maruyama-flavoured Verlet: the leapfrog structure
    on the deterministic part remains 2nd-order in dt, and the stochastic
    perturbation is O(sqrt(dt)) as expected for an SDE.

    Parameters
    ----------
    C_mean : scalar or length-2 array
        Long-run mean drag.
    theta : float
        Reversion rate. theta * duration >~ 3 for the process to sample its
        stationary distribution over a match.
    sigma : float
        Noise volatility per unit time.
    dt : float
        Integrator step size; used to size the noise increment.
    seed : int, optional
        RNG seed. If None, uses the global default_rng — sim is not
        deterministic.
    clip_min : float
        Lower clip on the drag coefficient (negative drag = thrust; the
        engine supports it but at high magnitudes it can blow up).
    """
    C_mean = np.asarray(C_mean, dtype=float)
    rng = np.random.default_rng(seed)
    sqrt_dt = float(np.sqrt(dt))
    state = {"t": -1.0, "C": C_mean.copy()}

    def schedule(t: float):
        if t > state["t"] + 0.5 * dt:
            # step the SDE forward
            eta = rng.standard_normal(size=C_mean.shape)
            state["C"] = (
                state["C"] + theta * (C_mean - state["C"]) * dt
                + sigma * sqrt_dt * eta
            )
            state["C"] = np.maximum(state["C"], clip_min)
            state["t"] = t
        return state["C"]
    return schedule


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

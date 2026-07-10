"""Counterfactual interventions — the white-box core.

Orbita's advantage over a black-box forecaster is that its parameters are
*named forces*, so a user can ask a legible what-if — "the striker is out",
"they'll park the bus", "the home side starts on top" — and watch the
forecast move for a reason you can state in one sentence.

An :class:`Intervention` maps such a question onto the engine's physical
levers. A structural law governs which levers actually work (see
``docs/injury_mass_result.md``): the draw well sits at the geometric centre,
so **scalar** operations (uniform mass-cut, uniform drag) leak probability
into the draw — which reality does not do — while **directional** operations
move probability along the result axis the way real matches do. Every lever
below is therefore directional:

    * **mass transfer** — a weakening moves win-probability from one side to
      the *opponent* (not symmetrically into the draw). Injury / suspension.
      Validated as the result-axis re-spec in exp 24.
    * **momentum**      — who starts on top: an initial push toward a side's
      well. Early pressure; the extra "pinned back" shove of a red card.
      Validated on red cards in exp 22.
    * **favourite-lock** — a sign-gated directional drag: the leading side
      resists the state drifting *away* from its well (killing the
      counter-attack) without the uniform damping that a low-tempo game was
      wrongly modelled with. Re-spec of the rejected scalar drag (exp 23);
      pending its own interventional-validity backtest.

:func:`reprice` runs the Monte-Carlo forecast with and without the
intervention and returns both distributions plus a one-sentence
explanation of what moved and why. Faithful by construction: the
explanation *is* the mechanism, not a post-hoc approximation of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .domain import Attractor, EventSpace
from .forces import SOFTENING, favourite_lock_force
from .integrator import simulate_from_state

# soccer 3-well geometry (matches the sport template)
_POS = {"home": np.array([5.0, 0.0]),
        "draw": np.array([0.0, 5.0]),
        "away": np.array([-5.0, 0.0])}
_C_D = 0.04
_DURATION = 600.0
_IC_SCALE = 2.5


@dataclass
class Intervention:
    """A named what-if expressed in the engine's physical levers."""

    name: str
    description: str
    mass_scale: Dict[str, float] = field(default_factory=dict)  # per-outcome ×
    # (from_side, to_side, fraction): move `fraction` of from's mass to `to`,
    # the result-axis weakening — leaves the draw untouched (exp 24).
    mass_transfer: Optional[Tuple[str, str, float]] = None
    drag_scale: float = 1.0                                      # global C_d ×
    momentum: Optional[np.ndarray] = None                       # added to p0
    # (side | "favourite", strength): directional favourite-lock drag.
    lock: Optional[Tuple[str, float]] = None

    def __post_init__(self):
        if self.momentum is not None:
            self.momentum = np.asarray(self.momentum, dtype=float)


# ---- named intervention factories (the user-facing vocabulary) -----------

def injury(side: str, severity: float = 0.14) -> Intervention:
    """A key player out: transfer ``severity`` of that side's win-probability
    to the *opponent* (a result-axis weakening, draw untouched — the mechanism
    validated in exp 24; a symmetric mass-cut leaks into the draw and fails).
    Default is the Bayesian-calibrated magnitude (exp 26, posterior mean ≈0.126
    on 60 drift matches)."""
    opponent = "home" if side == "away" else "away"
    return Intervention(
        name=f"injury:{side}",
        description=f"{side} weakened by {severity:.0%} (key player out)",
        mass_transfer=(side, opponent, severity))


def red_card(side: str, severity: float = 0.15,
             pressure: float = 0.16) -> Intervention:
    """A sending-off: transfer the side's mass to the opponent AND push
    momentum toward the opponent (10 men get pinned back and concede). Both
    levers are directional, so it hurts the carded side more than a like-sized
    injury (which is the transfer alone). Defaults are the Bayesian-calibrated
    magnitude from experiment 26 (posterior mean ≈0.145 on 60 real single-card
    matches; was an illustrative 0.25)."""
    opponent = "home" if side == "away" else "away"
    direction = _POS[opponent] / np.linalg.norm(_POS[opponent])
    return Intervention(
        name=f"red_card:{side}",
        description=(f"{side} down to 10 men (−{severity:.0%} toward "
                     f"{opponent}, pinned back)"),
        mass_transfer=(side, opponent, severity), momentum=pressure * direction)


def low_tempo(strength: float = 0.15) -> Intervention:
    """A cagey, controlled game where the favourite kills tempo to hold the
    result: a directional favourite-lock that resists the state drifting away
    from the pre-match favourite's well. Re-spec of the rejected scalar drag
    (exp 23) — it lifts the *favourite*, not the draw. The favourite is
    resolved from the priors at forecast time. Direction is validated by the
    structural law; the default ``strength`` is illustrative and awaits its own
    interventional-validity calibration."""
    return Intervention(
        name="low_tempo",
        description=f"favourite kills the tempo to hold the result "
                    f"(lock {strength:.2f})",
        lock=("favourite", strength))


def early_pressure(side: str, magnitude: float = 0.6) -> Intervention:
    """One side starts on top: an initial momentum push toward its well."""
    target = _POS[side]
    direction = target / np.linalg.norm(target)
    return Intervention(
        name=f"early_pressure:{side}",
        description=f"{side} starts on top (early momentum {magnitude:+.1f})",
        momentum=magnitude * direction)


# ---- the repricer --------------------------------------------------------

def _space(priors: Dict[str, float]) -> EventSpace:
    return EventSpace([Attractor(k, _POS[k], priors[k]) for k in ("home", "draw", "away")])


def _apply(priors: Dict[str, float], iv: Intervention) -> Dict[str, float]:
    scaled = {k: priors[k] * iv.mass_scale.get(k, 1.0) for k in priors}
    if iv.mass_transfer is not None:
        frm, to, frac = iv.mass_transfer
        moved = scaled[frm] * frac
        scaled[frm] -= moved
        scaled[to] += moved
    s = sum(scaled.values())
    return {k: v / s for k, v in scaled.items()}


_LABELS = ("home", "draw", "away")


def _resolve_lock(priors: Dict[str, float], iv: Optional[Intervention]):
    """Return ``(fav_pos, strength)`` for the integrator, or ``None``.
    A ``"favourite"`` side is resolved to the stronger of home/away."""
    if iv is None or iv.lock is None:
        return None
    side, strength = iv.lock
    if side == "favourite":
        side = "home" if priors["home"] >= priors["away"] else "away"
    return _POS[side], strength


def _batch_forecast(priors, C_d, mom, n_trials, seed, dt=0.1, alpha=2.0,
                    lock=None):
    """Vectorised MC H/D/A forecast (bodies batched) — fast enough for an
    interactive API. Same velocity-Verlet + soft posterior as the canonical
    path, one numpy op per step instead of a Python loop over wells.

    ``lock``, when given, is a ``(fav_pos, strength)`` pair adding the
    directional favourite-lock drag (body mass is 1 here, so velocity = p)."""
    pos = np.array([_POS[k] for k in _LABELS], float)
    mass = np.array([priors[k] for k in _LABELS], float)
    mass = mass / mass.sum()
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=np.array([0.3, 0.2]) * _IC_SCALE, size=(n_trials, 2))
    p = rng.normal(scale=np.array([0.15, 0.15]) * _IC_SCALE, size=(n_trials, 2)) + mom
    soft2 = SOFTENING ** 2
    m_col = mass[None, :, None]
    n_steps = int(_DURATION / dt)

    def force(qb, pb):
        r = pos[None, :, :] - qb[:, None, :]
        d2 = np.einsum("twk,twk->tw", r, r) + soft2
        g = (m_col * r / d2[:, :, None] ** 1.5).sum(axis=1)
        F = g - C_d * pb
        if lock is not None:
            F = F + favourite_lock_force(qb, pb, lock[0], lock[1])
        return F

    F = force(q, p)
    for _ in range(n_steps):
        p_half = p + 0.5 * dt * F
        q = q + dt * p_half
        F = force(q, p_half)
        p = p_half + 0.5 * dt * F
    diff = pos[None, :, :] - q[:, None, :]
    d2 = np.einsum("twk,twk->tw", diff, diff) + soft2
    w = mass[None, :] / np.sqrt(d2) ** alpha
    w = w / w.sum(axis=1, keepdims=True)
    probs = w.mean(axis=0)
    return {k: float(probs[i]) for i, k in enumerate(_LABELS)}


def forecast(priors: Dict[str, float], iv: Optional[Intervention] = None,
             n_trials: int = 200, seed: int = 42) -> Dict[str, float]:
    """Monte-Carlo H/D/A forecast, optionally under an intervention."""
    p = _apply(priors, iv) if iv else dict(priors)
    C_d = _C_D * (iv.drag_scale if iv else 1.0)
    mom = iv.momentum if (iv and iv.momentum is not None) else np.zeros(2)
    lock = _resolve_lock(priors, iv)
    return _batch_forecast(p, C_d, mom, n_trials, seed, lock=lock)


def reprice(priors: Dict[str, float], iv: Intervention,
            n_trials: int = 200, seed: int = 42
            ) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """Return (before, after, explanation) for one intervention."""
    before = forecast(priors, None, n_trials, seed)
    after = forecast(priors, iv, n_trials, seed)
    mover = max(before, key=lambda k: abs(after[k] - before[k]))
    delta = after[mover] - before[mover]
    expl = (f"{iv.description}: P({mover}) {delta:+.1%} "
            f"({before[mover]:.0%}→{after[mover]:.0%}). "
            + ", ".join(f"{k} {after[k]:.0%}" for k in ("home", "draw", "away")))
    return before, after, expl


def trajectory(priors: Dict[str, float], iv: Optional[Intervention] = None,
               seed: int = 7, n_saves: int = 400) -> np.ndarray:
    """One representative orbit (fixed IC) for visualisation — base vs
    intervened paths bend visibly through the same well field."""
    p = _apply(priors, iv) if iv else dict(priors)
    space = _space(p)
    C_d = _C_D * (iv.drag_scale if iv else 1.0)
    mom = iv.momentum if (iv and iv.momentum is not None) else np.zeros(2)
    lock = _resolve_lock(priors, iv)
    rng = np.random.default_rng(seed)
    q0 = rng.normal(scale=np.array([0.3, 0.2]) * _IC_SCALE)
    p0 = rng.normal(scale=np.array([0.15, 0.15]) * _IC_SCALE) + mom
    sol = simulate_from_state(space, q0, p0, duration=_DURATION, C_d=C_d,
                              dt=0.1, n_saves=n_saves, lock=lock)
    return sol["q"]

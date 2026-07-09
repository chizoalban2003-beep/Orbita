"""Counterfactual interventions — the white-box core.

Orbita's advantage over a black-box forecaster is that its parameters are
*named forces*, so a user can ask a legible what-if — "the striker is out",
"they'll park the bus", "the home side starts on top" — and watch the
forecast move for a reason you can state in one sentence.

An :class:`Intervention` maps such a question onto the three physical
levers the engine exposes:

    * **well mass**  — an outcome's likelihood / a side's quality. An
      injury or suspension shrinks a side's win-well mass.
    * **drag**       — how fast the match state settles. More drag = a
      controlled, low-event game that favours the central (draw) basin and
      the prior leader; less drag = chaos and upsets.
    * **momentum**   — who starts on top. An initial push toward a side's
      well models early territorial pressure.

:func:`reprice` runs the Monte-Carlo forecast with and without the
intervention and returns both distributions plus a one-sentence
explanation of what moved and why. Faithful by construction: the
explanation *is* the mechanism, not a post-hoc approximation of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .domain import Attractor, Body, EventSpace
from .forces import SOFTENING
from .integrator import simulate

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
    mass_scale: Dict[str, float] = field(default_factory=dict)  # per-outcome
    drag_scale: float = 1.0                                      # global C_d ×
    momentum: Optional[np.ndarray] = None                       # added to p0

    def __post_init__(self):
        if self.momentum is not None:
            self.momentum = np.asarray(self.momentum, dtype=float)


# ---- named intervention factories (the user-facing vocabulary) -----------

def injury(side: str, severity: float = 0.20) -> Intervention:
    """A key player out: shrink that side's win-well mass by ``severity``."""
    return Intervention(
        name=f"injury:{side}",
        description=f"{side} weakened by {severity:.0%} (key player out)",
        mass_scale={side: 1.0 - severity})


def red_card(side: str, severity: float = 0.35,
             pressure: float = 0.5) -> Intervention:
    """A sending-off: shrink the side's mass AND push momentum toward the
    opponent (10 men get pinned back and concede), so it hurts the carded
    side directionally — more than a like-sized injury."""
    opponent = "home" if side == "away" else "away"
    direction = _POS[opponent] / np.linalg.norm(_POS[opponent])
    return Intervention(
        name=f"red_card:{side}",
        description=(f"{side} down to 10 men (−{severity:.0%} mass, "
                     f"pinned back toward {opponent})"),
        mass_scale={side: 1.0 - severity}, momentum=pressure * direction)


def low_tempo(scale: float = 1.5) -> Intervention:
    """A cagey, controlled game: raise drag so the state settles early."""
    return Intervention(
        name="low_tempo",
        description=f"controlled low-event game (drag ×{scale:.2f})",
        drag_scale=scale)


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
    s = sum(scaled.values())
    return {k: v / s for k, v in scaled.items()}


_LABELS = ("home", "draw", "away")


def _batch_forecast(priors, C_d, mom, n_trials, seed, dt=0.1, alpha=2.0):
    """Vectorised MC H/D/A forecast (bodies batched) — fast enough for an
    interactive API. Same velocity-Verlet + soft posterior as the canonical
    path, one numpy op per step instead of a Python loop over wells."""
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
        return g - C_d * pb

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
    return _batch_forecast(p, C_d, mom, n_trials, seed)


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
    rng = np.random.default_rng(seed)
    q0 = rng.normal(scale=np.array([0.3, 0.2]) * _IC_SCALE)
    p0 = rng.normal(scale=np.array([0.15, 0.15]) * _IC_SCALE) + mom
    sol = simulate(space, Body(q0=q0, p0=p0), duration=_DURATION, C_d=C_d,
                   dt=0.1, n_saves=n_saves)
    return sol["q"]

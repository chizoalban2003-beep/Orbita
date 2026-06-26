"""Core types: attractors, event spaces, and orbiting bodies."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class Attractor:
    """A possible outcome of the event, represented as a gravitational well.

    The ``mass`` field is the prior probability before any live data has been
    seen. The calibration loop updates it from observed outcomes.
    """

    label: str
    position: np.ndarray
    mass: float

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (2,):
            raise ValueError(
                f"position must be shape (2,), got {self.position.shape}"
            )


class EventSpace:
    """The set of possible outcomes plus the geometry they live in.

    Masses are renormalized to sum to 1.0 so they remain interpretable as a
    probability simplex.
    """

    def __init__(self, attractors: Iterable[Attractor]) -> None:
        attractors = list(attractors)
        total = sum(a.mass for a in attractors)
        if not np.isclose(total, 1.0, atol=1e-6):
            warnings.warn(
                f"Attractor masses sum to {total}, not 1.0. Renormalizing.",
                stacklevel=2,
            )
            attractors = [
                Attractor(a.label, a.position, a.mass / total) for a in attractors
            ]
        self.attractors: List[Attractor] = attractors

    def __iter__(self):
        return iter(self.attractors)

    def __len__(self) -> int:
        return len(self.attractors)

    # ---- sensor layer (issue #2) ---------------------------------------

    def renormalize(self) -> None:
        """Rescale masses to sum to 1.0 in-place.

        Sensor updates multiply attractor masses; this restores the simplex
        invariant. Idempotent.
        """
        total = sum(a.mass for a in self.attractors)
        if total <= 0:
            raise ValueError("Total mass collapsed to <= 0; check sensor "
                             "likelihoods.")
        for a in self.attractors:
            a.mass = a.mass / total

    def apply_observation(
        self, obs: "Observation", sensor: "Sensor"
    ) -> None:
        """Mutate the targeted well's mass according to the sensor's
        likelihood, then renormalize.

        Sensor likelihood is interpreted as a mass *multiplier*: values
        above 1 grow the well, below 1 shrink it.
        """
        target = sensor.target
        found = False
        for a in self.attractors:
            if a.label == target:
                a.mass *= sensor.likelihood(obs.value)
                found = True
                break
        if not found:
            raise KeyError(
                f"Sensor {sensor.name!r} targets well {target!r}, but no "
                f"such attractor exists in this EventSpace."
            )
        self.renormalize()


# ---------------------------------------------------------------------------
# Sensor layer (Phase-2, issue #2)
# ---------------------------------------------------------------------------

@dataclass
class Sensor:
    """A streamed observation source that updates a well's mass posterior.

    The ``likelihood`` callable takes the raw observation value and returns
    a *mass multiplier* (>1 grows the well, <1 shrinks it). For binary
    events use a constant multiplier; for continuous signals use a
    Gaussian kernel or similar.

    Sensors are NOT bodies and NOT wells — they are measurements about
    the field. See issue #2 for the rationale.
    """

    name: str
    target: str
    likelihood: Callable[[float], float]


@dataclass(order=True)
class Observation:
    """A single streamed reading: ``sensor`` saw ``value`` at time ``t``."""

    t: float
    sensor: str
    value: float


@dataclass
class Body:
    """The live event state, orbiting through the event space."""

    mass: float = 1.0
    q0: np.ndarray = None  # type: ignore[assignment]
    p0: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.q0 is None:
            self.q0 = np.zeros(2)
        if self.p0 is None:
            self.p0 = np.zeros(2)
        self.q0 = np.asarray(self.q0, dtype=float)
        self.p0 = np.asarray(self.p0, dtype=float)
        if self.q0.shape != (2,) or self.p0.shape != (2,):
            raise ValueError("q0 and p0 must each be shape (2,)")


# ---------------------------------------------------------------------------
# Roster layer (Phase-2 minimum viable — see GitHub issue #1)
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """An athlete or competitor whose presence reshapes a well's mass.

    Reserve ``team`` for the well label this player contributes to (e.g.
    ``\"france_win\"``). Sports without teams (tennis, boxing) use the player's
    own win-label as the team.
    """

    name: str
    team: str
    rating: float          # 0-100; aggregate skill/form going into the event
    available: bool = True


@dataclass
class Roster:
    """Bundle of all players on both sides of a head-to-head event.

    The roster does not become a body — it modifies the *masses* of the
    outcome wells via :meth:`well_mass_multiplier`. See issue #1 for the
    rationale ("players are field perturbations, not orbiting objects").
    """

    players: List[Player] = field(default_factory=list)

    def strength(self, team_label: str) -> float:
        """Mean rating of available players on this side, in [0, 1].

        Returns 0.0 if every player on the side is unavailable (an edge
        case; in practice the caller would forfeit the event).
        """
        active = [p.rating for p in self.players
                  if p.team == team_label and p.available]
        if not active:
            return 0.0
        return float(np.mean(active)) / 100.0

    def well_mass_multiplier(
        self, target: str, opponent: str, share: float = 1.0
    ) -> float:
        """How much to scale the ``target`` win-well's mass.

        Uses normalised relative strength so the multiplier hovers around
        1.0 when both sides are equally stocked, climbs above 1 when
        ``target`` is the stronger side, and falls below 1 when weaker.
        ``share`` lets the caller down-weight the roster effect (e.g.
        0.3 = "roster only counts for 30% of well mass adjustment").
        """
        s_t = self.strength(target)
        s_o = self.strength(opponent)
        if s_t + s_o == 0:
            return 1.0
        ratio = 2 * s_t / (s_t + s_o)   # in [0, 2]
        return 1.0 + share * (ratio - 1.0)


def event_space_from_rosters(
    base_priors: Dict[str, float],
    positions: Dict[str, Iterable[float]],
    roster: Optional[Roster] = None,
    head_to_head: Optional[tuple] = None,
    roster_share: float = 1.0,
) -> "EventSpace":
    """Build an :class:`EventSpace` whose well masses are scaled by a roster.

    Parameters
    ----------
    base_priors : dict
        Pre-roster prior probability for each outcome label (e.g. bookmaker
        consensus). Must sum to ~1.0.
    positions : dict
        2-D position for each outcome label, same keys as ``base_priors``.
    roster : Roster, optional
        Player list. If ``None``, this degenerates to a plain ``EventSpace``.
    head_to_head : tuple, optional
        ``(side_a_label, side_b_label)`` — which two well labels are the
        team wins (the draw label, if any, is left alone).
    roster_share : float
        Scale on the roster effect. ``0.0`` = roster ignored, ``1.0`` = full
        roster effect.

    Returns
    -------
    EventSpace
        Renormalised so masses sum to 1.0.
    """
    if roster is None or head_to_head is None:
        attractors = [
            Attractor(label, np.asarray(pos, dtype=float), base_priors[label])
            for label, pos in positions.items()
        ]
        return EventSpace(attractors)

    side_a, side_b = head_to_head
    m_a = roster.well_mass_multiplier(side_a, side_b, share=roster_share)
    m_b = roster.well_mass_multiplier(side_b, side_a, share=roster_share)
    scales = {side_a: m_a, side_b: m_b}

    attractors = []
    for label, pos in positions.items():
        scale = scales.get(label, 1.0)   # draw / non-team wells: scale=1
        attractors.append(
            Attractor(label, np.asarray(pos, dtype=float),
                      base_priors[label] * scale)
        )
    return EventSpace(attractors)

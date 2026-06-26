"""Core types: attractors, event spaces, and orbiting bodies."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

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

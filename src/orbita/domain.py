"""Core types: attractors, event spaces, and orbiting bodies."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from .forces import G, SOFTENING, gravity_force


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
        if self.position.ndim != 1 or self.position.shape[0] < 2:
            raise ValueError(
                f"position must be shape (n,) with n>=2, got "
                f"{self.position.shape}"
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

    # ---- saddle / confidence (issue #6) --------------------------------

    def _force_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Jacobian of the gravity force at ``q``.

        J = ∂F/∂q where F = G·Σ mₐ·(qₐ - q)/rₐ³. Used by Newton iteration
        to locate critical points of the potential.
        """
        n = q.shape[0]
        J = np.zeros((n, n))
        s2 = SOFTENING ** 2
        for a in self.attractors:
            delta = a.position - q
            r2 = float(delta @ delta) + s2
            r3 = r2 ** 1.5
            r5 = r2 ** 2.5
            J += G * a.mass * (
                -np.eye(n) / r3 + 3 * np.outer(delta, delta) / r5
            )
        return J

    def saddle_points(
        self, tol: float = 1e-8, max_iter: int = 50
    ) -> List[np.ndarray]:
        """Locate the saddle points of the gravitational potential.

        Starts a Newton iteration from the midpoint of every attractor
        pair. Critical points where the force Jacobian has negative
        determinant are saddles (mixed-sign Hessian of U → unstable
        equilibrium between basins).

        Returns deduplicated saddle positions.
        """
        saddles: List[np.ndarray] = []
        attrs = self.attractors
        for i in range(len(attrs)):
            for j in range(i + 1, len(attrs)):
                q = 0.5 * (attrs[i].position + attrs[j].position)
                converged = False
                for _ in range(max_iter):
                    F = gravity_force(q, attrs)
                    if float(F @ F) < tol ** 2:
                        converged = True
                        break
                    J = self._force_jacobian(q)
                    detJ = np.linalg.det(J)
                    if abs(detJ) < 1e-14:
                        break
                    q = q - np.linalg.solve(J, F)
                if not converged:
                    continue
                J_final = self._force_jacobian(q)
                # Hessian of U is -J; det(-J) = det(J) for 2x2. Saddle of
                # U ⇔ det(Hessian) < 0 ⇔ det(J) < 0.
                if np.linalg.det(J_final) >= 0:
                    continue
                # Dedupe against existing saddles.
                if any(np.linalg.norm(q - s) < 1e-4 for s in saddles):
                    continue
                saddles.append(q)
        return saddles

    def confidence(self, q: np.ndarray) -> float:
        """Confidence in the well classification of position ``q``.

        Returns a value in [0, 1]: high when ``q`` sits deep inside a
        single basin, low when ``q`` sits near a saddle between basins
        (where small perturbations would flip the outcome).

        Defined as ``1 - exp(-d_saddle / d_well)`` where ``d_saddle`` is
        the distance to the nearest saddle and ``d_well`` is the distance
        to the nearest attractor.
        """
        q = np.asarray(q, dtype=float)
        saddles = self.saddle_points()
        if not saddles:
            return 1.0
        d_well = min(float(np.linalg.norm(q - a.position))
                     for a in self.attractors)
        if d_well < 1e-9:
            return 1.0
        d_saddle = min(float(np.linalg.norm(q - s)) for s in saddles)
        return float(1.0 - np.exp(-d_saddle / d_well))


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
        if self.q0.shape != self.p0.shape or self.q0.ndim != 1 \
                or self.q0.shape[0] < 2:
            raise ValueError(
                f"q0 and p0 must be matching 1D arrays of length >= 2, "
                f"got q0.shape={self.q0.shape}, p0.shape={self.p0.shape}"
            )


# ---------------------------------------------------------------------------
# Roster layer (Phase-2 minimum viable — see GitHub issue #1)
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """An athlete or competitor whose presence reshapes a well's mass.

    Reserve ``team`` for the well label this player contributes to (e.g.
    ``\"france_win\"``). Sports without teams (tennis, boxing) use the player's
    own win-label as the team.

    ``position`` is a free-form string (``"GK"``, ``"DEF"``, ``"MID"``,
    ``"FWD"`` for soccer; ``"PG"``/``"SG"``/... for NBA; etc.). It only
    matters when :meth:`Roster.position_weighted_strength` is called.

    ``recent_form`` is the list of recent match grades (newest last) on
    the same 0–100 scale as ``rating``. Used by
    :meth:`Roster.with_form_decay` to recompute ratings with an exponential
    half-life weighting — useful when the season-aggregate rating doesn't
    reflect a player's current trajectory.
    """

    name: str
    team: str
    rating: float          # 0-100; aggregate skill/form going into the event
    available: bool = True
    position: str = ""
    recent_form: List[float] = field(default_factory=list)


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

    # ---- richer roster signal (issue #8) -------------------------------

    def position_weighted_strength(
        self, team_label: str, weights: Dict[str, float]
    ) -> float:
        """Strength of one side, weighted by player position.

        ``weights`` maps a position string to its relative importance for
        the market under consideration (e.g. ``{"FWD": 2.0, "MID": 1.0,
        "DEF": 0.8, "GK": 0.5}`` for an over-2.5 goals market). Missing
        positions get weight 1.0. Returns a value in [0, 1].

        Falls back to :meth:`strength` if no player on the side has a
        non-empty ``position`` field — keeps backward compatibility with
        the v0.2 flat-rating roster.
        """
        active = [p for p in self.players
                  if p.team == team_label and p.available]
        if not active:
            return 0.0
        if not any(p.position for p in active):
            return self.strength(team_label)
        num = 0.0
        den = 0.0
        for p in active:
            w = weights.get(p.position, 1.0)
            num += w * p.rating
            den += w
        if den == 0:
            return 0.0
        return float(num / den) / 100.0

    def with_form_decay(self, half_life: int = 5) -> "Roster":
        """Return a new ``Roster`` whose ratings are recomputed from
        ``recent_form`` with exponential half-life weighting.

        For a player with form list ``[f_0, f_1, ..., f_{n-1}]`` (oldest
        first), the form-adjusted rating is

            sum_i f_i · 2^{-(n-1-i)/half_life} / sum_i 2^{-(n-1-i)/half_life}

        so the newest game contributes weight 1, a game ``half_life``
        matches old contributes weight 0.5, and so on. Players with no
        recent_form data keep their existing rating.
        """
        if half_life <= 0:
            raise ValueError(f"half_life must be > 0, got {half_life}")
        new_players: List[Player] = []
        for p in self.players:
            if not p.recent_form:
                new_players.append(Player(
                    name=p.name, team=p.team, rating=p.rating,
                    available=p.available, position=p.position,
                    recent_form=list(p.recent_form),
                ))
                continue
            n = len(p.recent_form)
            weights = [2 ** (-(n - 1 - i) / half_life) for i in range(n)]
            wsum = sum(weights)
            decayed = sum(f * w for f, w in zip(p.recent_form, weights)) / wsum
            new_players.append(Player(
                name=p.name, team=p.team, rating=float(decayed),
                available=p.available, position=p.position,
                recent_form=list(p.recent_form),
            ))
        return Roster(players=new_players)

    def well_mass_multiplier(
        self, target: str, opponent: str, share: float = 1.0,
        position_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """How much to scale the ``target`` win-well's mass.

        Uses normalised relative strength so the multiplier hovers around
        1.0 when both sides are equally stocked, climbs above 1 when
        ``target`` is the stronger side, and falls below 1 when weaker.
        ``share`` lets the caller down-weight the roster effect (e.g.
        0.3 = "roster only counts for 30% of well mass adjustment").
        ``position_weights``, if provided, switches the strength
        calculation to :meth:`position_weighted_strength`.
        """
        if position_weights is None:
            s_t = self.strength(target)
            s_o = self.strength(opponent)
        else:
            s_t = self.position_weighted_strength(target, position_weights)
            s_o = self.position_weighted_strength(opponent, position_weights)
        if s_t + s_o == 0:
            return 1.0
        ratio = 2 * s_t / (s_t + s_o)   # in [0, 2]
        return 1.0 + share * (ratio - 1.0)


def sensors_from_lineup(roster: "Roster") -> List["Sensor"]:
    """Build a default set of in-play sensors keyed to a roster's players.

    For every available player, emits three sensors targeting that
    player's team well:

        - ``"<name>_goal"`` — multiplier 1.6, the player scored
        - ``"<name>_red_card"`` — multiplier 0.4, the player was sent off
        - ``"<name>_subbed_off"`` — multiplier scaled by player rating,
          a small dip proportional to how much the side relies on them

    These connect the roster layer (#8) to the v0.3 sensor layer (#2):
    once a real provider supplies the lineup, in-play events tied to
    specific players can update the well field directly.
    """
    sensors: List["Sensor"] = []
    for p in roster.players:
        if not p.available:
            continue
        sensors.append(Sensor(
            name=f"{p.name}_goal",
            target=p.team,
            likelihood=lambda v, _r=p.rating: 1.6,
        ))
        sensors.append(Sensor(
            name=f"{p.name}_red_card",
            target=p.team,
            likelihood=lambda v: 0.4,
        ))
        # Heavier reliance → bigger dip when subbed off. Tops out at ~30% dip.
        dip = 1.0 - 0.30 * (p.rating / 100.0)
        sensors.append(Sensor(
            name=f"{p.name}_subbed_off",
            target=p.team,
            likelihood=lambda v, _d=dip: _d,
        ))
    return sensors


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

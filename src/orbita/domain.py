"""Core types: attractors, event spaces, and orbiting bodies."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List

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

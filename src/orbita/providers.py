"""Player-rating providers (GitHub issue #8).

Provides a pluggable interface for sourcing real per-player ratings into
a :class:`~orbita.domain.Roster`. Ships a single concrete adapter,
:class:`SnapshotProvider`, that reads a TOML snapshot file. Live
adapters (FBref scraper, Understat JSON, Opta/StatsBomb subscriptions)
implement the same interface — to be filed as follow-up issues.

Snapshot TOML schema::

    # team aggregates (optional — used as a fallback when no player rows
    # match the requested team)
    [team_strength]
    spain = 88.0
    uruguay = 78.0

    # one block per player
    [[player]]
    name = "L. Yamal"
    team = "spain"           # well label, not display name
    rating = 89.5            # 0-100
    position = "FWD"         # GK/DEF/MID/FWD, or sport-specific
    available = true
    recent_form = [88.0, 90.0, 91.5, 87.0, 92.0]   # optional

A real provider would also expose ``as_of_date`` so historical backtests
remain reproducible.
"""
from __future__ import annotations

import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .domain import Player, Roster


class RatingProvider(ABC):
    """Abstract base for any source of per-player ratings.

    Implementations must return a :class:`Roster` containing the union of
    players on the two named sides, with at minimum a ``rating`` field
    populated. ``position``, ``recent_form`` and ``available`` are
    optional but increase the signal available to downstream code.
    """

    @abstractmethod
    def fetch(
        self,
        team_a: str,
        team_b: str,
        as_of_date: Optional[str] = None,
    ) -> Roster:
        """Return a Roster covering ``team_a`` and ``team_b``."""


@dataclass
class SnapshotProvider(RatingProvider):
    """Reads ratings from a TOML snapshot file.

    No network access; the snapshot is a point-in-time record. Useful
    for reproducible backtests and for the v0.3.2 demo before a live
    adapter is wired up.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Snapshot not found: {self.path}")
        with self.path.open("rb") as fh:
            self._data = tomllib.load(fh)

    @property
    def team_strength(self) -> Dict[str, float]:
        """Optional per-team aggregate ratings carried in the snapshot."""
        return dict(self._data.get("team_strength", {}))

    def fetch(
        self,
        team_a: str,
        team_b: str,
        as_of_date: Optional[str] = None,
    ) -> Roster:
        rows = self._data.get("player", [])
        players: List[Player] = []
        for row in rows:
            team = row.get("team")
            if team not in (team_a, team_b):
                continue
            players.append(Player(
                name=str(row["name"]),
                team=str(team),
                rating=float(row["rating"]),
                available=bool(row.get("available", True)),
                position=str(row.get("position", "")),
                recent_form=[float(x) for x in row.get("recent_form", [])],
            ))
        # Fallback: if no rows match, build a single aggregate "player"
        # per side from team_strength. Keeps backwards compat with the
        # v0.2 flat-rating roster when only aggregates are available.
        if not players:
            strength = self.team_strength
            for team in (team_a, team_b):
                if team in strength:
                    players.append(Player(
                        name=f"{team}-agg",
                        team=team,
                        rating=float(strength[team]),
                    ))
        return Roster(players=players)

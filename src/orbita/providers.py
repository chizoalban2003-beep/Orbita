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

import json
import re
import tomllib
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Understat live adapter (GitHub issue #9)
# ---------------------------------------------------------------------------

UNDERSTAT_MATCH_URL = "https://understat.com/match/{match_id}"

# match_info is the only JSON.parse block on a modern Understat match
# page (shots/rosters lazy-load via XHR). The regex must accept hex
# escapes inside the single-quoted JSON literal.
_MATCH_INFO_RE = re.compile(
    r"match_info\s*=\s*JSON\.parse\('([^']+)'\)",
    re.MULTILINE,
)


@dataclass
class UnderstatMatch:
    """One Understat match record, normalised for the backtest harness.

    Mirrors the shape of ``[[match]]`` entries in
    ``data/backtest_matches.toml`` so a list of these can be dropped
    straight into the existing v0.3.3 backtest loop.

    ``h_xg`` / ``a_xg`` / shot counts are carried separately for
    downstream use (sensor-layer Observations, position-weighted roster
    fits) but are not used by the W/D/L Brier scoring.
    """

    match_id: str
    sport: str
    date: str
    event: str
    side_a_label: str
    side_b_label: str
    draw_label: str
    prior_a: float
    prior_b: float
    prior_draw: float
    actual: str
    h_xg: float
    a_xg: float
    h_shot: int
    a_shot: int
    h_goals: int
    a_goals: int

    def as_backtest_dict(self) -> Dict[str, Any]:
        """Render as a dict matching the backtest_matches.toml schema."""
        return {
            "sport": self.sport,
            "event": self.event,
            "date": self.date,
            "side_a_label": self.side_a_label,
            "side_b_label": self.side_b_label,
            "draw_label": self.draw_label,
            "prior_a": self.prior_a,
            "prior_b": self.prior_b,
            "prior_draw": self.prior_draw,
            "actual": self.actual,
        }


@dataclass
class UnderstatMatchProvider:
    """Fetches per-match data from understat.com.

    Understat's match pages embed a ``match_info`` JSON literal with
    rich xG, win-probability, and outcome data — everything needed to
    drive the v0.3.3 backtest for soccer fixtures from EPL, La Liga,
    Bundesliga, Serie A, Ligue 1, and the RFPL going back to 2014/15.

    Network access is via stdlib ``urllib`` (no external dependency).
    Responses are cached on disk under ``cache_dir`` (default
    ``~/.cache/orbita/understat``) so the same match ID is fetched
    only once. Tests inject an ``html_loader`` callable instead.

    Parameters
    ----------
    cache_dir : Path, optional
        Where to cache fetched HTML. If ``None``, a default under
        ``~/.cache/orbita/understat`` is created on first use.
    html_loader : callable, optional
        Override for the HTTP layer. ``html_loader(match_id) -> str``
        returns the raw page HTML. Tests pass a stub that reads a
        fixture from disk; production leaves this ``None`` so the
        provider fetches via urllib.
    user_agent : str
        Sent as the ``User-Agent`` header on real fetches. Understat
        rejects empty UAs.
    """

    cache_dir: Optional[Path] = None
    html_loader: Optional[Callable[[str], str]] = None
    user_agent: str = "orbita/0.3.3 (+https://github.com/chizoalban2003-beep/Orbita)"

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "orbita" / "understat"

    # ---- I/O ----------------------------------------------------------

    def _fetch_html(self, match_id: str) -> str:
        """Return the raw HTML for a match page.

        Resolution order: in-memory ``html_loader`` override → disk
        cache → live HTTP fetch (and write through to cache).
        """
        if self.html_loader is not None:
            return self.html_loader(match_id)

        assert self.cache_dir is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"match_{match_id}.html"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        url = UNDERSTAT_MATCH_URL.format(match_id=match_id)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Understat fetch failed for {match_id}: {e}") from e
        cache_path.write_text(html, encoding="utf-8")
        return html

    # ---- parsing ------------------------------------------------------

    @staticmethod
    def _parse_match_info(html: str) -> Dict[str, Any]:
        m = _MATCH_INFO_RE.search(html)
        if not m:
            raise ValueError(
                "match_info JSON not found in HTML — Understat schema "
                "may have changed."
            )
        # Understat hex-escapes the JSON before embedding it in a single
        # quoted JS literal. Use unicode_escape to decode the \xNN
        # sequences back into normal characters.
        raw = m.group(1)
        decoded = raw.encode("latin-1").decode("unicode_escape")
        return json.loads(decoded)

    # ---- public API ---------------------------------------------------

    def fetch_match(self, match_id: str) -> UnderstatMatch:
        """Return the parsed :class:`UnderstatMatch` for ``match_id``.

        Determines the actual outcome from ``h_goals`` vs ``a_goals``
        and normalises labels so the result is a drop-in for the
        existing backtest harness.
        """
        html = self._fetch_html(str(match_id))
        info = self._parse_match_info(html)

        h_goals = int(info["h_goals"])
        a_goals = int(info["a_goals"])
        team_h = str(info["team_h"])
        team_a = str(info["team_a"])

        side_a_label = f"{_slug(team_h)}_win"
        side_b_label = f"{_slug(team_a)}_win"
        draw_label = "draw"

        if h_goals > a_goals:
            actual = side_a_label
        elif h_goals < a_goals:
            actual = side_b_label
        else:
            actual = draw_label

        return UnderstatMatch(
            match_id=str(info["id"]),
            sport="soccer",
            date=str(info["date"])[:10],
            event=f"{info.get('league','?')} {team_h} vs {team_a}",
            side_a_label=side_a_label,
            side_b_label=side_b_label,
            draw_label=draw_label,
            prior_a=float(info["h_w"]),
            prior_b=float(info["h_l"]),
            prior_draw=float(info["h_d"]),
            actual=actual,
            h_xg=float(info["h_xg"]),
            a_xg=float(info["a_xg"]),
            h_shot=int(info["h_shot"]),
            a_shot=int(info["a_shot"]),
            h_goals=h_goals,
            a_goals=a_goals,
        )


def _slug(team_name: str) -> str:
    """Normalise a team display name into a well-label slug."""
    s = team_name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

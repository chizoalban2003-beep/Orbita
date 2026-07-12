"""The Odds API ingestion — the mechanical seam that replaces the manual paste.

Trustworthy Information-Alpha needs a trustworthy *closing line*, and the box
cannot scrape one: the odds-history sites (oddsportal, betexplorer) render their
grids via JavaScript that fetch tools can't execute. A keyed JSON feed is the
only reliable path. This module pulls 1X2 odds from The Odds API
(https://the-odds-api.com), which aggregates ~40 books **including Pinnacle** —
so we keep Pinnacle as the sharp reference and just change the *access method*
from human eyeballs to an API call.

Scope, kept honest:
  * This is Layer-A automation (mechanics), not alpha. It removes friction; it
    does not beat the market.
  * The **free tier serves current/upcoming odds only** — it cannot retrieve the
    close of an already-played match. So this pipeline is for snapshotting the
    T-15 line of a *future* read, not for recovering a past settlement.
  * Needs a key: set ``ORBITA_ODDS_API_KEY`` in the environment. Without it every
    call raises cleanly; nothing is faked.

Devig: ships **Shin's method** (insider-trading margin model) as the more
accurate baseline than proportional normalisation — it accounts for the sharp
money embedded in the margin rather than smearing the overround uniformly. The
ledger still devigs proportionally for now; adopting Shin means switching *both*
the open and close devig coherently (a deliberate migration), not a one-sided
patch that would make Iα internally inconsistent.

Stdlib-only (urllib + json) to preserve the numpy-only runtime ethos.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

API_BASE = "https://api.the-odds-api.com/v4"
_LAB = ("home", "draw", "away")

# common Odds API sport keys (extend as needed)
SPORT_KEYS = {
    "eliteserien": "soccer_norway_eliteserien",
    "epl": "soccer_epl",
    "allsvenskan": "soccer_sweden_allsvenskan",
    "mls": "soccer_usa_mls",
}


class OddsAPIError(RuntimeError):
    """Any failure sourcing odds — missing key, network, quota, or no match."""


# --------------------------------------------------------------------------
# devig — Shin's method (the accurate baseline)
# --------------------------------------------------------------------------
def shin_devig(odds: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """Recover true probabilities from 1X2 decimal odds via Shin's model.

    Shin assumes a fraction ``z`` of money is informed; the bookmaker widens
    margins to protect against it, and that widening is *not* uniform — it falls
    harder on the outcomes insiders back. Recovering ``p_i`` therefore corrects
    the favourite–longshot distortion that proportional normalisation leaves in.

    For booked probabilities ``π_i = 1/o_i`` with overround ``B = Σπ_i``::

        p_i(z) = [ sqrt(z² + 4(1−z)·π_i²/B) − z ] / [ 2(1−z) ]

    ``Σ p_i(z)`` decreases monotonically in ``z``; we bisect for the ``z`` that
    makes it 1. Returns ``(probs, z)`` where ``z`` is the fitted insider share.
    """
    o = np.array([float(odds[k]) for k in _LAB])
    if np.any(o <= 1.0) or not np.all(np.isfinite(o)):
        raise OddsAPIError(f"invalid odds (must be decimal > 1.0): {odds}")
    pi = 1.0 / o
    B = float(pi.sum())

    def p_of(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / B) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.9                       # z ∈ [0,1); 0.9 brackets any real margin
    if p_of(hi).sum() > 1.0:                # pathological overround — clamp
        z = hi
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if p_of(mid).sum() > 1.0:
                lo = mid
            else:
                hi = mid
        z = 0.5 * (lo + hi)
    p = p_of(z)
    p = p / p.sum()                         # tidy any residual rounding
    return {k: float(p[i]) for i, k in enumerate(_LAB)}, float(z)


def proportional_devig(odds: Dict[str, float]) -> Dict[str, float]:
    """Naive normalisation (what the ledger currently uses) — for comparison."""
    inv = {k: 1.0 / float(odds[k]) for k in _LAB}
    s = sum(inv.values())
    return {k: inv[k] / s for k in _LAB}


# --------------------------------------------------------------------------
# fetch + parse
# --------------------------------------------------------------------------
def _resolve_sport(sport: str) -> str:
    return SPORT_KEYS.get(sport.lower(), sport)


def fetch_events(sport: str, *, regions: str = "eu", markets: str = "h2h",
                 api_key: Optional[str] = None, base: str = API_BASE,
                 timeout: float = 20.0) -> List[Dict]:
    """GET the current/upcoming events + odds for a sport. Raises OddsAPIError
    with a clear cause on missing key, network failure, or quota (HTTP 401/402)."""
    key = api_key or os.environ.get("ORBITA_ODDS_API_KEY")
    if not key:
        raise OddsAPIError("no API key — set ORBITA_ODDS_API_KEY (the-odds-api.com)")
    sk = _resolve_sport(sport)
    url = (f"{base}/sports/{sk}/odds/?apiKey={key}&regions={regions}"
           f"&markets={markets}&oddsFormat=decimal")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        hint = {401: "bad API key", 402: "quota exhausted",
                422: "bad sport/region"}.get(e.code, f"HTTP {e.code}")
        raise OddsAPIError(f"Odds API request failed ({hint})") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OddsAPIError(f"network error reaching Odds API: {e}") from e


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _team_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    return na in nb or nb in na


def extract_1x2(event: Dict, home: str, away: str, *,
                book: Optional[str] = "pinnacle") -> Tuple[Dict[str, float], str]:
    """Pull home/draw/away decimal odds from one event's h2h market.

    ``book`` selects a bookmaker key (default Pinnacle, the sharp reference);
    pass ``None`` to average the implied probabilities across all books present.
    Returns ``(odds, book_label)``. Raises if the market or book is absent."""
    books = event.get("bookmakers", [])
    if not books:
        raise OddsAPIError("event has no bookmakers")

    def h2h_of(bk):
        for m in bk.get("markets", []):
            if m.get("key") == "h2h":
                return m.get("outcomes", [])
        return None

    def to_hda(outcomes):
        d: Dict[str, float] = {}
        for o in outcomes:
            nm, price = o.get("name", ""), o.get("price")
            if price is None:
                continue
            if _norm(nm) == _norm("Draw"):
                d["draw"] = float(price)
            elif _team_match(nm, event.get("home_team", home)) or _team_match(nm, home):
                d["home"] = float(price)
            elif _team_match(nm, event.get("away_team", away)) or _team_match(nm, away):
                d["away"] = float(price)
        return d if all(k in d for k in _LAB) else None

    if book is None:                        # consensus: average implied probs
        acc, n = {k: 0.0 for k in _LAB}, 0
        for bk in books:
            oc = h2h_of(bk)
            hda = to_hda(oc) if oc else None
            if not hda:
                continue
            for k in _LAB:
                acc[k] += 1.0 / hda[k]
            n += 1
        if not n:
            raise OddsAPIError("no book had a complete h2h market")
        return ({k: n / acc[k] for k in _LAB}, f"consensus/{n} books")

    for bk in books:
        if bk.get("key") == book:
            oc = h2h_of(bk)
            hda = to_hda(oc) if oc else None
            if hda:
                return hda, bk.get("title", book)
            raise OddsAPIError(f"{book}: no complete h2h market")
    avail = ", ".join(sorted(bk.get("key", "?") for bk in books))
    raise OddsAPIError(f"book {book!r} not present (have: {avail})")


def closing_line(sport: str, home: str, away: str, *, book: Optional[str] = "pinnacle",
                 api_key: Optional[str] = None) -> Dict[str, float]:
    """Fetch the current 1X2 line for a fixture, ready to hand to
    ``ledger.settle(close_odds=...)``. Call it ~T-15 before kickoff to capture
    the near-closing sharp price. Matches the fixture by team-name containment."""
    events = fetch_events(sport, api_key=api_key)
    for ev in events:
        if (_team_match(ev.get("home_team", ""), home)
                and _team_match(ev.get("away_team", ""), away)):
            odds, _ = extract_1x2(ev, home, away, book=book)
            return odds
    raise OddsAPIError(f"fixture {home} v {away} not found in current {sport} slate "
                       "(the free tier serves only upcoming games)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Fetch a 1X2 line from The Odds API")
    p.add_argument("--sport", default="eliteserien")
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--book", default="pinnacle", help="bookmaker key, or 'consensus'")
    args = p.parse_args(argv)
    book = None if args.book == "consensus" else args.book
    odds = closing_line(args.sport, args.home, args.away, book=book)
    shin, z = shin_devig(odds)
    prop = proportional_devig(odds)
    print(f"odds  : {odds}")
    print(f"shin  : {{{', '.join(f'{k} {shin[k]:.3f}' for k in _LAB)}}}  (insider z={z:.3f})")
    print(f"prop  : {{{', '.join(f'{k} {prop[k]:.3f}' for k in _LAB)}}}")


if __name__ == "__main__":
    try:
        _main()
    except OddsAPIError as e:
        raise SystemExit(f"odds_api: {e}")

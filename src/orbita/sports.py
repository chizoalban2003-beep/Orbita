"""Sport-specific event-space templates (GitHub issue #4).

Each template returns the well geometry and integrator parameters for a
sport. The per-sport knobs are defensible from sport mechanics in one
sentence (see TEMPLATES below) — they are NOT back-fit to outcomes.

The unified v0.1 layout (well distance 5, draw at y=4, C_d=0.04,
duration=2700) was implicitly tuned for soccer. The templates here adapt
the metaphor to each sport's actual dynamics:

    - Tennis : service holds compound, no draw, low drag, narrow wells
    - NBA    : possessions are noisy, final score regresses to the prior
    - Soccer : real draws, 90-min chaos, mid drag, draw well off-axis
    - MMA    : one decisive blow ends the fight — sharp wells, low drag
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .domain import EventSpace, Roster, event_space_from_rosters


# Per-sport templates. The justification field is part of the API —
# every parameter must be defendable from sport mechanics.
TEMPLATES: Dict[str, Dict] = {
    "tennis": {
        "positions": {"a": [6.0, 0.0], "b": [-6.0, 0.0]},
        "C_d": 0.02,
        "duration": 400.0,
        "has_draw": False,
        "justification": (
            "Service holds compound; momentum doesn't bleed → low drag. "
            "Best-of-5 sets are decisive → moderate duration. "
            "No tied final score → 2 wells, no draw."
        ),
    },
    "nba": {
        "positions": {"a": [4.0, 0.0], "b": [-4.0, 0.0]},
        "C_d": 0.08,
        "duration": 800.0,
        "has_draw": False,
        "justification": (
            "Every possession is noise; final score regresses to the prior "
            "→ high drag. 100+ possessions per game → long duration. "
            "Overtime breaks ties → no draw."
        ),
    },
    "soccer": {
        "positions": {"a": [5.0, 0.0], "b": [-5.0, 0.0], "draw": [0.0, 5.0]},
        "C_d": 0.04,
        "duration": 600.0,
        "has_draw": True,
        "justification": (
            "90 minutes of chaos but low scoring → mid drag. "
            "Three terminal outcomes → draw well at y=5 (deeper than v0.1's "
            "y=4, so 0-0 / 1-1 acts as a stable equilibrium not a midpoint)."
        ),
    },
    "mma": {
        "positions": {"a": [7.0, 0.0], "b": [-7.0, 0.0]},
        "C_d": 0.015,
        "duration": 300.0,
        "has_draw": False,
        "justification": (
            "A single decisive blow ends the fight → sharp narrow wells "
            "(far apart) + low drag. 3-5 rounds, often ends early → short "
            "duration. Draws are statistically negligible → 2 wells."
        ),
    },
}


def template_for(sport: str) -> Dict:
    """Look up the template dict for a sport. Raises if unknown."""
    if sport not in TEMPLATES:
        raise ValueError(
            f"No template for sport={sport!r}; have {list(TEMPLATES)}"
        )
    return TEMPLATES[sport]


def build_space(
    sport: str,
    side_a_label: str,
    side_b_label: str,
    prior_a: float,
    prior_b: float,
    *,
    prior_draw: Optional[float] = None,
    draw_label: Optional[str] = None,
    roster: Optional[Roster] = None,
    roster_share: float = 1.0,
) -> Tuple[EventSpace, Dict]:
    """Construct the event-space for a single match using the sport template.

    Returns
    -------
    (space, sim_kwargs)
        ``sim_kwargs`` carries ``C_d`` and ``duration`` so the caller can
        pass them straight into :func:`orbita.simulate`.
    """
    t = template_for(sport)
    if t["has_draw"]:
        if prior_draw is None or draw_label is None:
            raise ValueError(
                f"sport={sport!r} has a draw well — pass prior_draw and "
                f"draw_label"
            )
        base_priors = {
            side_a_label: prior_a,
            draw_label:   prior_draw,
            side_b_label: prior_b,
        }
        positions = {
            side_a_label: t["positions"]["a"],
            draw_label:   t["positions"]["draw"],
            side_b_label: t["positions"]["b"],
        }
    else:
        base_priors = {side_a_label: prior_a, side_b_label: prior_b}
        positions = {
            side_a_label: t["positions"]["a"],
            side_b_label: t["positions"]["b"],
        }

    space = event_space_from_rosters(
        base_priors=base_priors,
        positions=positions,
        roster=roster,
        head_to_head=(side_a_label, side_b_label),
        roster_share=roster_share,
    )
    return space, {"C_d": t["C_d"], "duration": t["duration"]}

"""Tests for sport-specific event-space templates (GitHub issue #4)."""
from __future__ import annotations

import warnings

import pytest

from orbita import TEMPLATES, build_space, template_for


def test_all_templates_have_justification() -> None:
    """Every template parameter must be defensible from sport mechanics."""
    for sport, t in TEMPLATES.items():
        assert "justification" in t, f"{sport} missing justification"
        assert len(t["justification"]) > 40, \
            f"{sport}: justification too short to be meaningful"


def test_template_for_unknown_sport_raises() -> None:
    with pytest.raises(ValueError, match="No template"):
        template_for("quidditch")


def test_tennis_has_no_draw() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        space, kw = build_space(
            "tennis", "a_win", "b_win", prior_a=0.7, prior_b=0.3,
        )
    labels = [a.label for a in space.attractors]
    assert "draw" not in labels
    assert len(space.attractors) == 2


def test_soccer_requires_draw_prior() -> None:
    with pytest.raises(ValueError, match="draw"):
        build_space(
            "soccer", "a_win", "b_win", prior_a=0.6, prior_b=0.2,
        )


def test_soccer_builds_three_well_space() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        space, kw = build_space(
            "soccer", "a_win", "b_win",
            prior_a=0.6, prior_b=0.2,
            prior_draw=0.2, draw_label="draw",
        )
    labels = [a.label for a in space.attractors]
    assert set(labels) == {"a_win", "b_win", "draw"}


def test_templates_yield_distinct_drag_per_sport() -> None:
    """The whole point: per-sport templates should NOT collapse to the same
    drag value."""
    drags = {sport: t["C_d"] for sport, t in TEMPLATES.items()}
    assert len(set(drags.values())) >= 3, \
        f"Templates have collapsing drag values: {drags}"


def test_mma_wells_are_further_apart_than_nba() -> None:
    """Sport-mechanics check: MMA has 'sharp narrow wells far apart' (one
    decisive blow ends it), NBA has 'noisy possessions, mean reversion'
    → MMA well separation > NBA."""
    mma_a = abs(TEMPLATES["mma"]["positions"]["a"][0])
    nba_a = abs(TEMPLATES["nba"]["positions"]["a"][0])
    assert mma_a > nba_a

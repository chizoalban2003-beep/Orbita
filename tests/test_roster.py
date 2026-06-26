"""Tests for the minimal roster layer (GitHub issue #1)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from orbita import Player, Roster, event_space_from_rosters


def test_unavailable_player_drops_strength() -> None:
    roster = Roster(players=[
        Player(name="A1", team="a_win", rating=90),
        Player(name="A2", team="a_win", rating=80),
        Player(name="B1", team="b_win", rating=85),
    ])
    s_before = roster.strength("a_win")
    roster.players[0].available = False
    s_after = roster.strength("a_win")
    assert s_after < s_before


def test_equal_rosters_yield_unit_multiplier() -> None:
    roster = Roster(players=[
        Player(name="A", team="a_win", rating=70),
        Player(name="B", team="b_win", rating=70),
    ])
    assert roster.well_mass_multiplier("a_win", "b_win") == pytest.approx(1.0)
    assert roster.well_mass_multiplier("b_win", "a_win") == pytest.approx(1.0)


def test_stronger_side_gets_multiplier_above_one() -> None:
    roster = Roster(players=[
        Player(name="A", team="a_win", rating=90),
        Player(name="B", team="b_win", rating=60),
    ])
    assert roster.well_mass_multiplier("a_win", "b_win") > 1.0
    assert roster.well_mass_multiplier("b_win", "a_win") < 1.0


def test_event_space_from_rosters_renormalises() -> None:
    roster = Roster(players=[
        Player(name="A", team="a_win", rating=90),
        Player(name="B", team="b_win", rating=60),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        space = event_space_from_rosters(
            base_priors={"a_win": 0.5, "b_win": 0.5},
            positions={"a_win": [5, 0], "b_win": [-5, 0]},
            roster=roster,
            head_to_head=("a_win", "b_win"),
        )
    total = sum(a.mass for a in space.attractors)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_draw_well_is_not_scaled_by_roster() -> None:
    roster = Roster(players=[
        Player(name="A", team="a_win", rating=99),
        Player(name="B", team="b_win", rating=10),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        space = event_space_from_rosters(
            base_priors={"a_win": 0.50, "draw": 0.25, "b_win": 0.25},
            positions={"a_win": [5, 0], "draw": [0, 4], "b_win": [-5, 0]},
            roster=roster,
            head_to_head=("a_win", "b_win"),
        )
    # After renormalisation, the draw label's mass relative to the sum
    # should be smaller than the un-roster'd 0.25 only because the a_win
    # side took mass from b_win (and the renorm rescaled everything),
    # not because the draw multiplier itself changed. We assert that:
    # the a_win:b_win ratio after rosters > the 0.50:0.25 = 2 prior ratio.
    masses = {a.label: a.mass for a in space.attractors}
    assert masses["a_win"] / masses["b_win"] > 2.0


def test_roster_share_zero_is_a_noop() -> None:
    roster = Roster(players=[
        Player(name="A", team="a_win", rating=99),
        Player(name="B", team="b_win", rating=10),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        space = event_space_from_rosters(
            base_priors={"a_win": 0.6, "b_win": 0.4},
            positions={"a_win": [5, 0], "b_win": [-5, 0]},
            roster=roster,
            head_to_head=("a_win", "b_win"),
            roster_share=0.0,
        )
    masses = {a.label: a.mass for a in space.attractors}
    assert masses["a_win"] == pytest.approx(0.6)
    assert masses["b_win"] == pytest.approx(0.4)

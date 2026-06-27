"""Tests for the rich roster layer (issue #8)."""
from __future__ import annotations

from pathlib import Path

import pytest

from orbita import (
    Player,
    RatingProvider,
    Roster,
    SnapshotProvider,
    sensors_from_lineup,
)


SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_snapshot.toml"


# ---- Player / Roster extensions ------------------------------------------

def test_player_defaults_are_backwards_compatible() -> None:
    p = Player(name="x", team="a", rating=80.0)
    assert p.position == ""
    assert p.recent_form == []
    assert p.available is True


def test_position_weighted_strength_uses_weights() -> None:
    # Two players, equal ratings, different positions.
    roster = Roster(players=[
        Player(name="def", team="a", rating=80.0, position="DEF"),
        Player(name="fwd", team="a", rating=80.0, position="FWD"),
    ])
    # Uniform weights → mean.
    s_flat = roster.position_weighted_strength(
        "a", {"DEF": 1.0, "FWD": 1.0})
    assert s_flat == pytest.approx(0.80)
    # Weight forwards 3× — strength still 0.80 because ratings are equal.
    s_weighted = roster.position_weighted_strength(
        "a", {"DEF": 1.0, "FWD": 3.0})
    assert s_weighted == pytest.approx(0.80)


def test_position_weighted_strength_shifts_with_unequal_ratings() -> None:
    roster = Roster(players=[
        Player(name="def", team="a", rating=70.0, position="DEF"),
        Player(name="fwd", team="a", rating=90.0, position="FWD"),
    ])
    flat = roster.position_weighted_strength(
        "a", {"DEF": 1.0, "FWD": 1.0})
    fwd_heavy = roster.position_weighted_strength(
        "a", {"DEF": 1.0, "FWD": 3.0})
    assert flat == pytest.approx(0.80)
    assert fwd_heavy > flat   # heavier weight on the stronger position


def test_position_weighted_falls_back_to_flat_when_no_positions() -> None:
    # No positions set on any player → behave like Roster.strength.
    roster = Roster(players=[
        Player(name="x", team="a", rating=80.0),
        Player(name="y", team="a", rating=60.0),
    ])
    assert roster.position_weighted_strength(
        "a", {"FWD": 2.0}) == pytest.approx(roster.strength("a"))


def test_form_decay_recent_games_dominate() -> None:
    p = Player(name="trend", team="a", rating=70.0,
               recent_form=[60.0, 65.0, 70.0, 75.0, 90.0])
    roster = Roster(players=[p])
    decayed = roster.with_form_decay(half_life=2)
    new_rating = decayed.players[0].rating
    # Newest game weight 1, oldest weight 0.5^(4/2)=0.25, so the decayed
    # rating should sit above the unweighted mean of 72 and below the
    # newest game's 90.
    assert 72.0 < new_rating < 90.0
    # Original roster unchanged (immutable update).
    assert roster.players[0].rating == 70.0


def test_form_decay_no_form_keeps_rating() -> None:
    p = Player(name="static", team="a", rating=80.0, recent_form=[])
    roster = Roster(players=[p])
    decayed = roster.with_form_decay(half_life=5)
    assert decayed.players[0].rating == 80.0


def test_form_decay_invalid_half_life_raises() -> None:
    roster = Roster(players=[Player(name="x", team="a", rating=80.0)])
    with pytest.raises(ValueError):
        roster.with_form_decay(half_life=0)


def test_well_mass_multiplier_with_position_weights() -> None:
    roster = Roster(players=[
        Player(name="a_fwd", team="a", rating=90.0, position="FWD"),
        Player(name="a_def", team="a", rating=70.0, position="DEF"),
        Player(name="b_fwd", team="b", rating=70.0, position="FWD"),
        Player(name="b_def", team="b", rating=90.0, position="DEF"),
    ])
    # Flat: both sides equal → multiplier 1.0 each way.
    assert roster.well_mass_multiplier("a", "b") == pytest.approx(1.0)
    # Weight forwards heavily → side a (better forward) gets a > 1.0
    # multiplier, side b gets < 1.0.
    fwd_weights = {"FWD": 3.0, "DEF": 1.0}
    m_a = roster.well_mass_multiplier("a", "b",
                                      position_weights=fwd_weights)
    m_b = roster.well_mass_multiplier("b", "a",
                                      position_weights=fwd_weights)
    assert m_a > 1.0
    assert m_b < 1.0


# ---- Lineup-level sensors ------------------------------------------------

def test_sensors_from_lineup_emits_three_sensors_per_player() -> None:
    roster = Roster(players=[
        Player(name="alice", team="a", rating=80.0),
        Player(name="bob", team="b", rating=70.0, available=False),  # bench
    ])
    sensors = sensors_from_lineup(roster)
    names = {s.name for s in sensors}
    assert "alice_goal" in names
    assert "alice_red_card" in names
    assert "alice_subbed_off" in names
    # Bob is unavailable — no sensors for him.
    assert not any("bob" in n for n in names)


def test_lineup_sensor_targets_match_player_team() -> None:
    roster = Roster(players=[
        Player(name="alice", team="france_win", rating=85.0),
    ])
    sensors = sensors_from_lineup(roster)
    for s in sensors:
        assert s.target == "france_win"


def test_lineup_sensor_likelihoods_have_expected_signs() -> None:
    roster = Roster(players=[
        Player(name="x", team="a", rating=80.0),
    ])
    by_name = {s.name: s for s in sensors_from_lineup(roster)}
    assert by_name["x_goal"].likelihood(1.0) > 1.0
    assert by_name["x_red_card"].likelihood(1.0) < 1.0
    assert 0.0 < by_name["x_subbed_off"].likelihood(1.0) < 1.0


# ---- SnapshotProvider ----------------------------------------------------

def test_snapshot_provider_is_a_RatingProvider() -> None:
    assert issubclass(SnapshotProvider, RatingProvider)


def test_snapshot_provider_loads_players_for_named_teams() -> None:
    sp = SnapshotProvider(path=SAMPLE)
    roster = sp.fetch(team_a="spain", team_b="uruguay")
    teams = {p.team for p in roster.players}
    assert teams == {"spain", "uruguay"}
    assert any(p.position == "FWD" for p in roster.players)


def test_snapshot_provider_respects_availability_flag() -> None:
    sp = SnapshotProvider(path=SAMPLE)
    roster = sp.fetch(team_a="spain", team_b="uruguay")
    pellistri = next(p for p in roster.players if p.name == "Pellistri")
    assert pellistri.available is False


def test_snapshot_provider_falls_back_to_team_strength() -> None:
    sp = SnapshotProvider(path=SAMPLE)
    # Team that exists in [team_strength] but has no [[player]] rows.
    roster = sp.fetch(team_a="croatia", team_b="ghana")
    assert roster.players == []   # no rows AND no team_strength entries


def test_snapshot_provider_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        SnapshotProvider(path=Path("/nonexistent.toml"))


def test_provider_to_form_decay_pipeline() -> None:
    sp = SnapshotProvider(path=SAMPLE)
    roster = sp.fetch(team_a="spain", team_b="uruguay")
    decayed = roster.with_form_decay(half_life=3)
    # Yamal's recent_form trends up (85 → 92), so the decayed rating
    # should exceed his static rating.
    yamal_static = next(p.rating for p in roster.players if p.name == "Yamal")
    yamal_decayed = next(p.rating for p in decayed.players if p.name == "Yamal")
    assert yamal_decayed > yamal_static

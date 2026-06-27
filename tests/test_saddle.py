"""Tests for saddle-point detection and confidence (issue #6)."""
from __future__ import annotations

import numpy as np
import pytest

from orbita import Attractor, Body, EventSpace, simulate


def make_symmetric_2well() -> EventSpace:
    return EventSpace([
        Attractor("a", [5.0, 0.0], 0.5),
        Attractor("b", [-5.0, 0.0], 0.5),
    ])


def make_asymmetric_2well() -> EventSpace:
    # Heavier 'a' should pull the saddle toward 'b' (the lighter side).
    return EventSpace([
        Attractor("a", [5.0, 0.0], 0.8),
        Attractor("b", [-5.0, 0.0], 0.2),
    ])


def make_soccer_3well() -> EventSpace:
    # Match the soccer template geometry.
    return EventSpace([
        Attractor("a", [5.0, 0.0], 0.45),
        Attractor("draw", [0.0, 5.0], 0.30),
        Attractor("b", [-5.0, 0.0], 0.25),
    ])


def test_symmetric_2well_saddle_at_origin() -> None:
    space = make_symmetric_2well()
    saddles = space.saddle_points()
    assert len(saddles) == 1
    assert np.allclose(saddles[0], [0.0, 0.0], atol=1e-6)


def test_asymmetric_2well_saddle_shifts_toward_lighter() -> None:
    space = make_asymmetric_2well()
    saddles = space.saddle_points()
    assert len(saddles) == 1
    # Lighter well is at x=-5; saddle should land at negative x.
    assert saddles[0][0] < -0.1
    assert abs(saddles[0][1]) < 1e-6


def test_soccer_3well_has_two_saddles() -> None:
    # Morse theory: in the plane with one max at infinity, # saddles =
    # # wells - 1. So 3 wells → 2 saddles, not 3 (the issue text
    # asserted 3 incorrectly).
    space = make_soccer_3well()
    saddles = space.saddle_points()
    assert len(saddles) == 2
    # Both saddles should sit roughly between draw and one of the team
    # wells — i.e. positive y.
    assert all(s[1] > 0 for s in saddles)


def test_confidence_at_saddle_is_zero() -> None:
    space = make_symmetric_2well()
    s = space.saddle_points()[0]
    assert space.confidence(s) == pytest.approx(0.0, abs=1e-6)


def test_confidence_at_well_centre_is_one() -> None:
    space = make_symmetric_2well()
    assert space.confidence(np.array([5.0, 0.0])) == pytest.approx(1.0)


def test_confidence_grows_as_body_moves_into_well() -> None:
    space = make_symmetric_2well()
    points = [np.array([x, 0.0]) for x in (0.1, 1.0, 2.0, 3.0, 4.0)]
    confs = [space.confidence(p) for p in points]
    assert confs == sorted(confs)   # monotonically increasing toward the well


def test_confidence_symmetric_about_origin() -> None:
    space = make_symmetric_2well()
    assert space.confidence(np.array([1.5, 0.0])) == pytest.approx(
        space.confidence(np.array([-1.5, 0.0])), abs=1e-9
    )


def test_simulate_exposes_confidence_scalar() -> None:
    space = make_symmetric_2well()
    body = Body(mass=1.0, q0=[0.5, 0.0], p0=[0.2, 0.0])
    sol = simulate(space, body=body, duration=50.0, C_d=0.05, dt=0.05)
    assert "confidence" in sol
    assert 0.0 <= sol["confidence"] <= 1.0


def test_confidence_zero_when_no_saddles() -> None:
    # Degenerate 1-well case: no saddles → max confidence everywhere.
    space = EventSpace([Attractor("solo", [0.0, 0.0], 1.0)])
    assert space.confidence(np.array([3.0, 4.0])) == 1.0

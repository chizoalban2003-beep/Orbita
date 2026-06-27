"""Tests for the soft Plummer posterior (v0.3.3).

The soft posterior replaces hard nearest-well classification with a
mass-weighted assignment proportional to ``mass_k / r_k**alpha`` (with
Plummer softening). It's the v0.3.3 fix for the over-sharpening
diagnosed in ``experiments/01_sharpening_triage.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from orbita import (
    Attractor,
    Body,
    EventSpace,
    final_well,
    final_well_posterior,
    simulate,
)


def _two_well_space():
    return EventSpace([
        Attractor("a", [5.0, 0.0],  0.5),
        Attractor("b", [-5.0, 0.0], 0.5),
    ])


def _three_well_space():
    return EventSpace([
        Attractor("a",    [5.0, 0.0],  0.5),
        Attractor("b",    [-5.0, 0.0], 0.3),
        Attractor("draw", [0.0, 5.0],  0.2),
    ])


def _solution_with_final_q(space, q_end):
    """Build a minimal sol dict whose final position is ``q_end``."""
    return {
        "q": np.array([np.zeros(2), np.asarray(q_end, dtype=float)]),
        "p": np.zeros((2, 2)),
        "t": np.array([0.0, 1.0]),
    }


# ---- API shape -----------------------------------------------------------

def test_posterior_sums_to_one() -> None:
    space = _three_well_space()
    sol = _solution_with_final_q(space, [1.0, 2.0])
    posterior = final_well_posterior(sol, space, alpha=2.0)
    assert pytest.approx(sum(posterior.values()), abs=1e-9) == 1.0


def test_posterior_covers_every_well() -> None:
    space = _three_well_space()
    sol = _solution_with_final_q(space, [0.0, 0.0])
    posterior = final_well_posterior(sol, space, alpha=2.0)
    assert set(posterior) == {"a", "b", "draw"}


def test_posterior_all_probabilities_nonnegative() -> None:
    space = _three_well_space()
    sol = _solution_with_final_q(space, [3.0, -1.0])
    posterior = final_well_posterior(sol, space, alpha=2.0)
    assert all(p >= 0 for p in posterior.values())


# ---- Behaviour -----------------------------------------------------------

def test_posterior_at_well_centre_mass_goes_to_that_well() -> None:
    """At a well's exact centre, alpha=2 puts essentially all mass on it."""
    space = _two_well_space()
    sol = _solution_with_final_q(space, [5.0, 0.0])  # sitting on well a
    posterior = final_well_posterior(sol, space, alpha=2.0)
    assert posterior["a"] > 0.99


def test_posterior_at_midpoint_splits_by_prior() -> None:
    """Equidistant from both wells of equal mass → 50/50."""
    space = _two_well_space()
    sol = _solution_with_final_q(space, [0.0, 0.0])
    posterior = final_well_posterior(sol, space, alpha=2.0)
    assert posterior["a"] == pytest.approx(0.5, abs=1e-9)
    assert posterior["b"] == pytest.approx(0.5, abs=1e-9)


def test_posterior_unequal_masses_skew_midpoint() -> None:
    """Equidistant but unequal masses → heavier well takes more."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        space = EventSpace([
            Attractor("heavy", [5.0, 0.0],  0.8),
            Attractor("light", [-5.0, 0.0], 0.2),
        ])
    sol = _solution_with_final_q(space, [0.0, 0.0])
    posterior = final_well_posterior(sol, space, alpha=2.0)
    assert posterior["heavy"] > posterior["light"]
    # Equidistant + alpha=anything → ratio == mass ratio.
    assert posterior["heavy"] / posterior["light"] == pytest.approx(4.0)


def test_higher_alpha_sharper_assignment() -> None:
    """Larger alpha pulls more mass toward the nearest well."""
    space = _two_well_space()
    sol = _solution_with_final_q(space, [3.0, 0.0])  # closer to a
    low = final_well_posterior(sol, space, alpha=1.0)
    high = final_well_posterior(sol, space, alpha=4.0)
    assert high["a"] > low["a"]


def test_posterior_agrees_with_hard_in_limit() -> None:
    """As alpha → large, the soft posterior should pick the same modal
    well as ``final_well``."""
    space = _three_well_space()
    sol = _solution_with_final_q(space, [4.5, 0.5])
    soft = final_well_posterior(sol, space, alpha=10.0)
    hard = final_well(sol, space)
    assert max(soft, key=soft.get) == hard


# ---- Pipeline with the integrator ----------------------------------------

def test_posterior_runs_after_simulate() -> None:
    """End-to-end: simulate → soft posterior. Returns a valid distribution
    over the wells with no NaNs."""
    space = _two_well_space()
    body = Body(mass=1.0, q0=[0.0, 0.5], p0=[0.2, 0.1])
    sol = simulate(space, body=body, duration=200.0, C_d=0.02, dt=0.1)
    posterior = final_well_posterior(sol, space, alpha=2.0)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert all(not np.isnan(p) for p in posterior.values())

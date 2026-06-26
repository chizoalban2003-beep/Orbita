"""Tests for the alpha-blend sharpening calibration (GitHub issue #3)."""
from __future__ import annotations

import numpy as np
import pytest

from orbita import aggregate_brier, blend, brier, fit_alpha, loocv_alpha


def test_blend_alpha_zero_recovers_prior() -> None:
    prior = {"a": 0.6, "b": 0.4}
    engine = {"a": 0.9, "b": 0.1}
    out = blend(prior, engine, alpha=0.0)
    assert out["a"] == pytest.approx(0.6)
    assert out["b"] == pytest.approx(0.4)


def test_blend_alpha_one_recovers_engine() -> None:
    prior = {"a": 0.6, "b": 0.4}
    engine = {"a": 0.9, "b": 0.1}
    out = blend(prior, engine, alpha=1.0)
    assert out["a"] == pytest.approx(0.9)
    assert out["b"] == pytest.approx(0.1)


def test_blend_sums_to_one() -> None:
    prior = {"a": 0.50, "draw": 0.30, "b": 0.20}
    engine = {"a": 0.10, "draw": 0.30, "b": 0.60}
    for alpha in (0.0, 0.1, 0.5, 0.9, 1.0):
        out = blend(prior, engine, alpha)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_fit_alpha_picks_zero_when_engine_is_random() -> None:
    """If the engine output is uncorrelated noise, alpha should collapse
    toward zero (trust the prior)."""
    prior = {"a": 0.7, "b": 0.3}
    rng = np.random.default_rng(42)
    records = []
    for _ in range(50):
        engine_a = float(rng.uniform(0.1, 0.9))
        engine = {"a": engine_a, "b": 1 - engine_a}
        actual = "a" if rng.uniform() < 0.7 else "b"
        records.append((prior, engine, actual))
    alpha, _ = fit_alpha(records)
    assert alpha < 0.3


def test_fit_alpha_picks_one_when_engine_is_oracle() -> None:
    """If the engine output equals the truth, alpha should approach 1."""
    prior = {"a": 0.5, "b": 0.5}
    records = []
    for actual in ["a"] * 10 + ["b"] * 10:
        engine = {"a": 0.99 if actual == "a" else 0.01,
                  "b": 0.01 if actual == "a" else 0.99}
        records.append((prior, engine, actual))
    alpha, _ = fit_alpha(records)
    assert alpha > 0.9


def test_loocv_alpha_returns_one_per_fold() -> None:
    prior = {"a": 0.6, "b": 0.4}
    records = []
    for actual in ["a", "a", "b", "a", "b"]:
        engine = {"a": 0.8, "b": 0.2}
        records.append((prior, engine, actual))
    fold_alphas, mean_brier = loocv_alpha(records)
    assert len(fold_alphas) == 5
    assert mean_brier > 0


def test_aggregate_brier_zero_for_perfect_engine() -> None:
    prior = {"a": 0.5, "b": 0.5}
    records = [(prior, {"a": 1.0, "b": 0.0}, "a"),
               (prior, {"a": 0.0, "b": 1.0}, "b")]
    assert aggregate_brier(records, alpha=1.0) == pytest.approx(0.0)

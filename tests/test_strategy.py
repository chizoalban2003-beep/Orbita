"""Tests for the strategy layer (issue #7)."""
from __future__ import annotations

import pytest

from orbita import Forecast, strategy


def _forecast(
    probs=None, confidence=None,
) -> Forecast:
    """Minimal hand-built Forecast for unit tests — bypasses the
    Monte Carlo so the strategy logic can be tested deterministically."""
    probs = probs or {}
    confidence = confidence or {n: 1.0 for n in probs}
    return Forecast(
        market_names=tuple(probs),
        probs=probs,
        confidence=confidence,
        joint_counts={},
        n_trials=1,
    )


def test_kelly_skips_below_min_edge() -> None:
    fc = _forecast(probs={"wdl": {"a": 0.55, "b": 0.45}})
    portfolio = strategy.kelly(
        forecast=fc,
        lines={"wdl": {"a": 1.90, "b": 2.10}},
        bankroll=1000.0, min_edge=0.10,
    )
    assert portfolio.bets == []


def test_kelly_sizes_with_edge() -> None:
    # implied_a = 1/1.6 = 0.625; engine_a = 0.75; edge = 0.125
    fc = _forecast(probs={"wdl": {"a": 0.75, "b": 0.25}})
    portfolio = strategy.kelly(
        forecast=fc,
        lines={"wdl": {"a": 1.60}},
        bankroll=1000.0, fraction=0.25, min_edge=0.05,
        min_confidence=0.0,
    )
    assert len(portfolio.bets) == 1
    bet = portfolio.bets[0]
    assert bet.market == "wdl"
    assert bet.selection == "a"
    assert bet.stake > 0
    assert bet.edge == pytest.approx(0.125, abs=1e-9)


def test_kelly_skips_below_min_confidence() -> None:
    fc = _forecast(
        probs={"wdl": {"a": 0.75, "b": 0.25}},
        confidence={"wdl": 0.30},
    )
    portfolio = strategy.kelly(
        forecast=fc,
        lines={"wdl": {"a": 1.60}},
        bankroll=1000.0, min_confidence=0.50,
    )
    assert portfolio.bets == []
    assert any("confidence" in n for n in portfolio.notes)


def test_kelly_total_stake_capped_at_bankroll() -> None:
    fc = _forecast(probs={
        "m1": {"a": 0.95, "b": 0.05},
        "m2": {"a": 0.95, "b": 0.05},
        "m3": {"a": 0.95, "b": 0.05},
    })
    # Full Kelly at p=0.95 / odds 2.0 is enormous; three markets together
    # would blow past bankroll without the cap.
    portfolio = strategy.kelly(
        forecast=fc,
        lines={"m1": {"a": 2.0}, "m2": {"a": 2.0}, "m3": {"a": 2.0}},
        bankroll=100.0, fraction=1.0, min_edge=0.0,
        min_confidence=0.0,
    )
    assert portfolio.total_stake == pytest.approx(100.0)


def test_confidence_weighted_kelly_shrinks_with_confidence() -> None:
    high = _forecast(
        probs={"wdl": {"a": 0.75, "b": 0.25}},
        confidence={"wdl": 0.9},
    )
    low = _forecast(
        probs={"wdl": {"a": 0.75, "b": 0.25}},
        confidence={"wdl": 0.3},
    )
    lines = {"wdl": {"a": 1.60}}
    p_high = strategy.confidence_weighted_kelly(high, lines, bankroll=1000.0,
                                                min_edge=0.0)
    p_low = strategy.confidence_weighted_kelly(low, lines, bankroll=1000.0,
                                               min_edge=0.0)
    assert p_high.bets[0].stake > p_low.bets[0].stake


def test_flat_emits_unit_stake() -> None:
    fc = _forecast(probs={"wdl": {"a": 0.75, "b": 0.25}})
    portfolio = strategy.flat(
        forecast=fc,
        lines={"wdl": {"a": 1.60}},
        bankroll=1000.0, stake_per_bet=10.0,
        min_edge=0.05, min_confidence=0.0,
    )
    assert len(portfolio.bets) == 1
    assert portfolio.bets[0].stake == 10.0


def test_ev_only_lists_all_positive_edges() -> None:
    fc = _forecast(probs={"wdl": {"a": 0.60, "b": 0.45}})
    portfolio = strategy.ev_only(
        forecast=fc,
        lines={"wdl": {"a": 2.00, "b": 2.00}},   # implied 0.5 each
        min_edge=0.05,
    )
    assert len(portfolio.bets) == 1   # only 'a' clears 0.5 + 0.05
    assert portfolio.bets[0].selection == "a"


def test_hedge_returns_none_when_market_overround_kills_ev() -> None:
    fc = _forecast(probs={"ou": {"over": 0.5, "under": 0.5}})
    # Sum of implied probs at odds 1.85 each is 2/1.85 ≈ 1.08 > 1 → -EV
    h = strategy.hedge(
        forecast=fc, market_name="ou",
        selection_a="over", selection_b="under",
        odds_a=1.85, odds_b=1.85, total_stake=100.0,
    )
    assert h is None


def test_hedge_emits_when_two_way_market_sums_below_one() -> None:
    fc = _forecast(probs={"ou": {"over": 0.55, "under": 0.45}})
    # Sum of implied probs at odds 2.50 each is 2/2.50 = 0.8 < 1 → +EV
    h = strategy.hedge(
        forecast=fc, market_name="ou",
        selection_a="over", selection_b="under",
        odds_a=2.50, odds_b=2.50, total_stake=100.0,
    )
    assert h is not None
    assert h.guaranteed_return > 0
    # Either leg winning yields the same return.
    legA = h.legs[0]; legB = h.legs[1]
    return_if_a_wins = legA.stake * legA.odds - (legA.stake + legB.stake)
    return_if_b_wins = legB.stake * legB.odds - (legA.stake + legB.stake)
    assert return_if_a_wins == pytest.approx(return_if_b_wins, abs=1e-9)


def test_portfolio_expected_value() -> None:
    fc = _forecast(probs={"wdl": {"a": 0.75, "b": 0.25}})
    portfolio = strategy.flat(
        forecast=fc, lines={"wdl": {"a": 1.60}},
        bankroll=1000.0, stake_per_bet=100.0,
        min_edge=0.05, min_confidence=0.0,
    )
    # EV = 0.75 × (100 × 0.6) - 0.25 × 100 = 45 - 25 = 20
    assert portfolio.expected_value == pytest.approx(20.0)

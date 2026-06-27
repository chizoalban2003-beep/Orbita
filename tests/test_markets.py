"""Tests for the multi-market layer (issue #5)."""
from __future__ import annotations

import pytest

from orbita import Market, Match


def test_over_under_market_constructs() -> None:
    m = Market.over_under(line=2.5, prior_over=0.55)
    assert m.name == "over_under_2.5"
    masses = {a.label: a.mass for a in m.space.attractors}
    assert masses["over"] == pytest.approx(0.55)
    assert masses["under"] == pytest.approx(0.45)


def test_btts_market_constructs() -> None:
    m = Market.btts(prior_yes=0.6)
    assert m.name == "btts"
    masses = {a.label: a.mass for a in m.space.attractors}
    assert masses["yes"] == pytest.approx(0.6)
    assert masses["no"] == pytest.approx(0.4)


def test_handicap_market_constructs() -> None:
    m = Market.asian_handicap(line=-1.5, prior_a=0.62,
                              side_a_label="home", side_b_label="away")
    assert m.name == "handicap_-1.5"
    masses = {a.label: a.mass for a in m.space.attractors}
    assert masses["home"] == pytest.approx(0.62)
    assert masses["away"] == pytest.approx(0.38)


def test_invalid_prior_raises() -> None:
    with pytest.raises(ValueError):
        Market.over_under(line=2.5, prior_over=0.0)
    with pytest.raises(ValueError):
        Market.btts(prior_yes=1.1)


def test_match_simulate_returns_per_market_marginals() -> None:
    match = Match(markets=[
        Market.from_sport(
            sport="soccer",
            side_a_label="spain", prior_a=0.59,
            side_b_label="uruguay", prior_b=0.16,
            prior_draw=0.25, draw_label="draw",
        ),
        Market.over_under(line=2.5, prior_over=0.55),
        Market.btts(prior_yes=0.52),
    ])
    forecast = match.simulate(n_trials=20, seed=1, dt=0.2)

    wdl = forecast.market("win_draw_lose")
    assert set(wdl) == {"spain", "draw", "uruguay"}
    assert sum(wdl.values()) == pytest.approx(1.0)

    ou = forecast.market("over_under_2.5")
    assert set(ou) == {"over", "under"}
    assert sum(ou.values()) == pytest.approx(1.0)

    btts = forecast.market("btts")
    assert set(btts) == {"yes", "no"}


def test_forecast_exposes_confidence_per_market() -> None:
    match = Match(markets=[
        Market.over_under(line=2.5, prior_over=0.6),
        Market.btts(prior_yes=0.5),
    ])
    forecast = match.simulate(n_trials=20, seed=2, dt=0.2)
    for name in ("over_under_2.5", "btts"):
        c = forecast.confidence_for(name)
        assert 0.0 <= c <= 1.0


def test_joint_distribution_sums_to_one_and_can_marginalise() -> None:
    match = Match(markets=[
        Market.over_under(line=2.5, prior_over=0.55),
        Market.btts(prior_yes=0.52),
    ])
    forecast = match.simulate(n_trials=30, seed=3, dt=0.2)

    full = forecast.joint(["over_under_2.5", "btts"])
    assert sum(full.values()) == pytest.approx(1.0)

    # Marginalising joint over btts should recover the over_under marginal.
    ou_marg = forecast.joint(["over_under_2.5"])
    direct = forecast.market("over_under_2.5")
    for label, p in direct.items():
        assert ou_marg[(label,)] == pytest.approx(p)


def test_unknown_market_raises() -> None:
    match = Match(markets=[Market.btts(prior_yes=0.5)])
    forecast = match.simulate(n_trials=10, seed=4, dt=0.2)
    with pytest.raises(KeyError):
        forecast.market("does_not_exist")
    with pytest.raises(KeyError):
        forecast.joint(["btts", "ghost"])


def test_shared_initial_conditions_drive_correlation() -> None:
    """Two markets with the SAME well geometry should produce identical
    per-trial outcomes because the (q0, p0) draw is shared in lockstep."""
    twin_a = Market.over_under(line=2.5, prior_over=0.5)
    twin_b = Market(name="twin_clone", space=twin_a.space,
                    sim_kwargs=twin_a.sim_kwargs)
    match = Match(markets=[twin_a, twin_b])
    forecast = match.simulate(n_trials=25, seed=5, dt=0.2)

    # Joint distribution should be concentrated entirely on (over,over)
    # and (under,under) — never (over,under) or (under,over).
    joint = forecast.joint(["over_under_2.5", "twin_clone"])
    diag = joint.get(("over", "over"), 0.0) + joint.get(("under", "under"), 0.0)
    assert diag == pytest.approx(1.0)

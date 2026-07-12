"""The Odds API ingestion — devig math + response parsing (no network)."""
import numpy as np
import pytest

from orbita import odds_api as oa


# ---- Shin devig ----------------------------------------------------------
def test_shin_devig_is_a_valid_distribution():
    probs, z = oa.shin_devig({"home": 2.90, "draw": 3.60, "away": 2.00})
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert all(0.0 < probs[k] < 1.0 for k in ("home", "draw", "away"))
    assert 0.0 <= z < 0.9                       # fitted insider share, small for football


def test_shin_differs_from_proportional_and_removes_more_overround():
    odds = {"home": 2.90, "draw": 3.60, "away": 2.00}
    shin, z = oa.shin_devig(odds)
    prop = oa.proportional_devig(odds)
    # a real (non-zero) insider share, so the two methods genuinely diverge
    assert z > 0.0
    assert any(abs(shin[k] - prop[k]) > 1e-4 for k in ("home", "draw", "away"))
    # both are valid distributions
    assert abs(sum(prop.values()) - 1.0) < 1e-9


def test_shin_reduces_to_fair_book_when_no_overround():
    # a fair 3-way book (Σ 1/o = 1) has no margin, so Shin returns the odds as-is
    fair = {"home": 3.0, "draw": 3.0, "away": 3.0}
    probs, z = oa.shin_devig(fair)
    assert z < 1e-3
    for k in ("home", "draw", "away"):
        assert abs(probs[k] - 1 / 3) < 1e-3


def test_shin_rejects_invalid_odds():
    with pytest.raises(oa.OddsAPIError):
        oa.shin_devig({"home": 1.0, "draw": 3.6, "away": 2.0})


# ---- response parsing ----------------------------------------------------
def _event():
    return {
        "home_team": "Aalesunds FK", "away_team": "Molde FK",
        "commence_time": "2026-07-11T14:00:00Z",
        "bookmakers": [
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Aalesunds FK", "price": 3.05},
                    {"name": "Molde FK", "price": 2.02},
                    {"name": "Draw", "price": 3.70}]}]},
            {"key": "betano", "title": "Betano", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Aalesunds FK", "price": 3.10},
                    {"name": "Molde FK", "price": 2.08},
                    {"name": "Draw", "price": 3.65}]}]},
        ],
    }


def test_extract_1x2_picks_the_named_book_and_maps_outcomes():
    odds, label = oa.extract_1x2(_event(), "Aalesunds FK", "Molde FK", book="pinnacle")
    assert label == "Pinnacle"
    assert odds == {"home": 3.05, "draw": 3.70, "away": 2.02}


def test_extract_1x2_consensus_averages_implied_probs():
    odds, label = oa.extract_1x2(_event(), "Aalesunds FK", "Molde FK", book=None)
    assert "consensus" in label
    # away is the favourite in both books -> lowest odds after averaging
    assert odds["away"] < odds["home"] and odds["away"] < odds["draw"]
    # averaging is on implied prob: 1/away between the two books' 1/2.02 and 1/2.08
    assert 2.02 < odds["away"] < 2.08


def test_extract_1x2_raises_for_absent_book():
    with pytest.raises(oa.OddsAPIError):
        oa.extract_1x2(_event(), "Aalesunds FK", "Molde FK", book="bet365")


# ---- key / network guards ------------------------------------------------
def test_fetch_without_key_raises_cleanly(monkeypatch):
    monkeypatch.delenv("ORBITA_ODDS_API_KEY", raising=False)
    with pytest.raises(oa.OddsAPIError) as ei:
        oa.fetch_events("eliteserien")
    assert "key" in str(ei.value).lower()


def test_sport_key_resolves_alias():
    assert oa._resolve_sport("eliteserien") == "soccer_norway_eliteserien"
    assert oa._resolve_sport("soccer_epl") == "soccer_epl"          # passthrough

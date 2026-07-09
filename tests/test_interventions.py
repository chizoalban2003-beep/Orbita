"""Counterfactual intervention engine."""
import numpy as np

from orbita.interventions import (early_pressure, forecast, injury, low_tempo,
                                   red_card, reprice)

PRIORS = {"home": 0.50, "draw": 0.27, "away": 0.23}


def _sums_to_one(d):
    return abs(sum(d.values()) - 1.0) < 1e-9


def test_base_forecast_valid():
    f = forecast(PRIORS, None, n_trials=120)
    assert _sums_to_one(f) and all(0 <= v <= 1 for v in f.values())


def test_injury_lowers_that_side():
    before, after, _ = reprice(PRIORS, injury("home", 0.30), n_trials=200)
    assert after["home"] < before["home"]           # weakened side drops
    assert _sums_to_one(after)


def test_red_card_hits_harder_than_injury():
    _, a_inj, _ = reprice(PRIORS, injury("away", 0.30), n_trials=200)
    _, a_red, _ = reprice(PRIORS, red_card("away", 0.30), n_trials=200)
    assert a_red["away"] <= a_inj["away"] + 1e-9     # red card ≥ as damaging


def test_early_pressure_lifts_that_side():
    before, after, _ = reprice(PRIORS, early_pressure("away", 0.8), n_trials=200)
    assert after["away"] > before["away"]


def test_low_tempo_favours_draw():
    before, after, _ = reprice(PRIORS, low_tempo(1.8), n_trials=200)
    assert after["draw"] >= before["draw"] - 1e-9    # cagey game lifts the draw


def test_explanation_mentions_a_number():
    _, _, expl = reprice(PRIORS, injury("home"), n_trials=120)
    assert "%" in expl and "P(" in expl

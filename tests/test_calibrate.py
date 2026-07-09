"""Bayesian lever calibration."""
import numpy as np

from orbita.calibrate import (LeverPosterior, historical_prior, likelihood_update,
                              make_iv_for, gaussian_logprior)
from orbita.interventions import Intervention

PRI = {"home": 0.50, "draw": 0.27, "away": 0.23}


def test_posterior_math():
    grid = np.linspace(0, 1, 21)
    post = LeverPosterior("x", "m", grid, gaussian_logprior(grid, 0.6, 0.1))
    assert abs(post.probs().sum() - 1.0) < 1e-9
    assert 0.55 < post.mean() < 0.65 and 0.55 <= post.mode() <= 0.65
    lo, hi = post.ci(0.9)
    assert lo < post.mean() < hi


def test_historical_priors_present():
    for lever in ("red_card", "injury", "low_tempo", "early_pressure"):
        p = historical_prior(lever)
        assert p.lever == lever and len(p.grid) == len(p.logpost) and p.provenance


def test_make_iv_builds_interventions():
    assert isinstance(make_iv_for("injury", "home")(0.2), Intervention)
    iv = make_iv_for("red_card", "away")(0.3)
    assert iv.mass_transfer is not None and iv.momentum is not None


def test_update_increments_and_is_finite():
    post = historical_prior("injury")
    up = likelihood_update(post, PRI, "home", "away", n_trials=40)
    assert up.n_obs == post.n_obs + 1
    assert np.all(np.isfinite(up.logpost))


def test_likelihood_moves_toward_stronger_lever_when_opponent_wins():
    # home carded. If the opponent (away) actually WON, larger red-card magnitude
    # explains it better -> posterior mean should exceed the case where home won.
    flat = LeverPosterior("red_card", "m", np.linspace(0.05, 0.5, 10),
                          np.zeros(10))
    up_away = likelihood_update(flat, PRI, "home", "away", n_trials=60)
    up_home = likelihood_update(flat, PRI, "home", "home", n_trials=60)
    assert up_away.mean() > up_home.mean()

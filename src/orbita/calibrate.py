"""Bayesian calibration of the lever constants — Orbita's slow learner.

The engine does not start cold. Its lever magnitudes (how far a red card pushes
the result, how much an injury transfers) are *physics*, and they can be trained
on data we already hold: 10 seasons of natural experiments (exp 22 / 24). This
module turns that history into a **posterior over each lever constant**, and
then refines it online as the ledger's live reads settle.

The mechanism is the discrete cousin of the differentiable loop the roadmap
points at. For a lever with magnitude ``θ`` and a settled datapoint
``(priors, result)``, the likelihood of ``θ`` is the probability the engine —
run with the lever at ``θ`` — assigned to the outcome that actually happened::

    L(θ) = P_engine(result | priors, lever(θ))

Folding ``log L(θ)`` over many matches sharpens a prior into a posterior.
Historical natural experiments give the *prior* by Aug 21; the ledger's live
settlements give the *online update* — the same Bayes rule, one match at a time,
which is the honest small-``n`` alternative to raw SGD on ∂L/∂θ.

Scope, kept honest: this calibrates the low-dimensional lever magnitudes only —
it does **not** re-tune the engine against the market (the campaign proved that
overfits an efficient line). It answers "how strong is this lever", not "can we
beat the close".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .interventions import (Intervention, early_pressure, forecast, injury,
                            low_tempo, red_card)

_LAB = ("home", "draw", "away")
_EPS = 1e-6


@dataclass
class LeverPosterior:
    """A posterior over one lever's magnitude, on a fixed grid."""

    lever: str
    param: str
    grid: np.ndarray
    logpost: np.ndarray                 # unnormalised log posterior
    provenance: str = ""
    n_obs: int = 0

    def probs(self) -> np.ndarray:
        z = self.logpost - self.logpost.max()
        w = np.exp(z)
        return w / w.sum()

    def mean(self) -> float:
        return float(np.sum(self.grid * self.probs()))

    def mode(self) -> float:
        return float(self.grid[int(np.argmax(self.logpost))])

    def ci(self, level: float = 0.90):
        p = self.probs()
        c = np.cumsum(p)
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        lo = float(np.interp(lo_q, c, self.grid))
        hi = float(np.interp(hi_q, c, self.grid))
        return lo, hi

    def summary(self) -> str:
        lo, hi = self.ci()
        return (f"{self.lever}.{self.param}: mean {self.mean():.3f} "
                f"mode {self.mode():.3f}  90% CI [{lo:.3f},{hi:.3f}]  "
                f"n={self.n_obs}  ({self.provenance})")


def gaussian_logprior(grid, mu: float, sigma: float) -> np.ndarray:
    return -0.5 * ((grid - mu) / sigma) ** 2


# the lever vocabulary → an intervention factory of one scalar magnitude θ
def make_iv_for(lever: str, side: str = "home") -> Callable[[float], Optional[Intervention]]:
    if lever == "red_card":
        return lambda t: red_card(side, t, min(0.5, t * 1.1))
    if lever == "injury":
        return lambda t: injury(side, t)
    if lever == "low_tempo":
        return lambda t: low_tempo(t)
    if lever == "early_pressure":
        return lambda t: early_pressure(side, t)
    raise ValueError(f"unknown lever {lever!r}")


def _default_grid(lever: str) -> np.ndarray:
    if lever in ("red_card", "injury"):
        return np.linspace(0.02, 0.5, 13)
    if lever == "low_tempo":
        return np.linspace(0.02, 0.5, 13)
    return np.linspace(0.1, 1.0, 10)          # early_pressure


def historical_prior(lever: str) -> LeverPosterior:
    """Prior distilled from the completed campaign — the engine's state of
    knowledge before a single live read is logged."""
    grid = _default_grid(lever)
    if lever == "red_card":
        lp = gaussian_logprior(grid, 0.145, 0.045)
        prov = "exp26: calibrated on 60 red-card matches (corrected exp22's 0.30)"
    elif lever == "injury":
        lp = gaussian_logprior(grid, 0.126, 0.05)
        prov = "exp26: calibrated on 60 drift matches (confirmed exp24)"
    elif lever == "low_tempo":
        lp = gaussian_logprior(grid, 0.15, 0.20)          # broad / weak
        prov = "exp23 scalar REJECTED; favourite-lock re-spec UNVALIDATED"
    else:
        lp = np.zeros_like(grid)                          # flat
        prov = "uninformative"
    return LeverPosterior(lever, "magnitude", grid, lp, prov, n_obs=0)


def likelihood_update(post: LeverPosterior, priors: Dict[str, float], side: str,
                      result: str, *, n_trials: int = 80, seed: int = 42
                      ) -> LeverPosterior:
    """One Bayesian step: multiply in L(θ)=P_engine(result|priors,lever(θ))."""
    mk = make_iv_for(post.lever, side)
    add = np.empty_like(post.grid)
    for i, t in enumerate(post.grid):
        p = forecast(priors, mk(float(t)), n_trials=n_trials, seed=seed)
        add[i] = np.log(max(p[result], _EPS))
    return LeverPosterior(post.lever, post.param, post.grid,
                          post.logpost + add, post.provenance, post.n_obs + 1)


def calibrate_from_matches(lever: str, matches: List[Dict], *,
                           start: Optional[LeverPosterior] = None,
                           n_trials: int = 80) -> LeverPosterior:
    """Fold the Bayesian update over historical natural-experiment matches.
    Each match is ``{priors, side, result}``. This is 'train on past data'."""
    post = start if start is not None else historical_prior(lever)
    for m in matches:
        post = likelihood_update(post, m["priors"], m["side"], m["result"],
                                 n_trials=n_trials)
    return post


def update_from_ledger(lever: str, path=None, *, n_trials: int = 80
                       ) -> LeverPosterior:
    """Refine the historical prior with the ledger's settled live reads for one
    lever — the online loop."""
    from . import ledger as _ledger
    data = _ledger.load(path) if path else _ledger.load()
    settled = {s["entry_id"]: s for s in data["settlement"]}
    post = historical_prior(lever)
    for e in data["entry"]:
        if e["read"].get("scenario") != lever or e["id"] not in settled:
            continue
        post = likelihood_update(post, e["market"]["priors"],
                                 e["read"].get("side", "home"),
                                 settled[e["id"]]["result"], n_trials=n_trials)
    return post

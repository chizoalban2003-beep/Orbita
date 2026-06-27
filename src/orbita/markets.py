"""Multi-market event-spaces (GitHub issue #5).

A ``Market`` bundles an :class:`EventSpace` (its own well geometry) with
the integrator kwargs needed to simulate it. A ``Match`` runs one Monte
Carlo loop and classifies each trial against every market in lockstep —
the trial's ``(q0, p0)`` draw is shared, so per-market outcomes are
correlated through their common initial conditions (a Spain rout draw
that produces an extreme W/D/L position also pushes the over/under
geometry the same way).

The result is a :class:`Forecast` with per-market marginal probabilities,
per-market mean confidence (from the saddle-detection layer, #6) and a
joint distribution over outcome tuples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from .domain import Attractor, Body, EventSpace
from .integrator import final_well, simulate
from .sports import template_for


_DEFAULT_CD = 0.04        # soccer template's drag — fine for prop markets
_DEFAULT_DURATION = 600.0


@dataclass
class Market:
    """A single betting market plus the well geometry that classifies it.

    Use the factory classmethods rather than constructing directly.
    """

    name: str
    space: EventSpace
    sim_kwargs: Dict[str, float] = field(
        default_factory=lambda: {"C_d": _DEFAULT_CD,
                                 "duration": _DEFAULT_DURATION}
    )

    @classmethod
    def from_sport(
        cls,
        sport: str,
        side_a_label: str,
        side_b_label: str,
        prior_a: float,
        prior_b: float,
        *,
        prior_draw: Optional[float] = None,
        draw_label: Optional[str] = None,
    ) -> "Market":
        """Win/draw/lose (or win/lose) market using a sport template."""
        from .sports import build_space   # local to avoid cycle
        space, sim_kwargs = build_space(
            sport=sport,
            side_a_label=side_a_label, prior_a=prior_a,
            side_b_label=side_b_label, prior_b=prior_b,
            prior_draw=prior_draw, draw_label=draw_label,
        )
        return cls(name="win_draw_lose", space=space, sim_kwargs=sim_kwargs)

    @classmethod
    def over_under(
        cls, line: float, prior_over: float,
        label_over: str = "over", label_under: str = "under",
    ) -> "Market":
        """Total-goals over/under as a 2-well event-space."""
        if not 0.0 < prior_over < 1.0:
            raise ValueError(f"prior_over must be in (0,1), got {prior_over}")
        space = EventSpace([
            Attractor(label_over,  [5.0, 0.0], prior_over),
            Attractor(label_under, [-5.0, 0.0], 1.0 - prior_over),
        ])
        return cls(name=f"over_under_{line}", space=space)

    @classmethod
    def btts(
        cls, prior_yes: float,
        label_yes: str = "yes", label_no: str = "no",
    ) -> "Market":
        """Both teams to score as a 2-well event-space."""
        if not 0.0 < prior_yes < 1.0:
            raise ValueError(f"prior_yes must be in (0,1), got {prior_yes}")
        space = EventSpace([
            Attractor(label_yes, [5.0, 0.0], prior_yes),
            Attractor(label_no,  [-5.0, 0.0], 1.0 - prior_yes),
        ])
        return cls(name="btts", space=space)

    @classmethod
    def asian_handicap(
        cls,
        line: float,
        prior_a: float,
        side_a_label: str = "a",
        side_b_label: str = "b",
    ) -> "Market":
        """Asian-handicap two-way market (no draw — push refunded)."""
        if not 0.0 < prior_a < 1.0:
            raise ValueError(f"prior_a must be in (0,1), got {prior_a}")
        space = EventSpace([
            Attractor(side_a_label, [5.0, 0.0], prior_a),
            Attractor(side_b_label, [-5.0, 0.0], 1.0 - prior_a),
        ])
        return cls(name=f"handicap_{line:+g}", space=space)


@dataclass
class Forecast:
    """Output of :meth:`Match.simulate`.

    Holds per-market marginal probabilities, mean confidence, and the
    joint distribution over outcome tuples (one entry per (label_market_1,
    label_market_2, ...) combination observed in the Monte Carlo).
    """

    market_names: Tuple[str, ...]
    probs: Dict[str, Dict[str, float]]
    confidence: Dict[str, float]
    joint_counts: Dict[Tuple[str, ...], int]
    n_trials: int

    def market(self, name: str) -> Dict[str, float]:
        if name not in self.probs:
            raise KeyError(
                f"No market {name!r} in forecast; have {list(self.probs)}"
            )
        return dict(self.probs[name])

    def confidence_for(self, name: str) -> float:
        if name not in self.confidence:
            raise KeyError(
                f"No market {name!r} in forecast; have {list(self.confidence)}"
            )
        return self.confidence[name]

    def joint(self, names: List[str]) -> Dict[Tuple[str, ...], float]:
        """Marginalise the full joint down to the requested market subset."""
        for n in names:
            if n not in self.market_names:
                raise KeyError(
                    f"No market {n!r} in forecast; have {self.market_names}"
                )
        idx = [self.market_names.index(n) for n in names]
        out: Dict[Tuple[str, ...], int] = {}
        for key, count in self.joint_counts.items():
            sub = tuple(key[i] for i in idx)
            out[sub] = out.get(sub, 0) + count
        return {k: v / self.n_trials for k, v in out.items()}


@dataclass
class Match:
    """A single event with one or more betting markets attached.

    Each market gets its own event-space (its own well geometry). The
    Monte Carlo loop draws one ``(q0, p0)`` per trial and reuses it
    across every market — this is the channel that makes per-market
    outcomes correlated (a draw biased toward extreme initial momentum
    will land at extremes in BOTH the W/D/L and the over/under markets).
    """

    markets: List[Market]

    def simulate(
        self,
        n_trials: int = 100,
        seed: int = 42,
        dt: float = 0.1,
    ) -> Forecast:
        rng = np.random.default_rng(seed=seed)
        names = tuple(m.name for m in self.markets)
        counts: Dict[str, Dict[str, int]] = {
            m.name: {a.label: 0 for a in m.space.attractors}
            for m in self.markets
        }
        confidence_sum: Dict[str, float] = {m.name: 0.0 for m in self.markets}
        joint_counts: Dict[Tuple[str, ...], int] = {}

        for _ in range(n_trials):
            q0 = rng.normal(scale=[0.3, 0.2], size=2)
            p0 = rng.normal(scale=[0.15, 0.15], size=2)
            trial_outcome: List[str] = []
            for m in self.markets:
                body = Body(mass=1.0, q0=q0.copy(), p0=p0.copy())
                sol = simulate(m.space, body=body, dt=dt, **m.sim_kwargs)
                label = final_well(sol, m.space)
                counts[m.name][label] += 1
                confidence_sum[m.name] += sol["confidence"]
                trial_outcome.append(label)
            key = tuple(trial_outcome)
            joint_counts[key] = joint_counts.get(key, 0) + 1

        probs = {
            name: {label: c / n_trials for label, c in counts[name].items()}
            for name in counts
        }
        confidence = {
            name: confidence_sum[name] / n_trials for name in confidence_sum
        }
        return Forecast(
            market_names=names,
            probs=probs,
            confidence=confidence,
            joint_counts=joint_counts,
            n_trials=n_trials,
        )

"""Orbita — a mechanistic predictive engine for probabilistic events."""
from .calibration import (
    aggregate_brier,
    blend,
    brier,
    fit_alpha,
    loocv_alpha,
)
from .domain import (
    Attractor,
    Body,
    EventSpace,
    Observation,
    Player,
    Roster,
    Sensor,
    event_space_from_rosters,
    sensors_from_lineup,
)
from .providers import (
    RatingProvider,
    SnapshotProvider,
    UnderstatMatch,
    UnderstatMatchProvider,
)
from .forces import (
    gravity_force,
    drag_force,
    linear_ramp_schedule,
    piecewise_constant_schedule,
    ornstein_uhlenbeck_schedule,
    potential_energy,
    hamiltonian,
)
from .integrator import (
    simulate,
    simulate_from_state,
    kick,
    final_well,
    final_well_posterior,
)
from .markets import Forecast, Market, Match
from .sports import TEMPLATES, build_space, template_for
from . import strategy
from .strategy import Bet, HedgeBet, Portfolio

__version__ = "0.3.7"

__all__ = [
    "Attractor",
    "Bet",
    "EventSpace",
    "Body",
    "Forecast",
    "HedgeBet",
    "Market",
    "Match",
    "Portfolio",
    "strategy",
    "Observation",
    "Player",
    "RatingProvider",
    "Roster",
    "Sensor",
    "SnapshotProvider",
    "event_space_from_rosters",
    "sensors_from_lineup",
    "gravity_force",
    "drag_force",
    "linear_ramp_schedule",
    "piecewise_constant_schedule",
    "ornstein_uhlenbeck_schedule",
    "potential_energy",
    "hamiltonian",
    "simulate",
    "simulate_from_state",
    "kick",
    "final_well",
    "final_well_posterior",
    "TEMPLATES",
    "build_space",
    "template_for",
    "blend",
    "brier",
    "aggregate_brier",
    "fit_alpha",
    "loocv_alpha",
]

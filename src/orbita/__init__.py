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
)
from .forces import gravity_force, drag_force, potential_energy, hamiltonian
from .integrator import simulate, final_well
from .sports import TEMPLATES, build_space, template_for

__version__ = "0.3.0"

__all__ = [
    "Attractor",
    "EventSpace",
    "Body",
    "Observation",
    "Player",
    "Roster",
    "Sensor",
    "event_space_from_rosters",
    "gravity_force",
    "drag_force",
    "potential_energy",
    "hamiltonian",
    "simulate",
    "final_well",
    "TEMPLATES",
    "build_space",
    "template_for",
    "blend",
    "brier",
    "aggregate_brier",
    "fit_alpha",
    "loocv_alpha",
]

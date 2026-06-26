"""
04_market_regime.py

Financial demo: the same engine, applied to a market-regime forecast.

The orbital body is the S&P 500, represented in (drawdown, momentum)
coordinates. The wells are two macro regimes:

    - Bull regime: index at all-time-high
    - Bear regime: index 20% off ATH (correction territory)

Drag is realized volatility — high VIX bleeds momentum. The engine forecasts
which regime the index drifts toward over the next ~30 trading days.

This is a *demo*, not an investment thesis. The priors below are placeholders
to illustrate the engine working on a non-sports domain.

Run with::

    python examples/04_market_regime.py
"""
from __future__ import annotations

from orbita import Attractor, Body, EventSpace, final_well, simulate


# Hypothetical state for the demo
current_drawdown_pct   = -3.0    # 3% off ATH
current_momentum_5d    = 1.5     # +1.5% over last 5 days
VIX_level              = 18.0    # below the 20-vol threshold

# Two regime attractors
space = EventSpace([
    Attractor(label="bull_regime", position=[0.0,   0.0], mass=0.55),
    Attractor(label="bear_regime", position=[-20.0, 0.0], mass=0.45),
])

body = Body(
    mass=1.0,
    q0=[current_drawdown_pct, 0.0],
    p0=[current_momentum_5d,   0.0],
)

# VIX-derived drag — low vol → low friction, high vol → high friction
C_d = max(0.005, (VIX_level - 10.0) / 200.0)

print("=== Orbita: market-regime forecast (30-day horizon) ===")
print(f"Bull prior        = {space.attractors[0].mass}")
print(f"Bear prior        = {space.attractors[1].mass}")
print(f"Body position     = drawdown={current_drawdown_pct:+.1f}%, momentum={current_momentum_5d:+.1f}%")
print(f"VIX-derived drag  = {C_d:.4f}")
print()

# Treat 1 sim-second ≈ 0.1 trading days; 30 days ≈ 300 sim-seconds.
sol = simulate(space, body=body, duration=300.0, C_d=C_d, dt=0.05)

qf = sol["q"][-1]
print(f"Final position    = ({qf[0]:.3f}, {qf[1]:.3f})")
print(f"Predicted regime  = {final_well(sol, space)}")

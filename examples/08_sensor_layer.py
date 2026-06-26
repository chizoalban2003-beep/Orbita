"""
08_sensor_layer.py

Demonstrate the in-play sensor layer (GitHub issue #2).

We replay the Norway/France 2026 match with a synthetic xG stream and a
red-card shock. The kickoff prior is the bookmaker consensus
(France 60% / Draw 21% / Norway 19%). Then through the match:

    - Norway accumulates xG over the first half (+0.1 chunks per ~3 min)
    - France gives away a red card at the hour mark
    - France stages a late xG push (the way they actually scored 4 goals
      in the real match)

The point is NOT to fit the real outcome (we already published a
calibration on that). It's to demonstrate that the engine's outcome
distribution NOW MOVES IN RESPONSE TO IN-PLAY DATA — the v0.2 engine
was frozen at kickoff, the v0.3 engine isn't.

Run with::

    PYTHONPATH=src python3 examples/08_sensor_layer.py
"""
from __future__ import annotations

import numpy as np

from orbita import (
    Body,
    Observation,
    Sensor,
    build_space,
    final_well,
    simulate,
)


N_TRIALS = 60
SEED = 20260626

# Bookmaker kickoff priors
PRIOR = {"france_win": 0.60, "draw": 0.21, "norway_win": 0.19}

# Higher drag than the v0.2 backtest soccer template so the body stays
# in flight long enough for in-play observations to bend its trajectory.
# (At C_d = 0.04 the body falls into the deepest well within ~5 sim-s
# regardless of mass updates — too fast for the demo.)
SIM_C_d = 0.20
SIM_DURATION = 600.0


# ----- sensors --------------------------------------------------------------
# Each sensor's likelihood returns a *mass multiplier* applied to its target
# well when an observation arrives. 1.0 = no change.

xg_norway = Sensor(
    name="xg_norway",
    target="norway_win",
    likelihood=lambda xg: 1.0 + 3.0 * xg,   # 0.2 xG → 1.6× boost on norway_win
)
xg_france = Sensor(
    name="xg_france",
    target="france_win",
    likelihood=lambda xg: 1.0 + 3.0 * xg,
)
red_card_france = Sensor(
    name="red_card_france",
    target="france_win",
    likelihood=lambda _: 0.40,              # red card cuts france well to 40%
)


# ----- in-play stream -------------------------------------------------------
# Times are in sim-seconds. We pack the early observations close to t=0
# so the mass field updates while the body is still in flight, before
# it commits to a well.

OBSERVATIONS = (
    # Norway pressing early — observations arrive at sim-t 0.5, 1, 1.5...
    [Observation(t=0.5 + 0.5 * i, sensor="xg_norway", value=0.2) for i in range(8)]
    # France red card at sim-t 4.5
    + [Observation(t=4.5, sensor="red_card_france", value=1.0)]
    # Late France push starting sim-t 5
    + [Observation(t=5.0 + 0.5 * i, sensor="xg_france", value=0.25) for i in range(8)]
)


def run(n_trials: int, observations) -> dict:
    """Run a Monte Carlo with the given observation stream and return the
    outcome distribution."""
    counts = {label: 0 for label in PRIOR}
    rng = np.random.default_rng(seed=SEED)
    for _ in range(n_trials):
        # Rebuild the space per trial — observations mutate it in place
        space, _ = build_space(
            sport="soccer",
            side_a_label="france_win",
            side_b_label="norway_win",
            prior_a=PRIOR["france_win"],
            prior_b=PRIOR["norway_win"],
            prior_draw=PRIOR["draw"],
            draw_label="draw",
        )
        q0 = rng.normal(scale=[0.3, 0.2], size=2)
        p0 = rng.normal(scale=[0.15, 0.15], size=2)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(
            space, body=body, dt=0.05,
            duration=SIM_DURATION, C_d=SIM_C_d,
            sensors=[xg_norway, xg_france, red_card_france],
            observations=observations,
        )
        counts[final_well(sol, space)] += 1
    total = sum(counts.values())
    return {label: c / total for label, c in counts.items()}


def fmt(probs: dict) -> str:
    return "  ".join(f"{k:<11s}={v:>5.1%}" for k, v in probs.items())


print("=== Orbita v0.3: sensor layer demo (Norway vs France replay) ===")
print(f"Kickoff prior         : {fmt(PRIOR)}")
print(f"Monte Carlo N         : {N_TRIALS}")
print()

# 1. No observations: equivalent to v0.2 priors-only run
no_obs = run(N_TRIALS, observations=[])
print("--- no in-play data (kickoff-frozen, equivalent to v0.2)")
print(f"  Outcome distribution : {fmt(no_obs)}")
print()

# 2. Norway-press only (first 30 minutes)
norway_press = [o for o in OBSERVATIONS if o.sensor == "xg_norway"]
norway_only = run(N_TRIALS, observations=norway_press)
print("--- after Norway's early xG push (no other events)")
print(f"  Outcome distribution : {fmt(norway_only)}")
print()

# 3. Add the France red card
norway_plus_red = norway_press + [o for o in OBSERVATIONS
                                  if o.sensor == "red_card_france"]
after_red = run(N_TRIALS, observations=norway_plus_red)
print("--- add France red card at the hour mark")
print(f"  Outcome distribution : {fmt(after_red)}")
print()

# 4. Full stream including France's late push
final = run(N_TRIALS, observations=OBSERVATIONS)
print("--- full in-play stream (Norway press + red + France late push)")
print(f"  Outcome distribution : {fmt(final)}")
print()

print("Interpretation:")
print("  - The forecast moves with each piece of in-play data — the engine")
print("    is no longer frozen at kickoff.")
print("  - This is the v0.3 capability the multi-sport backtest said was")
print("    needed: orthogonal signal beyond the bookmaker prior.")
print("  - The Bayesian-style update is mass-multiplicative; future work")
print("    is fitting the likelihood functions to historical data instead")
print("    of hand-picking them (see calibration loop, roadmap).")

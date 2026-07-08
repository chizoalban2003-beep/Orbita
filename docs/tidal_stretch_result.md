# Tidal Stretching (Game-State Deformation) on In-Play O/U

**Status:** Empirical result, July 2026. In-repo experiment 19.
Reproducible from `PYTHONPATH=src python3 experiments/19_tidal_stretch.py`.

## The hypothesis

A static closing line cannot price the non-linear tipping point of
game-state desperation: when a match is heading decisively toward a
result late on, the trailing side abandons defensive shape, so the goals
axis "stretches" and late goals become more likely. Encode this as a new
force coupling the two axes of the joint event space — the body's
proximity to a decisive W/D/L well deforms the Over/Under well — scaled by
an exponential time ramp toward the whistle.

Two concrete forms were tested (`orbita.forces.tidal_force` and the
directional variant in exp 19), both a y-only force with an exponential
time ramp `exp(λ·(t/T − 1))`:

* **symmetric** — magnitude `κ·|F_grav,x|`, direction `sign(q_y)`:
  amplify whichever O/U lean the state already has, scaled by result
  decisiveness.
* **directional** — over-pressure `κ·(|x|·e^{−|x|/x₀})`, always toward
  *over*: bump-shaped in the goal margin, ~0 at level, peaks near a
  one-goal margin (chasing team all-in), fades at a blowout (game dead).

## The test (honest, in-play)

`football-data.co.uk`'s cached EPL 2024/25 CSV carries **half-time
scores**. For each of 380 matches we seed the orbiting body at its real
half-time state — `x₀` from the goal difference, `y₀` from goals banked
vs the 2.5 line — with well masses left at the pre-match de-vigged B365
priors (the field is the market's belief; the body is the live state).
We then simulate the second half in the joint 6-well H/D/A × O/U space
with the tidal force OFF and ON, and score the actual full-time
Over/Under 2.5 outcome (Brier). `κ` is trained on the first 60% of the
season by date and evaluated out-of-sample on the last 40%.

## Result

Tidal deformation **does not help** — Brier degrades monotonically as `κ`
rises, for both forms, so training selected `κ = 0` (force off). N = 40
Monte-Carlo trials/match, λ = 3.0.

| κ (train, first 60%) | O/U Brier — symmetric | O/U Brier — directional |
| -------------------- | --------------------- | ----------------------- |
| **0.00**             | **0.3534**            | **0.3534**              |
| 0.02                 | 0.3536                | 0.3573                  |
| 0.05                 | 0.3538                | 0.3628                  |
| 0.10                 | 0.3546                | 0.3749                  |
| 0.20                 | 0.3587                | 0.3782                  |
| 0.35                 | 0.3624                | 0.3889                  |

With the trained `κ = 0`, the held-out set (n = 152) is by construction
identical on/off; the tidal force adds nothing to recover.

This is not a sign error or a bug: `κ = 0` reproduces the baseline
exactly (`tests/test_tidal.py`), and the *symmetric* form — which pushes
the goals axis in **both** directions following the lean — degrades just
as surely as the one-directional form. Adding any goals-axis push from
the half-time state moves a well-calibrated estimate away from
calibration.

## What did work (and the real reading)

The half-time seeding itself is strongly informative — the in-play engine
(tidal off) scores **0.3724** Brier on the held-out set versus **0.5096**
for the pre-match closing line. That is *not* a market edge (the in-play
engine knows the half-time score the pre-match line did not); it simply
confirms the in-play state-injection path works and stays calibrated.

The likely reason the tidal force fails: the two real sub-effects — a
chasing team creating goals (over) and a leader killing the game (under)
— roughly cancel in aggregate, and at the half-time (45') resolution the
bookmaker's conditional second-half goal expectation is already close to
efficient. The desperation tipping point lives in the 75'+ window, which
this data cannot isolate.

## Next test (before reviving the idea)

A fair test needs a genuine late-game checkpoint (75'+ state) and, ideally,
the **live** Over/Under line at that moment as the benchmark — not the
pre-match line. Minute-level state + in-play odds would let the tidal
force be judged against what the market actually prices in play, which is
the only setting where its claimed edge could exist.

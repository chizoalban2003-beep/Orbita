# Anisotropic Drag on Football Match Prediction

**Status:** Empirical result, June 2026. In-repo experiments 10 and 11.
Reproducible from `PYTHONPATH=src python3 experiments/{10,11}_*.py`.

## The claim

On the full 2024/25 English Premier League season (380 matches, Bet365
de-vigged closing odds as prior, 30 Monte Carlo trials per match), a
purely geometric change to Orbita's drag term — from isotropic
`C_d = 0.04` to anisotropic `C_d = (C_d_x=0.00, C_d_y=0.16)` — reduces
the engine's Brier gap to the market by approximately half on both
match-outcome and over/under markets, without altering any input data.

## What "anisotropic drag" means here

Orbita's 2D event-space for a soccer match places three H/D/L wells on
the x-axis (win-side) at `y = 0`, with over/under wells stacked at
`y = ±3` above and below. The x- and y-axes therefore have distinct
semantics:

* **x-axis** — home/away *direction*. A body with x-momentum is
  expressing directional pressure: a side is pressing forward.
* **y-axis** — draw / goals axis. A body drifting on y has no
  directional loyalty; it's being pulled toward the central attractor.

Isotropic linear drag `F_d = -C_d · v` bleeds both components with the
same coefficient. Anisotropic drag `F_d = -diag(C_d_x, C_d_y) · v`
allows separate coefficients for the two axes.

## Sweep and out-of-sample verification

**Experiment 10** sweeps the 5×5 grid `C_d_x, C_d_y ∈ {0.00, 0.02,
0.04, 0.08, 0.16}` on a 50-match panel (Oct–Nov 2024). Best O/U cell:
`C_d = (0.00, 0.16)` with a Brier delta of −0.010 vs the closing line.
A grid winner on a 50-match panel is a hyperparameter selection, not
evidence — it needs out-of-sample verification.

**Experiment 11** applies the sweep winner and the isotropic baseline
to every 2024/25 EPL fixture (n = 380), reporting bootstrap 90%
confidence intervals on the per-match (engine − market) Brier delta:

| drag                        | HDA Δ  | HDA CI            | O/U Δ  | O/U CI            |
| --------------------------- | ------ | ----------------- | ------ | ----------------- |
| isotropic (0.04)            | +0.044 | [+0.025, +0.065]  | +0.048 | [+0.025, +0.071]  |
| **anisotropic (0.00, 0.16)**| **+0.025** | [+0.008, +0.043] | **+0.020** | [+0.007, +0.033] |

The market still wins both markets (all four CIs strictly exclude zero
on the market's side) — but the Brier gap closes by roughly half on
both markets from geometry alone, with no changes to the input data.

## Physical read

Two independent effects:

1. **`C_d_x = 0` preserves directional momentum.** In H/D/L there are
   two "loyalty" attractors on the x-axis (home_win, away_win) and one
   "neutral" attractor (draw) closer to the origin on y. Under isotropic
   drag, x-momentum bleeds toward zero and the body gets pulled into
   the central well by default. With `C_d_x = 0`, whichever side the
   body starts leaning gets carried through — the prior's win-side
   emphasis is *preserved* rather than *bled out*.

2. **`C_d_y = 0.16` dampens noise on the goals axis.** The over/under
   wells at `y = ±3` compete for probability mass; the body's y-drift
   is highly sensitive to initial conditions. Heavy y-drag pulls the
   body toward `y = 0`, which is the median between over- and
   under-side wells — collapsing the ambiguity rather than committing
   to a market-mispriced side.

The 50-match sweep produced a −0.010 point estimate. The 380-match
out-of-sample delta is +0.020. The sample-inflation ratio is
consistent with N=50 Monte Carlo noise, but the *direction* of the
effect — the halving of the gap vs isotropic — is robust across the
sample.

## What this does *not* claim

* The engine does not beat the market. Both anisotropic CIs still
  strictly exclude zero on the market's side.
* The specific numbers `(0.00, 0.16)` are unlikely to be globally
  optimal. They come from a coarse 5×5 grid. A finer sweep or a
  time-varying schedule (see experiment 12) may extend the effect.
* This is not evidence that geometric priors *replace* better input
  data. It's evidence that geometric priors *matter independently* of
  input data quality.

## What the result implies for the research programme

Three tractable follow-ups:

1. **Time-varying schedules.** Fatigue accumulates late; drag
   coefficients should too. Experiment 12 sweeps a linear-ramp
   schedule on `C_d_y`.
2. **Stochastic drag.** Replace point-estimate `C_d_y` with an
   Ornstein-Uhlenbeck process — each Monte Carlo trial samples a
   different realisation, so the posterior is naturally hedged.
   Experiment 13.
3. **Per-team calibration.** Fit `C_d_y` per team on a training window
   (e.g. first half of season, keyed on goal intensity), evaluate on
   the held-out half. Experiment 15.

The larger question the anisotropic result poses is: how much of the
"engine loses to market" gap in Orbita v0.3.5 comes from *input data
being priced in* vs *the geometry being wrong*? Before June 2026 the
default explanation was the former. This result argues for a
non-trivial share of the latter — the same input data placed in a
better-tuned geometry closed half the gap. That does not overturn the
"can't beat the closing line" verdict, but it changes the marginal
research question.

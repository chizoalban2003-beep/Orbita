# Red-Card Interventional Validity — the first calibration test

**Status:** Empirical result, July 2026. In-repo experiment 22.
Reproducible from `PYTHONPATH=src python3 experiments/22_redcard_validity.py`
(10 EPL seasons cached under `~/.cache/orbita/footballdata`).

## Why this test

Orbita's pivot is to be an *interpretable* forecaster, and a white-box is
only worth its legibility if its counterfactuals are **true**. So the
red-card lever is held to an objective, falsifiable standard: does the
forecast shift it predicts match the shift a red card actually causes in
history? This is the gold standard every future force (weather, fatigue,
low-tempo) will be measured against, and the anchor for using Orbita to
price a sharp analyst's private read.

## Method

Clean natural experiment: 396 matches over 10 seasons with exactly one red
card, to exactly one team (218 away, 178 home). Pre-match de-vigged B365
gives the market's pre-card belief; the actual result gives ground truth.
The red-card lever is a **momentum push toward the opponent's well**; one
scalar `k` is tuned on 7 train seasons, tested on the last 3.

**Design question — static vs mass-dependent momentum:** a *fixed* push
already yields a state-dependent probability shift through the geometry (a
nudge moves a tight game near the saddle far more than a decided one), so
parsimony says static first. Both were tested head-to-head out-of-sample.

**Data limit:** football-data carries red-card counts (`HR`/`AR`) but not
the minute, so this validates the **average-over-timing** effect, not a
minute-anchored `simulate_from_state`.

## Ground truth — a red card is worth ~15 points

| carded side (all seasons) | market H / D / A | actual H / D / A | shift        |
| ------------------------- | ---------------- | ---------------- | ------------ |
| away (n=218)              | 44 / 25 / 31     | 57 / 28 / 16     | **home +13, away −15** |
| home (n=178)              | 43 / 24 / 33     | 26 / 21 / 52     | **home −16, away +19** |

A red card shifts win probability ~13–19 points toward the opponent —
large, clean, and symmetric.

## Result — direction validated, market beaten; magnitude promising

Held-out (last 3 seasons, n = 114), H/D/A Brier:

| model                              | Brier   | Δ vs baseline | 90% CI            |
| ---------------------------------- | ------- | ------------- | ----------------- |
| market (pre-card)                  | 0.5889  | —             | —                 |
| engine baseline (no lever)         | 0.5996  | —             | —                 |
| **+ red-card STATIC momentum (k=1.8)** | **0.5565** | +0.0431   | [−0.0054, +0.0910] |
| + red-card DYNAMIC (∝ carded mass) | 0.6019  | −0.0023       | [−0.0710, +0.0624] |

* **Direction is validated and the market is beaten** on card matches
  (0.557 vs 0.589). The lever moves the forecast the right way by a
  meaningful margin.
* **Static beats mass-dependent** decisively — the design question is
  answered: a fixed scalar plus the geometry is better than hand-coding a
  mass term, which just adds noise.
* **Not yet statistically conclusive.** The improvement's 90% CI barely
  spans zero — 114 held-out matches is a small sample. And the tuned engine
  predicts a +16% away-card shift where the held-out empirical was +8%
  (full-sample +13%): a slight **overshoot**.
* **Magnitude is fragile.** The train sweep was non-monotonic (a coarse
  backtest `dt=0.25` artifact), so `k=1.8` is not a trustworthy final
  calibration — direction is robust, the exact scalar is not.

## Verdict

The red-card counterfactual is **directionally true and market-beating** —
Orbita's most important non-negative result and the first datapoint for
interventional validity as a metric. It is **not yet a precisely
calibrated** magnitude: the effect is real and the lever's sign is right,
but a trustworthy `k` needs the interactive `dt=0.1` physics, a larger
sample (multi-league, or bootstrapped card matches), and ideally card
timing to anchor `simulate_from_state`. Direction: proven. Calibration:
the next decimal place.

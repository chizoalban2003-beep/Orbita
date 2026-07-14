# Out-of-Sample Lever Calibration — the constants generalise

**Status:** Robustness check, July 2026. In-repo experiment 28.
Reproducible from `PYTHONPATH=src python3 experiments/28_oos_calibration.py`
(env: `ORBITA_OOS_DIVS`, `ORBITA_NCAL`, `ORBITA_NTRIALS`).

## Why this test

exp26 fit the three lever magnitudes on six big-5 leagues (E0,E1,D1,SP1,I1,F1).
A fair worry: are those numbers *physics*, or did they overfit those particular
competitions? This refits each lever **from a flat (uninformative) prior** on a
**disjoint** set of leagues the original calibration never saw — Netherlands,
Belgium, Portugal, Turkey, Greece (N1,B1,P1,T1,G1) — and asks whether the
in-sample constant lands inside the independent held-out posterior.

## Result — all three replicate

80 matches per lever, flat prior, N=60 MC trials:

```
  lever            in-samp   OOS mean        OOS 90% CI    n   verdict
  red_card           0.145      0.173     [0.102, 0.206]   80  REPLICATES
  injury             0.126      0.121     [0.020, 0.288]   80  REPLICATES
  early_pressure     0.205      0.272     [0.114, 0.334]   80  REPLICATES
```

Every in-sample constant falls inside its held-out 90% CI. Fit on five leagues
we never touched, the magnitudes reappear — the levers are **football physics,
not big-5 artifacts.**

## Read honestly, lever by lever

* **red_card — strong.** OOS mean 0.173, tight CI [0.102, 0.206], in-sample
  0.145 comfortably inside. This is the cleanest replication, matching the
  lever's clean ~15pt natural-experiment signal.
* **injury — mean spot-on, identification weak.** OOS mean 0.121 ≈ in-sample
  0.126, but the CI is **wide** [0.020, 0.288]. The point estimate replicating
  is reassuring; the width restates exp24's standing caveat — the drift
  instrument is noisy and the effect sub-threshold, so the magnitude is poorly
  pinned in *any* sample.
* **early_pressure — replicates with upward drift.** OOS mean 0.272 vs 0.205
  in-sample; the in-sample value is inside the CI [0.114, 0.334] but the held-out
  centre is ~0.07 higher. Consistent with the lever's "upper bound" caveat: the
  HT-lead instrument carries a goal (mass), and league scoring rates differ, so
  the fitted magnitude drifts with the sample.

## Honest scope — what this does and does not establish

* **Does:** the lever *magnitudes* generalise across European football. It
  de-risks the calibration — Orbita is a **faithful mirror** of how these
  in-match conditions move a result, not a curve fit to five specific leagues.
  This is the "past data → faithful calculator" north-star, confirmed wider.
* **"Replicates" is a soft bar for two of three.** It means the in-sample value
  sits inside the held-out CI — and for injury/early_pressure those CIs are wide
  enough to swallow a range of values. Only red_card clears a *tight* bar. Call
  injury/early_pressure "consistent with replication," not "tightly confirmed".
* **Does NOT:** create or imply market edge. This validates engine fidelity, the
  thing the campaign already separated from alpha. A well-calibrated lever prices
  a private read faithfully; it does not beat the close.
* **Not fully independent:** same data source (football-data.co.uk), same three
  instruments — out-of-sample across *leagues*, not across *method*. A stronger
  future check would fit the same magnitudes from a different instrument.

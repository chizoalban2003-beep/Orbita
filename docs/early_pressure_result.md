# Early-Pressure Calibration — a data-anchored prior for the momentum lever

**Status:** Empirical result, July 2026. In-repo experiment 26 (`ORBITA_LEVER=early_pressure`).
Reproducible from `ORBITA_LEVER=early_pressure PYTHONPATH=src python3 experiments/26_precalibrate.py`
(env: `ORBITA_DIVS`, `ORBITA_NCAL`, `ORBITA_NTRIALS`, `ORBITA_HTMAX`).

## Why this test

The momentum primitive (`early_pressure`, and the momentum half of `red_card`)
was *directionally* validated on red cards (exp 22), but its **magnitude** was
never fit — `historical_prior("early_pressure")` shipped **flat / uninformative**
while `injury` (0.126) and `red_card` (0.145) already carried Bayesian priors
from exp 26. The first live read that used this lever (Sandefjord v HamKam,
`early_pressure('away', 0.15)`) was therefore an *unanchored* bet: a miss would
say nothing about "was the market wrong" versus "was 0.15 the wrong magnitude,"
because no prior pinned it. This test gives the lever a real prior.

## Sourcing the condition — the half-time lead

football-data has no in-match timeline besides the **half-time score**
(`HTHG/HTAG/HTR`). That is the only *temporally-early* signal available, so it is
the natural instrument for "one side started on top": **the side leading at
half-time is the side that got early momentum.** Prior = pre-match closing devig
(`PSCH/PSCD/PSCA`), event = who led at HT, ground truth = `FTR`.

**Honest confound (named, not hidden):** a half-time *lead* contains a goal,
which is partly **mass**, not pure momentum. Fitting a momentum lever to it
absorbs some mass effect, so the fitted magnitude is an **upper bound** on pure
early-momentum — the mirror of the injury lever's "sub-threshold" caveat. To
lean the instrument toward *edge* rather than *dominance*, the loader defaults to
**narrow (1-goal) HT leads only** (`ORBITA_HTMAX=1`), excluding blowouts that are
accumulated superiority rather than an early tilt.

## Result — the flat prior collapses to a tight peak

60 narrow-HT-lead matches (6 divisions × 10 seasons, subsampled), N=60 MC
trials/forecast, grid 0.1–1.0:

```
PRIOR      mean 0.550  mode 0.100  90% CI [0.100, 0.950]   (flat/uninformative)
POSTERIOR  mean 0.205  mode 0.200  90% CI [0.100, 0.279]   n=60

  0.100  0.164  ██████
  0.200  0.632  ██████████████████████
  0.300  0.196  ███████
  0.400  0.008
  0.500+ 0.000
```

The data pulled the flat prior into a **well-identified peak at ≈0.20** — and,
importantly, it did **not** rail at the grid ceiling. That matters: had a pure
momentum push been unable to reproduce a half-time lead at *any* magnitude, the
posterior would have piled up at 1.0 and told us the lever can't express the
instrument. Instead ~0.2 of momentum is *enough* to reproduce the final-result
distribution of narrow-HT-lead matches, and everything above 0.4 is ruled out.

## What was banked

* `calibrate.historical_prior("early_pressure")` → `gaussian(0.205, 0.055)`
  (σ from the 90% half-width), replacing the flat prior.
* `interventions.early_pressure(...)` default magnitude `0.6 → 0.205`, matching
  how `injury` and `red_card` defaults were set to their calibrated means.
* The live ledger refines this posterior one settled read at a time
  (`update_from_ledger("early_pressure")`).

## Caveat carried forward

This is an **upper bound**, not a clean estimate of pure early momentum, because
the only available early instrument (a HT lead) is contaminated by the goal it
contains. A cleaner instrument would need time-split shot/territory data the
football-data CSVs do not provide. Treat 0.205 as "the momentum magnitude that
reproduces a narrow half-time lead," and read it as an anchored prior, not a
law. The live Sandefjord read (0.15) sits just inside the lower CI, so it was a
mildly conservative expression of this lever.

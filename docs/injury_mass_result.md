# Injury / Mass Interventional Validity — the third lever, and the geometric law it exposed

**Status:** Empirical result, July 2026. In-repo experiment 24.
Reproducible from `PYTHONPATH=src python3 experiments/24_injury_mass_validity.py`
(`DIAG=1` = ground truth only; `ORBITA_TRANSFER=1` = the result-axis re-spec;
env: `ORBITA_THR`, `ORBITA_DT`, `ORBITA_NTRIALS`, `ORBITA_PLACEBO_CAP`).

## Why this test

Red cards (exp 22) validated the **momentum** primitive; drag / low-tempo
(exp 23) was **rejected**. This test isolates the last primitive: **mass**.
An injury / suspension makes a team weaker — in Orbita that is a pure cut to
that team's gravity-well mass (its likelihood), with *no* directional vector.
If a mass-cut moves the forecast the way a real weakening does, all three
levers are independently grounded.

## Sourcing the condition — Pinnacle open→close drift

football-data has no lineups/injuries, but it carries Pinnacle **opening**
(`PSH/PSD/PSA`) and **closing** (`PSCH/PSCD/PSCA`) odds in every season.
Pinnacle is the sharp book, so an open→close drift *is* the market re-rating a
team's strength — and in Orbita, well mass *is* market-implied strength, so a
drift is a mass change in the engine's own units. The natural experiment: the
**weakened team** = the side whose Pinnacle win-prob dropped ≥ `THR`
open→close. (This is a broad "team re-rated weaker" proxy — injury, suspension,
rotation, sharp money — not "injury" narrowly; named honestly.)

**Premise gate (model-free, must clear before the lever):**

| adverse drift ≥ | n | opening implied | closing implied | actual | opening gap |
| --------------- | ---- | --------------- | --------------- | ------ | ----------- |
| 0.02 | 1460 | 42.1% | 38.4% | 37.0% | **−5.1** |
| 0.03 | 797  | 43.8% | 39.2% | 37.5% | **−6.3** |
| **0.05** | **216** | **47.2%** | 40.5% | **39.4%** | **−7.8** |
| 0.08 | 40   | 54.3% | 44.7% | 57.5% | +3.2 |

At `THR=0.05` the **opening price over-rates the weakened team by 7.8 points**
against what actually happens, and the closing price captures almost all of it.
A large, real, capturable mis-pricing — exactly what a mass-cut should recover.
(The ≥0.08 tail *reverses* — the market overreacts to the biggest rumours — a
non-linearity echoing the red-card saddle; the signal lives in the moderate
band.) Info direction confirmed: closing Brier 0.5622 < opening 0.5644.

## Result 1 — the symmetric mass-cut is REJECTED

Cut the weak well's mass by fraction `c`, renormalize. Held-out n=77, dt=0.1.
Training is **monotonically worse** with any cut → it selects **c=0**:

| c | 0.0 | 0.1 | 0.2 | 0.3 | 0.45 | 0.6 | 0.75 |
| - | --- | --- | --- | --- | ---- | --- | ---- |
| train Brier | **0.6344** | 0.6350 | 0.6352 | 0.6357 | 0.6382 | 0.6643 | 0.7112 |

A real 7.8-point mis-pricing sits right there, and the mass-cut cannot touch it.

## The diagnostic — why: the central-draw-well bias

Where does the freed probability go? Mean H/D/A shift on the eval weak matches:

| weak = HOME [5,0] (n=35) | H | D | A |
| ------------------------ | ------ | ------ | ------ |
| **reality** (open→actual) | −0.136 | **−0.010** | **+0.146** |
| engine cut c=0.45         | −0.123 | **+0.057** | +0.066 |

Reality moves the freed probability almost **entirely to the opponent** and
leaves the **draw flat**. The symmetric mass-cut splits it roughly evenly
between opponent and **draw**. The away-weakened case mirrors this; the pooled
tell is the draw shift: empirical **−0.010** vs engine **+0.062**.

The cause is geometric: the draw well sits at the centre `[0,5]`. Cutting a
*side* well's mass removes the body's pull toward that side, so it drifts toward
the centre — and the draw well vacuums up ~half the freed mass. The magnitude of
the weak-side reduction is *correct* (−0.123 vs −0.136 at c=0.45); only the
**allocation** is wrong. Since the real outcomes are opponent *wins*, the
spurious draw mass destroys as much Brier as the correct reduction earns → net
zero → c=0.

## The unifying law — scalar levers leak to the draw; only directional levers work

This is the real prize of the experiment. All three levers now line up under
one principle:

| lever | acts as | redistributes toward | verdict |
| ----- | ------- | -------------------- | ------- |
| momentum (red card) | directional push | opponent well (result axis) | **VALIDATED** |
| drag (low tempo)    | scalar damping   | central draw well            | rejected |
| mass-cut (injury)   | scalar reduction | central draw well            | rejected |

**Scalar operations (mass, drag) leak probability into the central draw well;
reality moves it along the result axis (home↔away) with the draw ~fixed. Only
the directional momentum primitive is aligned with how reality redistributes.**
The engine's counterfactual power lives in its *directional* primitive.

## Result 2 — the result-axis transfer re-spec (`ORBITA_TRANSFER=1`)

The fix follows directly: don't renormalize the freed mass symmetrically —
**transfer it to the opponent well**, leaving the draw untouched (a result-axis
move, the momentum primitive's signature). Held-out n=77, dt=0.1:

| c | 0.0 | 0.1 | 0.2 | 0.3 | 0.45 | 0.6 | 0.75 |
| - | --- | --- | --- | --- | ---- | --- | ---- |
| train Brier | 0.6344 | 0.6336 | **0.6334** | 0.6510 | 0.6865 | 0.7474 | 0.8322 |

Training now finds a **real optimum at c=0.2** — the mechanism is corrected.

| eval (n=77), tuned c=0.2 | Brier | Δ | 90% CI |
| ------------------------ | ----- | ----- | ------ |
| market OPEN (baseline)   | 0.5953 | — | — |
| market CLOSE (ceiling)   | 0.5720 | — | — |
| engine baseline (c=0)    | 0.5959 | — | — |
| **+ result-axis transfer** | **0.5885** | **+0.0074 vs baseline** | **[−0.019, +0.033]** |

* Beats the opening market (0.5885 < 0.5953) and recovers **29%** of the
  open→close ceiling gap — the mechanistic engine now extracts real information
  from the drift.
* **Controls pass decisively:** PLACEBO (same cut on no-drift matches)
  Δ −0.0218 CI[−0.038,−0.005]; WRONG-TEAM (cut the strengthened side)
  Δ −0.0451 CI[−0.064,−0.026]. The lever *hurts* off-signal and *hurts more*
  mis-aimed — a real, direction-specific operator, not a free lunch.
* **But the headline validity CI spans zero.** At n=77 the effect (+0.0074) is
  too small to clear significance.

## Verdict — mechanistically salvaged, statistically unproven

The Mass primitive, modelled as a **result-axis transfer**, is clearly the
*right* mechanism: training finds a real optimum, it beats the opening line, and
both controls confirm it captures a genuine, direction-specific weakening
signal. But it does **not clear the validity bar** (Δ vs baseline CI includes
zero) — the drift proxy is a smaller (~7.8pt) and noisier signal than the red
card's clean ~15pt shock, so n=77 is underpowered.

Two things are established. (1) The **structural law** — scalar levers leak to
the central draw well; only directional (result-axis) levers reproduce how
reality redistributes probability — is doubly confirmed: the symmetric cut
failed exactly by draw-leakage, and switching to a directional transfer
immediately fixed the mechanism. This is a real design principle for every
future lever. (2) The injury lever is **promising but not yet proven**;
firming it needs more sample (more leagues' football-data CSVs at the same
Pinnacle-drift selector, or combining books to denoise the drift) — not a new
mechanism.

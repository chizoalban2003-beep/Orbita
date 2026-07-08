# State Inertia (per-match dispersion) on pre-match O/U 2.5

**Status:** Empirical result, July 2026. In-repo experiment 20.
Reproducible from `PYTHONPATH=src python3 experiments/20_state_inertia.py`.

## The hypothesis

Bookmakers price totals on Poisson/NegBin assumptions with a fixed
variance-to-mean law. Matches differ in *structural rigidity* — two
controlled sides produce a low-variance total; two chaotic transitional
sides an overdispersed one. If Orbita modulates the volatility of the
match state per match by a structural proxy, it could price dispersion
the market underprices, closing the pre-match Brier gap.

## Honest physics note

In Orbita gravitational *acceleration* is mass-independent
(`F_grav = m·g ⇒ a = g`; the equivalence principle). A literal body mass
therefore does **not** make the state "resist gravity" — it only touches
drag (`a_drag = −(C_d/m)v`) and `q̇ = p/m`. So we test the idea two ways:

* **(A) literal body mass** `m` — global sweep.
* **(B) per-match dispersion** — scale the Monte-Carlo IC spread per match
  by a leakage-free rigidity proxy (the engine's genuine variance knob):
  `disp = exp(s·z_rigidity)`.

Rigidity proxy: each team's EXPANDING variance of match total goals
(prior games only, `MIN_GAMES=4`), summed over the two sides, z-scored on
TRAIN statistics only. High = volatile. Train the knob on the season's
first 60%, evaluate out-of-sample on the last 40%.

## Result

Neither knob produces a reliable out-of-sample edge, and the engine still
trails the market on O/U 2.5. N = 32 MC trials/match, held-out n = 152.

| line (held-out)                | O/U 2.5 Brier | Δ vs baseline | 90% CI            |
| ------------------------------ | ------------- | ------------- | ----------------- |
| market (B365 closing)          | **0.5096**    | —             | —                 |
| engine baseline (m=1, disp=1)  | 0.5277        | —             | —                 |
| (A) trained mass m=1.5         | 0.5233        | +0.0044       | [−0.0123, +0.0214] |
| (B) trained dispersion s=+0.30 | 0.5292        | −0.0015       | [−0.0126, +0.0091] |

Both CIs span zero → **INCONCLUSIVE**. Training nudged both knobs in the
direction the overdispersion story predicts (mass slightly > 1; volatile
teams → wider spread), and (A) is faintly positive on both train and
eval — but the effect never clears noise on 152 matches, and (B) failed
to transfer from train to eval. The engine does not beat the market.

## Why the test is under-powered (the real lesson)

Three things blunt this test, and the fix defines the next experiment:

1. **The 2.5 line is the worst place to look for dispersion.** O/U 2.5
   sits near the median total, so it is the line *least* sensitive to
   variance — dispersion mispricing shows up in the **tails** (over 3.5 /
   under 1.5), in **correct-score**, and in **BTTS**. Testing
   overdispersion on the mean-line is looking for the effect where it is
   smallest.
2. **One season of expanding goal-variance is a noisy proxy.** With
   `MIN_GAMES=4`, early-season matches get a neutral proxy (188/228 train
   coverage) and even covered teams have a high-variance variance
   estimate. Stable structural ratings need multiple seasons.
3. **Train→eval regime shift.** Baseline O/U Brier was 0.468 on train vs
   0.528 on the held-out tail of the season — the periods differ, so a
   knob tuned on one does not cleanly transfer.

## Next test (before reviving the idea)

Test dispersion where it bites: **BTTS and the tail O/U lines
(1.5 / 3.5)**, over **multiple seasons** for a stable per-team variance
proxy. If a rigidity-driven dispersion knob cannot beat the market on the
tails, the overdispersion edge does not exist at this data resolution and
the engine should be treated as at market efficiency on totals.

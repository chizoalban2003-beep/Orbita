# Orbita

**A mechanistic predictive engine for probabilistic events.**

Orbita models uncertain outcomes as gravity wells in an event space. Live state
orbits through that space. Intangibles act as drag. Predictions fall out of the
trajectory.

> **This is not a deep learning model.** There are no weights to train, no
> millions of opaque parameters, no GPUs required. Every quantity Orbita uses
> has a physical meaning you can name.

---

## What problem does this solve?

Probabilistic forecasting today is dominated by black-box models: gradient-
boosted trees, transformers, large neural ensembles. They are accurate when
the training distribution matches reality and inscrutable when it doesn't.
When a model trained on 10 years of football data misses a match, nobody
can say *why*.

Orbita takes the opposite approach. The model is a small physics simulation
whose parameters are interpretable:

| Concept            | Physical analogue       | What it means                          |
| ------------------ | ----------------------- | -------------------------------------- |
| Outcome            | Gravitational attractor | A possible final state of the event    |
| Outcome likelihood | Mass of the attractor   | Larger mass = deeper well = more likely|
| Outcome magnitude  | Depth of the well       | A 4–0 win is a deeper well than 1–0    |
| Live event state   | Orbiting body           | Current position in possibility space  |
| Momentum           | Body velocity           | How fast the state is moving           |
| Intangibles        | Drag force              | Morale, fatigue, crowd, referee bias   |
| Calibration        | Bayesian posterior      | Updates parameters from past events    |

When Orbita predicts a 73% chance of outcome X, you can ask *which forces
produced that number*. Because there are only a handful of forces and they
all have names, the answer is legible.

## How is this different from a neural network?

| Property         | Neural Network         | Orbita                                  |
| ---------------- | ---------------------- | --------------------------------------- |
| Parameters       | Millions of weights    | ~10–50 physical constants               |
| Training         | Gradient descent       | Bayesian posterior over physical params |
| Data hunger      | High (10⁴–10⁹ samples) | Low (10²–10³ events)                    |
| Interpretability | Post-hoc, partial      | Built-in, total                         |
| Generalization   | In-distribution        | Cross-domain (same physics)             |
| Compute          | GPU-bound              | CPU, often real-time                    |
| Failure mode     | Silent, confident      | Visible, structural                     |

See [`docs/why_not_deep_learning.md`](docs/why_not_deep_learning.md) for the
full positioning.

## Architecture

Four subsystems with crisp seams:

1. **Physics Core** — Symplectic Hamiltonian integration. Energy is conserved
   to numerical precision when drag is off.
2. **Drag Ontology** — A typed mapping from real-world intangibles (fatigue,
   morale, crowd hostility) into three drag modes: isotropic, anisotropic,
   stochastic. See [`docs/drag_ontology.md`](docs/drag_ontology.md).
3. **Calibration Engine** — Reactive Bayesian updates over the physical
   parameters after each event concludes.
4. **Telemetry I/O** — WebSocket ingest of live event ticks, state-diff
   emission for downstream visualization clients.

The visualization layer is intentionally outside the core engine, so the
same physics can be rendered as 2D for analysts, 3D for broadcasts, or
served as a raw probability table for downstream consumers.

See [`docs/architecture.md`](docs/architecture.md) for the full breakdown.

## Quickstart

```bash
pip install -e .          # from a local clone
# or, once published:
pip install orbita
```

```python
from orbita import Attractor, EventSpace, Body, simulate, final_well

# Define the event space: three possible outcomes
space = EventSpace([
    Attractor(label="home_win", position=[5.0, 0.0],  mass=0.55),
    Attractor(label="draw",     position=[0.0, 4.0],  mass=0.20),
    Attractor(label="away_win", position=[-5.0, 0.0], mass=0.25),
])

# Simulate a 90-minute event with mild isotropic drag
sol = simulate(space, duration=5400.0, C_d=0.02)
print("Predicted outcome:", final_well(sol, space))
```

## Examples

- [`examples/01_two_well_validation.py`](examples/01_two_well_validation.py) —
  Proves the integrator conserves energy over a 90-minute simulation. Run this
  first; if it fails, nothing downstream is trustworthy.
- [`examples/02_soccer_match.py`](examples/02_soccer_match.py) — Full match
  simulation with realistic intangibles.
- [`examples/03_election_night.py`](examples/03_election_night.py) — Same
  engine, different domain. Two candidate attractors, polling momentum as
  the orbiting body.
- [`examples/04_market_regime.py`](examples/04_market_regime.py) — Bull vs
  bear regime forecast; VIX-derived drag.
- [`examples/05_world_cup_norway_france.py`](examples/05_world_cup_norway_france.py)
  — Public pre-kickoff World Cup 2026 forecast (Norway vs France).
- [`examples/05b_norway_france_ablation.py`](examples/05b_norway_france_ablation.py)
  — Prior-sensitivity ablation isolating priors / momentum / author shift.
- [`examples/06_calibration_review.py`](examples/06_calibration_review.py) —
  Post-match scoring of the Norway/France forecast.
- [`examples/07_multi_sport_backtest.py`](examples/07_multi_sport_backtest.py)
  — The multi-sport backtest harness. Reproduces the v0.2 results below.
- [`examples/08_sensor_layer.py`](examples/08_sensor_layer.py) — In-play
  sensor demo: Norway/France replay with synthetic xG + red card. Outcome
  distribution moves at every event.
- [`examples/09_today_world_cup.py`](examples/09_today_world_cup.py) —
  Daily slate forecast: priors-only run for every fixture on the card.
- [`examples/10_strategy_layer.py`](examples/10_strategy_layer.py) —
  Multi-market forecast + Kelly / confidence-weighted Kelly / EV
  portfolios + the honest read on why raw Kelly EV is a mirage on this
  engine version.

## Backtest: how the engine actually performs

13 verified head-to-head events across 4 sports (5 soccer, 3 NBA Finals,
3 tennis, 2 MMA), de-vigged sportsbook priors as input, scored by Brier
against actual outcomes.

### v0.3.3 (current)

| | bookmaker | engine + templates | engine + roster | engine + α (calibrated) |
| --- | --- | --- | --- | --- |
| tennis (3) | 0.092 | **0.024** | **0.015** | 0.037 |
| mma (2)    | 0.415 | **0.385** | 0.442 | 0.392 |
| soccer (5) | 0.451 | 0.465 | **0.437** | 0.459 |
| nba (3)    | 0.706 | 0.748 | 0.740 | 0.735 |
| **ALL**    | 0.421 | **0.416** | **0.410** | **0.415** |

- **Engine + roster beats bookmaker by 0.011 on aggregate** (0.410 vs 0.421).
- LOOCV mean Brier 0.426 — out-of-sample we tie the market.
- **Fitted `alpha = 0.68`** (was 0.005 in v0.2): the engine now genuinely
  contributes orthogonal signal that the calibration loop wants to use.
- The single biggest win is NBA: Brier 1.209 → 0.748. The Knicks-over-Spurs
  upsets that destroyed v0.2 are still misses, but the engine no longer
  triples down on the favourite.

> **Honest small-sample caveat.** 13 events is too few to claim a robust
> market-beating result. A 50-match EPL backtest with real Bet365
> closing odds (2024/25 Oct–Nov) and ClubElo team-strength ratings as
> roster input shows:
>
> | model | Brier | modal hit-rate |
> | --- | --- | --- |
> | bookmaker | 0.628 | 44% |
> | orbita_priors | 0.631 | 44% |
> | orbita_roster (Elo) | 0.634 | **50%** |
> | orbita_calibrated | 0.628 | 44% |
>
> The engine ties the market on Brier and picks the right winner 6 points
> more often than the market does. But Brier doesn't improve, even with
> a roster_share sweep (0.0 → 1.0 keeps Brier in [0.631, 0.636]). The
> structural reason: **ClubElo team strength is already priced into the
> closing line**, so feeding it back as a roster signal sharpens toward
> the favourite without adding new information. To beat the market on
> Brier we need roster data the market underweights — lineup quality,
> position matchups, recent-form differentials — which is what the rich
> roster API (issue #8) was built for, but requires per-player data
> beyond ClubElo. See
> [`experiments/03_footballdata_backtest.py`](experiments/03_footballdata_backtest.py)
> for the reproducible run.

### v0.3.5 — player attractors + multi-market wells (50-match panel)

The same 50-match panel with two architectural extensions:

1. **Multi-market joint event space.** Instead of 3 H/D/L wells, the engine
   uses 6 joint wells over (H/D/L) × (over 2.5 / under 2.5 goals). The match
   posterior factors into a market posterior and an O/U marginal.
2. **Player attractors (Option A).** 22 synthetic lineup attractors
   (4-3-3 per side, mass ∝ ClubElo team strength + per-player noise) join the
   force field but are excluded from the posterior — they perturb the orbit
   without competing for probability mass. The invariant is that
   `force_space` includes players, `outcome_space` does not.

| config | H/D/L Brier | H/D/L hit | O/U Brier | O/U hit |
| --- | --- | --- | --- | --- |
| bookmaker            | **0.628** | 44% | **0.459** | 58% |
| A_baseline (3 wells) | 0.682 | 38% | — | — |
| B_multimarket (6 wells) | 0.667 | 42% | 0.475 | **62%** |
| C_multi + players    | 0.652 | **46%** | 0.479 | 60% |

- **Engine modal H/D/L hit-rate beats the market for the first time on a
  50-match sample.** Config C picks the right winner 46% of the time vs the
  bookmaker's 44%. The 22-attractor lineup forces the body off the
  market-implied basin in upset-prone matches.
- **O/U modal hit-rate also beats the market.** Config B gets 62% vs market's
  58% — same direction of travel on a different market.
- **Brier still favours the market on H/D/L, but the gap closes monotonically**
  as structure is added: +0.053 (baseline) → +0.038 (multi-market) → +0.024
  (with players).
- **O/U Brier at N=100 is statistically tied with the market on this 50-match
  window** — experiment 07 reruns Config B at 100 trials/match and finds
  engine Brier 0.4524 vs market 0.4587, point estimate −0.006 in the
  engine's favour, bootstrap 90% CI [−0.036, +0.024] including zero.

### v0.3.5 — full-season reality check (380 matches)

Experiment 08 scales the panel to **every 2024/25 EPL fixture** (380
matches, B365 closing odds). The Oct–Nov "tie on O/U" does **not** hold:

| config | HDA Brier | HDA hit | O/U Brier | O/U hit |
| --- | --- | --- | --- | --- |
| bookmaker        | **0.579** | **54%** | **0.484** | 57% |
| A_baseline (3w)  | 0.608 | 46% | — | — |
| B_multimarket    | 0.623 | 45% | 0.532 | **58%** |

Bootstrap 90% CI on per-match (engine − market) deltas:

| config | HDA Δ | HDA CI | O/U Δ | O/U CI |
| --- | --- | --- | --- | --- |
| A_baseline    | +0.029 | [+0.006, +0.053] | — | — |
| B_multimarket | +0.044 | [+0.025, +0.065] | +0.048 | [+0.025, +0.071] |

**Both deltas have CIs strictly excluding zero — the closing line wins
on Brier with statistical significance on a full season.** The 50-match
Oct–Nov panel was a non-representative window. The only metric the engine
still wins on is O/U modal hit-rate (58% vs 57%), which is too close to
call.

**Read this honestly:** an efficient closing line is a high bar, and the
current configuration — vanilla Hamiltonian wells + ClubElo team strength
+ synthetic per-player lineups — does not clear it on a full EPL season.
The architectural extensions (multi-market joint wells, player
attractors) are sound; the data inputs are not informative enough beyond
what's already priced into the line. **The pitch is interpretability and
mechanism, not market-beating accuracy. Don't claim what we can't show.**

> **Update:** anisotropic drag (`C_d_x=0.00, C_d_y=0.16`) halves both
> losses on the same panel — HDA +0.044 → +0.025, O/U +0.048 → +0.020.
> Market still wins with statistical significance, but the gap is
> closing from geometry alone, before any input-data upgrades. See the
> anisotropic-drag section below.

Reproduce with
[`experiments/05_player_attractor_panel.py`](experiments/05_player_attractor_panel.py)
(50-match panel),
[`experiments/06_alpha_blend_sweep.py`](experiments/06_alpha_blend_sweep.py)
(blend),
[`experiments/07_ou_robustness.py`](experiments/07_ou_robustness.py)
(50-match bootstrap CI + LOOCV),
[`experiments/08_full_season_panel.py`](experiments/08_full_season_panel.py)
(full 380-match season). Single-match teardown at
[`experiments/04_player_attractor_prototype.py`](experiments/04_player_attractor_prototype.py).

### v0.3.5 — niche markets (corners, cards, BTTS)

football-data.co.uk doesn't publish closing odds for corners/cards/BTTS,
so experiment 09 scores three statistical baselines and the engine
against actuals on the full 2024/25 EPL season:

| market         | uniform | league_poisson | team_rolling (K=5) | engine |
| -------------- | ------: | -------------: | -----------------: | -----: |
| corners > 9.5  | 0.500 | **0.482** | 0.587 | 0.629 |
| cards > 4.5    | 0.500 | **0.483** | 0.519 | 0.522 |
| BTTS           | 0.500 | **0.489** | 0.562 | 0.599 |

Bootstrap 90% CI of (engine − team_rolling) per-match deltas:

* corners: +0.042 [+0.024, +0.060]   — team_rolling wins
* cards:   +0.002 [−0.017, +0.020]   — tied
* btts:    +0.037 [+0.016, +0.058]   — team_rolling wins

**Two counter-results worth surfacing:**

1. **The engine adds no signal on derivative markets.** A 2-well event
   space initialised with the rolling-Poisson prior and a small Elo skew
   produces, on average, a *noisier* version of its own prior. Without
   richer per-match inputs (lineups, in-play observations, possession
   style), the simulation has nothing physical to base a non-trivial
   re-weighting on.
2. **league_poisson — which uses no per-team information at all — beats
   team_rolling on every market.** Rolling K=5 over-fits short-term
   variance; shrinkage toward the league mean is a meaningful win on
   these noisy small-count outcomes.

Reproduce with
[`experiments/09_niche_markets_panel.py`](experiments/09_niche_markets_panel.py).

### v0.3.5 — anisotropic drag (halves the full-season loss)

The H/D/L event-space has a directional asymmetry that isotropic drag
ignores:

* **x-axis** = home/away win axis. Momentum here is directional
  pressure — a side that's pressing.
* **y-axis** = draw / goals axis. Drift here is not directional; it's
  the body being pulled into the central well.

Experiment 10 sweeps `C_d = (C_d_x, C_d_y)` on a 5×5 grid over the
50-match panel; experiment 11 tests the sweep's winner out-of-sample on
the full 380-match season.

Full-season 380-match Brier vs market (Config B, engine − market;
bootstrap 90% CI, N=30 trials/match):

| drag | HDA Δ | HDA CI | O/U Δ | O/U CI |
| --- | --- | --- | --- | --- |
| isotropic (0.04) | +0.044 | [+0.025, +0.065] | +0.048 | [+0.025, +0.071] |
| **anisotropic (0.00, 0.16)** | **+0.025** | [+0.008, +0.043] | **+0.020** | [+0.007, +0.033] |

**Anisotropic drag ~halves the market's Brier lead on both markets.**
Zero x-drag preserves directional momentum through win-side wells;
heavy y-drag dampens noise on the goals axis. Both CIs still exclude
zero on the market's side — the closing line still wins — but the gap
closed from +0.048 to +0.020 on O/U without changing a single input,
just the drag geometry.

The 50-match sweep found −0.010 as the O/U point estimate for
(0.00, 0.16); on 380 matches out-of-sample the honest number is
+0.020. The 50-match win was noise-inflated but the *direction* of the
effect is real and holds under statistical scrutiny.

Reproduce with
[`experiments/10_anisotropic_drag_sweep.py`](experiments/10_anisotropic_drag_sweep.py)
(sweep) and
[`experiments/11_anisotropic_full_season.py`](experiments/11_anisotropic_full_season.py)
(out-of-sample verdict). Standalone write-up:
[`docs/anisotropic_drag_result.md`](docs/anisotropic_drag_result.md).

### v0.3.6 — engine capabilities to push the anisotropic result further

Three infrastructure additions unlock the next round of experiments
without disturbing the 2D Config B baseline:

1. **n-D event space.** `Attractor.position` and the whole force /
   integrator stack now accept any dimension ≥ 2.
   [`examples/11_three_axis_event_space.py`](examples/11_three_axis_event_space.py)
   ships a 3D H/D/A × O/U × BTTS event-space (12 wells) as validation.
2. **Time-varying drag schedules.** `linear_ramp_schedule` and
   `piecewise_constant_schedule` produce callable `C_d(t)` that
   `drag_force` and the integrator evaluate per step. Fatigue and
   desperation are non-uniform in match minutes; drag should be too.
3. **Stochastic drag (Ornstein-Uhlenbeck).** `ornstein_uhlenbeck_schedule`
   returns a per-trial noise realisation around a mean coefficient.
   Each MC trial samples a different drag path — the posterior is
   naturally hedged, and calibrated confidence intervals become
   emitable for the first time. Breaks symplecticity but the leapfrog
   structure on the deterministic part remains 2nd-order.

Follow-up experiments (results below where run):

* [`experiments/12_time_varying_drag_sweep.py`](experiments/12_time_varying_drag_sweep.py)
  — 4×4 grid on linear-ramp y-drag `(Cy_start, Cy_end)` on the
  50-match panel. Partial verdict below.
* [`experiments/13_stochastic_drag.py`](experiments/13_stochastic_drag.py)
  — constant vs low- vs high-noise OU around `Cy = 0.16`. Verdict
  below.
* [`experiments/14_momentum_ic_features.py`](experiments/14_momentum_ic_features.py)
  — bias `p0` with pre-match rest-days and rolling-form
  differentials on the full 380-match season. Verdict below.
* [`experiments/15_per_team_drag_calibration.py`](experiments/15_per_team_drag_calibration.py)
  — fit per-team y-drag on first-half goal intensity, evaluate on
  held-out half.

**Experiment 13 — stochastic drag verdict (50-match, bootstrap 90% CI):**

| Config | HDA delta | HDA CI | O/U delta | O/U CI |
| --- | --- | --- | --- | --- |
| constant (0.00, 0.16)       | +0.037 | [−0.016, +0.090] | −0.011 | [−0.036, +0.017] |
| OU low-noise (σ=0.05)       | +0.063 | [−0.020, +0.144] | +0.016 | [−0.027, +0.061] |
| OU high-noise (σ=0.15)      | +0.163 | [+0.050, +0.273] | +0.010 | [−0.025, +0.047] |

**Stochastic drag does not help.** OU noise dilutes the deterministic
signal without adding predictive information. High-noise significantly
degrades HDA (CI excludes zero on the market's side). Deterministic
anisotropic remains the champion drag setting. **Honest negative result.**

**Experiment 12 — time-varying drag partial verdict (HDA rows 1-2 only,
50-match panel):**

Rows 3-4 not completed due to compute constraints on the shared box;
the partial pattern already answers the question.

|                    | Cy_end=0.00 | Cy_end=0.08 | Cy_end=0.16 | Cy_end=0.32 |
| ------------------ | ----------- | ----------- | ----------- | ----------- |
| Cy_start=0.00      | −0.0032     | +0.0408     | +0.0493     | +0.0037     |
| Cy_start=0.08      | +0.0229     | +0.0299     | +0.0281     | +0.0282     |

The (0.00, 0.00) constant-zero-drag reproduces experiment 10's HDA
winner (−0.0032). Every non-trivial ramp-up degrades HDA back toward
the market delta by +0.02 to +0.05. **The physical intuition — that
fatigue-driven drag should ramp up over match minutes — is not borne
out by Brier on this panel.** The constant drag setting from exp 10/11
outperforms every ramp variant tested.

**Experiment 15 — per-team drag calibration verdict (train/test split,
190/190, bootstrap 90% CI on TEST):**

Per-team `C_d_y` fitted from the first-half goal intensity of each
team (Wolves 3.84 goals/match → 0.08; Everton 2.17 → 0.21) then
evaluated on the second-half 190 matches. Match drag =
`0.5 * (home_C_d_y + away_C_d_y)`.

| Config on TEST | HDA delta | HDA CI | O/U delta | O/U CI |
| --- | --- | --- | --- | --- |
| constant (0.00, 0.16) | +0.031 | [+0.008, +0.054] | +0.025 | [+0.008, +0.043] |
| per-team calibrated   | +0.035 | [+0.014, +0.056] | +0.019 | [+0.003, +0.035] |

Per-team calibration moves HDA +0.004 worse and O/U −0.007 better
against baseline — both deltas well inside the ~0.02-wide CI overlap.
**Team-specific drag from goal intensity is not a real signal at this
resolution.** Both configs lose to market on this half (harder than
the full 380 in exp 11 — market gets sharper as the season progresses),
but the per-team layer adds no lift over constant. **Honest negative
result.**

**Experiment 14 — biased p0 from pre-match features verdict (full
380-match season, bootstrap 90% CI):**

Same anisotropic drag `(0.00, 0.16)` as baseline. `p0` biased with
`LAMBDA_X * (form_adv + 0.2*rest_adv) + LAMBDA_Y * (goals_recent - 2.7)`,
where form_adv is the K=5 rolling goal-differential differential
and rest_adv is the rest-days differential.

| Config on full 380 | HDA delta | HDA CI | O/U delta | O/U CI |
| --- | --- | --- | --- | --- |
| unbiased p0 (baseline)   | +0.025 | [+0.008, +0.043] | +0.020 | [+0.008, +0.033] |
| biased p0 (form + rest + goal intensity) | +0.025 | [+0.008, +0.041] | +0.020 | [+0.007, +0.033] |

**Bias moves both deltas by less than 0.001 — complete CI overlap.**
Pre-match rest days, rolling K=5 form, and recent goal intensity —
the three cheap public features football-data.co.uk ships — are
already priced into the closing line. Biasing the momentum IC with
them adds no material signal on either market. **Honest negative
result.**

**Combined read from exps 10, 11, 12, 13, 14, 15:** among drag AND
initial-condition interventions, the constant anisotropic geometry
(`C_d_x=0.00, C_d_y=0.16`) with unbiased `p0` is the strongest
mechanistic configuration we can reach with public pre-match inputs.
Time-varying drag (exp 12), stochastic drag (exp 13), per-team drag
calibration (exp 15), and biased-p0 from rest/form/intensity (exp 14)
all fail to extend the effect. **The remaining path to closing the
gap on Brier is data the market underweights — real per-player
ratings, injury/lineup information, in-play sensor data — not further
re-parametrisation of the mechanistic core.** The engine's value on
public pre-match data is interpretability and honest posterior
uncertainty, not accuracy beyond the closing line.

### v0.3.7 — participation angles: calibration, blend, in-play kick

After the v0.3.6 negative catalogue, three new investigations widen
the scope from "beat market on Brier" to how the engine can
*participate* alongside the market:

1. **[Exp 16](experiments/16_reliability_calibration.py) — reliability
   & ECE audit.** Are engine posteriors *calibrated* even if not
   sharper? Bin predictions into deciles, compute observed frequency
   per bin, report Expected & Maximum Calibration Error against the
   closing line on the full 380-match season.
2. **[Exp 17](experiments/17_engine_market_blend.py) — engine ⊕
   market alpha-blend on full season.** Post-hoc convex blend
   `p_blend = α·p_engine + (1-α)·p_market`. If the Brier surface has
   an interior minimum, engine adds *orthogonal* signal — the blend
   beats the market even if the engine alone does not.
3. **[Example 12](examples/12_inplay_kick.py) + in-play kernel.**
   `orbita.simulate_from_state` and `orbita.kick` let a body be
   perturbed mid-trajectory by an in-play event (goal, red card,
   penalty). Structurally, this is where mechanism has an
   architectural advantage over static features. Ships with
   `tests/test_inplay_kick.py`.

**Experiment 16 — reliability + ECE verdict (bootstrap 90% CI on
delta ECE):**

| Market | ECE engine | ECE market | delta | CI | Verdict |
| --- | --- | --- | --- | --- | --- |
| HDA | 0.0548 | 0.0158 | +0.0262 | [+0.0039, +0.0479] | market better calibrated |
| O/U | 0.0950 | 0.0266 | +0.0520 | [+0.0269, +0.0709] | market better calibrated |

Even on calibration, the market wins on this panel. **But one nuance
saves the engine's face:** on HDA the engine's *maximum* calibration
error (MCE) is 0.104 vs the market's 0.170 — the market has one
badly-calibrated bin (extreme favourites, where public sentiment
under-weights upsets), while the engine's error is more evenly
distributed. Different failure modes, not a strict domination.

**Experiment 17 — alpha-blend verdict (post-hoc, full 380):**

| Market | α* (best) | Brier(α*) | Brier(market) | delta | CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| HDA | 0.12 | 0.5782 | 0.5787 | -0.0004 | [-0.0025, +0.0016] | tied (CI ∋ 0) |
| O/U | 0.00 | 0.4837 | 0.4837 | +0.0000 | — | market alone best |

**Genuinely interesting:** the HDA Brier surface has an *interior*
minimum at `α = 0.12` — the engine contributes 12% orthogonal signal
to the market on HDA. Improvement is inside noise on this panel but
the *shape* of the surface confirms mechanism is not collinear with
market on HDA. If richer inputs (per-player, injury, in-play) raise
the engine's α* toward 0.5, the blend clears CI and becomes a
market-beat. On O/U the surface is monotonic — pure market wins;
engine contributes no O/U signal orthogonal to the closing line.

**In-play kick capability (v0.3.7 engine addition):** the pre-match
posterior is a *state*, not an *answer*. A goal at minute 60 is a
momentum kick applied to the body; the remaining 30 minutes are the
deterministic (or stochastic) evolution of the perturbed state.
`simulate_from_state(q0, p0, ..., t_start=...)` and
`kick(p, dp)` let callers splice a match around any in-play event:

```
sol_pre  = simulate(space, body, duration=200)     # 0 -> 60 min
q60, p60 = sol_pre["q"][-1], sol_pre["p"][-1]
p60_kick = kick(p60, [+1.5, 0.0])                  # goal for home
sol_post = simulate_from_state(space, q60, p60_kick,
                                duration=100, t_start=200)  # 60 -> 90 min
```

Demo in [`examples/12_inplay_kick.py`](examples/12_inplay_kick.py):
starting from Arsenal 55%/25%/20% priors and a slight home-lean
kickoff momentum, an Arsenal goal at 60' shifts the soft posterior
by +0.21 on `arsenal_win` and −0.24 on `draw` in 30 remaining sim
minutes. Same 3-well space, no re-priced attractors — the mechanism
does the work. Ships with `tests/test_inplay_kick.py` (energy
conservation across the splice; kick is a pure momentum shift; kick
toward a well raises that well's posterior).

*Structural angle:* this is where a static regression-plus-features
model has to be re-trained on in-play state; the mechanistic model
is architecturally live. No exp 10-15 result speaks to in-play — but
also no market of in-play odds is available in
football-data.co.uk to score against yet.

**Experiment 18 — 3D joint 12-well vs 2D-independent verdict
(bootstrap 90% CI on match-level Brier deltas, sign = 3D worse):**

| axis | Brier 3D | Brier 2D / null | Δ | 90% CI | verdict |
| --- | --- | --- | --- | --- | --- |
| H/D/A | 0.6198 | 0.6043 | +0.0158 | [-0.0080, +0.0379] | tied (CI ∋ 0) |
| O/U 2.5 | 0.5096 | 0.5040 | +0.0057 | [-0.0009, +0.0124] | tied (CI ∋ 0) |
| BTTS | 0.6412 | 0.4952 (Poisson null) | +0.1462 | [+0.1070, +0.1845] | **null beats 3D** |

Coupling axes through joint priors + shared trajectories does not
extract outcome correlation on 380 matches — the H/D/A and O/U
marginals of the 3D solution are point-estimate-worse than the
2D-independent cache, though CIs cross zero. BTTS is decisive: the
Poisson-independence null (deriving P(BTTS) from λ inferred from
P(over 2.5)) beats the 3D marginal by 0.146 Brier with CI clear of
zero. Two structural reads: (i) the joint mass allocation absorbs
signal that the 2D marginals concentrate cleanly; (ii) BTTS
correlation with total goals is genuinely close to the Poisson
factorisation, so the mechanistic joint has to actively *avoid*
distorting it — which it doesn't. This closes the "does joint
mechanics carry correlation the market can't price" question in the
negative for this event grammar. Cache written to
`experiments/_cache_18_3d_posteriors.csv`.

*Structural read across exp 16-18:* the engine's surviving
market-participation angles are the alpha-blend interior optimum at
α≈0.12 on H/D/A (exp 17), the in-play kick as a live-state posterior
update (v0.3.7 addition), and a lower MCE (0.104 vs 0.170) on H/D/A
than the market (exp 16). No axis where the standalone engine beats
the market on Brier or ECE, on this football-data.co.uk B365 EPL
panel.

The v0.3.3 fix is structural, not parameter-tuned. Two changes:

1. **Sport-specific initial-condition spread (`ic_scale`).** Tennis and MMA
   are favourite-wins-decisively regimes — narrow IC means the engine
   trusts the prior. Soccer and NBA are upset-prone — wide IC means the
   body explores the field instead of funneling into the modal well. The
   choice per sport is defended from mechanics, not back-fit.
2. **Soft Plummer posterior** in place of hard nearest-well classification.
   Each MC trial's final state contributes to every well in proportion to
   the mass-weighted Plummer attraction at that point, so basin-edge
   trials don't get snapped to a single label. See
   `experiments/01_sharpening_triage.py` for the diagnostic that picked
   this fix.

### v0.2 (for comparison)

| | bookmaker | engine + templates | engine + roster | engine + α (calibrated) |
| --- | --- | --- | --- | --- |
| tennis (3) | 0.092 | **0.007** | **0.004** | 0.091 |
| mma (2)    | 0.415 | **0.365** | 0.473 | 0.414 |
| soccer (5) | 0.451 | 0.520 | 0.480 | 0.450 |
| nba (3)    | 0.706 | 1.209 | 1.209 | 0.709 |
| **ALL**    | **0.421** | 0.537 | 0.537 | **0.421** |

The v0.2 engine beat the market on tennis (0.007) and MMA (0.365) but lost
heavily on soccer/NBA. The fitted `alpha` collapsed to ~0.005, meaning the
engine added essentially no orthogonal signal beyond the bookmaker prior.
That diagnostic is exactly what drove the v0.3.3 over-sharpening fix.

The Norway/France pre-kickoff prediction made before the match
(France 4–1, headline Brier 0.482 vs bookmaker 0.240) was an honest
miss — see [`examples/06_calibration_review.py`](examples/06_calibration_review.py)
and the post-mortem on issue #3.

## Roadmap

- [x] Symplectic integrator with energy-conservation gate
- [x] Two-well 2D validation
- [x] Cross-domain examples (finance, elections, World Cup)
- [x] Minimum-viable roster layer (players as mass-modifiers, [#1](https://github.com/chizoalban2003-beep/Orbita/issues/1))
- [x] Sport-specific event-space templates ([#4](https://github.com/chizoalban2003-beep/Orbita/issues/4))
- [x] Alpha sharpening calibration ([#3](https://github.com/chizoalban2003-beep/Orbita/issues/3))
- [x] Multi-sport backtest panel and harness
- [x] **Sensor layer**: in-play observations as Bayesian updates ([#2](https://github.com/chizoalban2003-beep/Orbita/issues/2))
- [x] Multi-market event-spaces (over/under, BTTS, handicap, joint
      distribution) ([#5](https://github.com/chizoalban2003-beep/Orbita/issues/5))
- [x] Saddle-point detection as confidence/hedge signal ([#6](https://github.com/chizoalban2003-beep/Orbita/issues/6))
- [x] Strategy layer (Kelly / confidence-weighted Kelly / flat / +EV /
      explicit hedge) ([#7](https://github.com/chizoalban2003-beep/Orbita/issues/7))
- [x] Expanded roster API: position weighting, form decay, lineup-level
      sensors, pluggable `RatingProvider` (TOML snapshot adapter shipped;
      live FBref/Understat scraper pending follow-up
      [#9](https://github.com/chizoalban2003-beep/Orbita/issues/9))
      ([#8](https://github.com/chizoalban2003-beep/Orbita/issues/8))
- [x] **Sport-specific IC scale + soft Plummer posterior (v0.3.3)** — fixes
      over-sharpening on multi-outcome sports; engine+roster beats the
      bookmaker on aggregate Brier (0.410 vs 0.421) for the first time
- [ ] Drag ontology for soccer (5 intangibles)
- [ ] NumPyro SVI Bayesian calibration loop
- [ ] WebSocket live-tick ingestion
- [ ] Three.js + WebGPU visualization client
- [ ] Paper: *Hamiltonian Inference for Non-Physical Systems*

## Related work

Orbita is one of three projects exploring physics as a computational substrate:

- **[Mycelium](https://github.com/chizoalban2003-beep/Mycelium)** — Physics-
  inspired ML primitives for an autonomous local agent.
- **[PRISM](https://github.com/chizoalban2003-beep/Prism)** — Local-first AI
  assistant with hardware-bridge organs.
- **Orbita** (this repo) — Physics-as-prediction for probabilistic events.

## License

MIT — see [LICENSE](./LICENSE).

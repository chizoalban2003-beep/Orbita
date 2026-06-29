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
- **O/U Brier at N=100 is statistically tied with the market** —
  experiment 07 reruns Config B at 100 trials/match and finds engine
  Brier 0.4524 vs market 0.4587, point estimate −0.006 in the engine's
  favour but bootstrap 90% CI [−0.036, +0.024] **includes zero**. Engine
  wins on 31/50 individual matches by margin. The headline is
  market-tie, not market-beat, on this panel.
- Reproduce with
  [`experiments/05_player_attractor_panel.py`](experiments/05_player_attractor_panel.py)
  (panel),
  [`experiments/06_alpha_blend_sweep.py`](experiments/06_alpha_blend_sweep.py)
  (blend),
  [`experiments/07_ou_robustness.py`](experiments/07_ou_robustness.py)
  (bootstrap CI + LOOCV). Single-match teardown at
  [`experiments/04_player_attractor_prototype.py`](experiments/04_player_attractor_prototype.py).

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

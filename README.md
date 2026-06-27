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

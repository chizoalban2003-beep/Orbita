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

## v0.2 backtest: how the engine actually performs

13 verified head-to-head events across 4 sports (5 soccer, 3 NBA Finals,
3 tennis, 2 MMA), de-vigged sportsbook priors as input, scored by Brier
against actual outcomes.

| | bookmaker | engine + templates | engine + roster | engine + α (calibrated) |
| --- | --- | --- | --- | --- |
| tennis (3) | 0.092 | **0.007** | **0.004** | 0.091 |
| mma (2)    | 0.415 | **0.365** | 0.473 | 0.414 |
| soccer (5) | 0.451 | 0.520 | 0.480 | 0.450 |
| nba (3)    | 0.706 | 1.209 | 1.209 | 0.709 |
| **ALL**    | **0.421** | 0.537 | 0.537 | **0.421** |

Two findings, both important and both honest:

1. **Tennis and MMA: the engine beats the market.** With sport-specific
   templates, priors-only Brier is 0.007 (tennis) and 0.365 (MMA) — better
   than the bookmaker. These are the regimes the v0.2 physics fits: binary
   outcomes, calibrated favourites, decisive endings.

2. **Soccer, NBA, and aggregate: the market wins.** The engine
   systematically over-sharpens its priors, which compounds Brier penalties
   on upsets (Bonfim over Muhammad, Australia over Türkiye, three Knicks–
   Spurs Finals upsets). The sharpening calibration (`alpha`) collapsed to
   ~0.005, meaning *the engine adds essentially no orthogonal signal beyond
   the bookmaker prior on this panel*. The calibrated forecaster ties the
   market only because it *is* the market.

This is the diagnostic the calibration loop was designed to produce. It
tells us the next phase has to be the sensor layer ([issue #2](https://github.com/chizoalban2003-beep/Orbita/issues/2)):
in-play observations that update the field with information the kickoff
prior doesn't reflect. Better priors and better geometry have now both
been tried and capped.

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
- [ ] **Sensor layer**: in-play observations as Bayesian updates ([#2](https://github.com/chizoalban2003-beep/Orbita/issues/2))
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

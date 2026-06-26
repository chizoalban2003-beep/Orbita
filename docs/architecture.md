# Architecture

Orbita has four subsystems with crisp seams. Each can evolve independently
because the seams are typed Julia interfaces, not network calls (yet).

```
┌────────────────────────────────────────────────────────────────────┐
│                         Telemetry I/O                              │
│   (WebSocket ingest of live event ticks; state-diff emission out)  │
└────────────────────────────────────────────────────────────────────┘
                              │  ▲
                              ▼  │
┌────────────────────────────────────────────────────────────────────┐
│                       Drag Ontology                                │
│   Intangibles (fatigue, morale, crowd, referee) →                  │
│       (C_iso(t), C_aniso(t, k), σ(t))                              │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Physics Core                                 │
│   Symplectic integration of H(q,p) over the EventSpace             │
│   (hand-rolled velocity-Verlet for v0.1; Diffrax LeapfrogMidpoint  │
│    for long-horizon precision in Phase 2)                                     │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Calibration Engine                             │
│   Reactive Bayesian updates over masses & drag weights after       │
│   each concluded event (NumPyro in Phase 2)                     │
└────────────────────────────────────────────────────────────────────┘
```

## Subsystem 1 — Physics Core (`src/orbita/integrator.py`, `src/orbita/forces.py`)

Solves the Hamiltonian system

    H(q,p) = |p|² / (2m) + m · Φ(q)

where `Φ(q)` is the sum of softened Newtonian potentials from each
attractor. Drag (`drag_force` in `forces.py`) is added as a
non-conservative term on the momentum equation; this dissipates
energy in a controlled way.

**Hard requirement:** with drag off, the integrator must conserve `H`
to numerical precision (< 1e-2 relative drift over 5400 seconds for
velocity-Verlet at dt=0.1). This is enforced by
`examples/01_two_well_validation.py` and
`tests/test_energy_conservation.py`. **Do not merge changes that
break this gate.**

## Subsystem 2 — Drag Ontology (`src/orbita/ontology/`, Phase 2)

Maps real-world intangibles to a structured drag vector. Three modes:

| Mode         | Form                                              | Example          |
| ------------ | ------------------------------------------------- | ---------------- |
| isotropic    | `F_d = -C_iso(t) · v`                             | fatigue, crowd   |
| anisotropic  | `F_d = -C_aniso(v̂ · r̂_k) · v` for attractor k    | morale collapse  |
| stochastic   | `dF_d = σ(t) dW`                                  | referee variance |

Stochastic drag breaks symplecticity → SDE solvers required when enabled.
See `docs/drag_ontology.md` for the full catalog.

## Subsystem 3 — Calibration Engine (`src/orbita/calibration/`, Phase 2)

After each event concludes:

1. Replay the event with the actual stream → predicted trajectory.
2. Observe the actual outcome (which well, score margin).
3. Posterior update over `(α_i, β_jk)`:
   `P(α,β | outcome) ∝ P(outcome | α,β) P(α,β)`.

NumPyro's `SVI` does this as a streaming variational update — no MCMC restart.

## Subsystem 4 — Telemetry I/O (`src/orbita/telemetry/`, Phase 2)

- **Ingest**: WebSocket consumer of live event ticks (a play, a goal,
  a possession change). Each tick produces a force kick on the body
  (`Δp`) plus optional updates to drag coefficients.
- **Emit**: state-diff stream (`q`, `p`, depth-of-each-well) for any
  downstream visualization client.

The frontend is deliberately outside the core engine. The same physics
state can be rendered as 2D for analysts, 3D for broadcast, or returned
as raw probability tables.

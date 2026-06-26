# `web/` — Visualization client (Phase 2)

The frontend lives here once Phase 1 is locked in.

Planned stack:
- **Three.js + WebGPU** for the warped-spacetime renderer (gravity wells as
  deformed mesh, orbits as worldlines).
- **WebSocket client** consuming state-diffs from the Julia backend's
  `src/telemetry/emit.jl`.
- **React** wrapper for the analyst UI; raw Three.js for the broadcast
  embed.

Do not start this until `examples/01_two_well_validation.jl` passes
reproducibly in CI. Pretty visualizations of a numerically-broken
physics core are worse than no visualization at all.

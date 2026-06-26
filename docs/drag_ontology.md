# Drag Ontology

This is the design doc that turns intangibles into numbers the integrator
consumes. Drag isn't one scalar — it's a decomposition into three physically
distinct modes, each with its own data source and time profile.

## Three modes of drag

| Mode         | Physical analogue   | When to use it                                            | Form                                              |
| ------------ | ------------------- | --------------------------------------------------------- | ------------------------------------------------- |
| Isotropic    | Atmospheric friction| Bleeds energy uniformly (fatigue, generic away-disadvantage) | `F_d = -C_iso(t) · v`                          |
| Anisotropic  | Headwind / tailwind | Resists motion toward specific attractors (morale)         | `F_d = -C_aniso(v̂ · r̂_k) · v`                  |
| Stochastic   | Brownian buffeting  | Adds variance, not bias (referee inconsistency)            | `dF_d = σ(t) dW`                                  |

Total drag on the body:

    F_drag(t, q, p) = -[ Σ_i α_i · c_i^iso(t) ] · v
                     - Σ_{j,k} β_{jk} · c_j^aniso(v̂ · r̂_k) · v
                     + ξ(t)

The `α_i, β_{jk}` are learnable; the `c_i, c_j` are fixed structural
mappings from raw data to a normalized `[0,1]` coefficient.

## Catalog of intangibles → coefficients (soccer v1)

| Intangible           | Proxy data source                                  | Normalization c(·) | Mode                          | Time profile        |
| -------------------- | -------------------------------------------------- | ------------------ | ----------------------------- | ------------------- |
| Player fatigue       | cumulative distance ÷ team-mean stamina            | linear [0,1]       | isotropic                     | monotonic ramp      |
| Crowd hostility      | home/away flag × stadium dB rating (z-scored)      | min-max            | isotropic                     | constant per match  |
| Squad-quality gap    | starting XI Elo vs. ideal XI                       | sigmoid            | isotropic                     | step on substitution|
| Morale               | rolling goal-diff over last N matches              | tanh               | anisotropic toward Win        | constant per match  |
| Trailing desperation | sign(score gap) · min(1, t/T)                      | linear             | anisotropic toward Loss (-ve) | grows with time     |
| Referee strictness   | card-rate over last 10 games refereed (z-score)    | abs-z              | stochastic (σ)                | constant per match  |
| Pitch conditions     | weather API + surface type (categorical → ordinal) | lookup             | stochastic (σ)                | constant per match  |
| Tactical mismatch    | formation-pair historical win-rate delta           | sigmoid            | anisotropic                   | constant per match  |

## How the calibration loop learns the weights

Each `α_i, β_{jk}` starts with a weakly-informative prior, e.g. `N(0, 1)`.
After every match:

1. Replay the match with the actual event stream → compute predicted trajectory.
2. Observe the actual outcome (which well, score margin).
3. Posterior update via streaming variational Bayes in NumPyro.

## Three commitments buried in the table

1. **Anisotropy is not symmetric.** "Morale-driven drag toward the Win well"
   is *not* the same as "morale boost toward the Win well." The sign of `β`
   encodes which it is. **Negative drag (thrust) is allowed** — desperation
   in a trailing team acts as anti-friction toward the opponent's well.
   Without that, the model can't explain late-game comebacks.

2. **Stochastic drag breaks symplecticity.** Once `ξ(t)` is on, the system
   is an SDE, not an ODE. Use Diffrax's SDE solvers (`UnsafeBrownianPath`
   + `EulerHeun`); expected energy drift is bounded by `σ² · t` rather
   than zero.

3. **The ontology is a v1 assumption, not a truth.** The calibration loop
   only learns the *weights*, not the *structure*. If intangibles interact
   non-linearly (morale × fatigue is multiplicative, not additive), the
   model will hit a precision ceiling. Leave a hook for swapping the
   linear sum for a small MLP later — this is the JAX/HNN escape hatch.

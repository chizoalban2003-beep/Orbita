# Combo-Correlation Edge — real correlation, but no per-match edge

**Status:** Empirical result, July 2026. In-repo experiment 25.
Reproducible from `PYTHONPATH=src python3 experiments/25_combo_correlation.py`
(6 leagues × 6 seasons cached; env `ORBITA_GAMMA`/`NTRIALS`/`DT`/`EVALCAP`).

## Why this test

Single outcomes are market-efficient (exp 22–24). The last market-beating
avenue is **correlation**: a bookmaker prices a combo (Result + Over/Under) as
`P(result) × P(over)` plus a blunt correction, because it multiplies marginal
spreadsheets. Orbita has no marginals — it integrates one shared trajectory
through a joint 6-well plane, so its result and totals axes are coupled
**per-match**. Hypothesis: the per-match coupling carries information a static
correlation matrix cannot.

## Baselines (the combo price is unsourceable offline)

No CSV caches historical combo/bet-builder prices, so both baselines are built
from single-leg **closing** odds (Pinnacle 1X2 + O/U 2.5, seasons ≥1920):

* **tier-1 — independence** `P(r) × P(o)`: the naive product.
* **tier-2 — empirical copula** `tier-1 × lift[r,o]`, one global lift matrix fit
  on train (leakage-free) = "the bookmaker's blunt correlation matrix."

Realized 6-cell outcome comes from the score. Orbita earns the edge only by
beating **both** out-of-sample on 6-cell Brier — beating tier-1 = it makes real
correlation; beating tier-2 = its **per-match** coupling beats a **global** one.

## Premise gate — the correlation is large, and global

Pooled n=13,954. The empirical lift matrix (train):

| | over | under |
| ---- | ---- | ----- |
| home | 1.20 | 0.79 |
| draw | **0.47** | **1.56** |
| away | 1.16 | 0.84 |

Independence mis-prices cells by up to 6pt (Draw∩Under realized 18.7% vs priced
12.6%). Knowing the structure buys **+0.0121 Brier** (tier-1 0.7757 → tier-2
0.7636) — a big ceiling. Real, exploitable-looking structure.

## Deploy Orbita as a correlation LAYER, not a forecaster

A first cut ran Orbita as a from-scratch joint forecast — it **lost to both
tiers** (0.831 vs 0.776/0.764). Cause: a full MC re-forecast distorts the
market's well-calibrated marginals, and that penalty swamps the correlation.
The market owns the marginals (exp 22–24); Orbita's only possible value-add is
the coupling. So the honest deployment extracts Orbita's **per-match lift**
(joint ÷ product of its own marginals) and applies it to the **market**
marginals — exactly parallel to tier-2, but per-match. A single `gamma` scales
the correlation strength the fixed geometry encodes, tuned on train.

## Result — beats independence, loses to the global matrix

Held-out n=1200, marginal-anchored (`orbita_corr`):

| model | 6-cell Brier | vs tier-1 | vs tier-2 |
| ----- | ------------ | --------- | --------- |
| tier-1 independence     | 0.7772 | — | — |
| tier-2 global copula    | 0.7597 | — | — |
| Orbita γ=1 (untuned)    | 0.7791 | Δ −0.0019 (ties) | Δ −0.0194 (loses) |
| **Orbita γ=2.5 (tuned)** | **0.7686** | **Δ +0.0086 [+0.005,+0.012] BEATS** | **Δ −0.0089 [−0.013,−0.005] LOSES** |

* Tuned Orbita **beats naive independence** (CI excludes zero): the shared
  trajectory genuinely manufactures useful, right-signed correlation from
  *independent* priors — a real mechanism result.
* But it **loses to the global copula** (CI excludes zero). Decisively, its
  per-match structure is *worse* than assuming a constant correlation.

## Verdict — the per-match edge does not exist; correlation is global

The hypothesis is falsified. Orbita's correlation is real (beats independence)
but a single global matrix beats it, so the per-match variation the physics adds
carries no signal — the Result×O/U dependency is effectively **match-
independent**, and a static matrix is optimal. This is structural, not a tuning
miss: to lift the draw cell (du 1.23, still short of the empirical 1.56) `gamma`
must overshoot the win cells (hu 0.87 vs 0.79). Orbita is a **one-knob
constrained** correlation model against a **six-number empirical fit**; the only
way it could win is per-match signal, and the eval shows there is none. The
bookmaker's "blunt global matrix" is near-optimal, not a flaw.

**Campaign conclusion.** Across marginals (exp 10–21), single-outcome
interventions (exp 22–24), and now joint correlation (exp 25), the market is
efficient at every level tested — including the dependency structure. Orbita's
enduring value is the **white-box**: it reproduces real football structure
(directional interventions, right-signed per-match correlation) legibly and from
first principles. It is a forecaster you can *interrogate*, not one that beats
the closing line. The remaining honest route to alpha is not a better lever or
market but **human-in-the-loop** — the analyst supplies a private read, and the
interpretable engine prices it into a coherent joint forecast.

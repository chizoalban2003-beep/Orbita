# Dispersion in the Tails — 10-season premise gate

**Status:** Empirical result, July 2026. In-repo experiment 21.
Reproducible from `PYTHONPATH=src python3 experiments/21_dispersion_tails.py`
(caches 10 EPL seasons of football-data.co.uk under
`~/.cache/orbita/footballdata`).

## Why this test

Exp 20 (state inertia) found a per-match dispersion knob can't beat the
market on O/U 2.5 — but 2.5 is the **median line**, the least
variance-sensitive market ("the median trap"). Overdispersion, if real
and exploitable, shows up in the **tails** (Over 3.5, Under 1.5). Before
porting a finer goals-geometry into Orbita, this is a premise gate on the
minimal model: does a leakage-free team-structure proxy let a
dispersion-aware model beat a Poisson extrapolation on realised tail
outcomes, out-of-sample across seasons? If not, no engine geometry can
exploit it.

Data: 2,279 EPL matches, 10 seasons. Baseline = invert the de-vigged B365
O/U 2.5 line to a Poisson mean λ, read the tail probabilities Poisson(λ)
implies (football-data carries no 1.5/3.5 odds, so this is the honest
"market assumes Poisson" null). Train on the first 7 seasons, evaluate on
the last 3. Proxies (expanding, prior-games-only, z-scored on train):

* `goalvar` — Var(prior total goals), the two teams summed.
* `sotxgvar` — Var(prior 0.3·SoT totals); a shots-on-target xG surrogate,
  used because multi-season football-data has **no xG** (the answer to
  "raw goals vs xG variance": xG isn't available, the SoT surrogate is
  the deeper metric that is).
* `fano` — Fano factor Var/Mean, dispersion **isolated from scoring
  level** (Poisson ⇒ 1). Added after the diagnostic below exposed a
  confound.

## Result — the premise fails

**Model-free diagnostic (the decisive part).** Realised tail rates by
proxy quartile:

| quartile | raw `goalvar`: over3.5 / under1.5 / mean | `fano`: over3.5 / under1.5 / mean |
| -------- | ---------------------------------------- | --------------------------------- |
| Q1 low   | 0.297 / **0.233** / 2.76                 | 0.343 / 0.218 / 2.93              |
| Q4 high  | 0.352 / 0.207 / 2.99                     | 0.314 / 0.222 / 2.84              |

Raw goal-variance looks like it separates the tails — but it is a **mean
shift, not dispersion**: high-variance teams are simply higher-scoring
(mean 2.76 → 2.99), so they get *more* overs **and fewer unders**. Genuine
overdispersion would fatten **both** tails. When dispersion is isolated
from the mean via the Fano factor, the pattern **disappears** — tail rates
are flat across quartiles (mean flat at 2.84–2.95). For count data
variance rises with the mean, and the market's O/U line already prices the
mean, so the variance proxy carries no independent tail information.

**Out-of-sample (last 3 seasons, n = 1,140).** Poisson baseline tail Brier
sum (over3.5 + under1.5) = **0.7578**. Every proxy-driven NB model trained
to `s = 0` (i.e. *ignore the proxy*), and the residual global NB
overdispersion made things marginally worse:

| proxy      | trained (r0, s) | eval Brier sum | Δ vs Poisson | verdict |
| ---------- | --------------- | -------------- | ------------ | ------- |
| goalvar    | (20, 0.0)       | 0.7599         | −0.0021      | hurts   |
| sotxgvar   | (20, 0.0)       | 0.7599         | −0.0021      | hurts   |
| fano       | (20, 0.0)       | 0.7599         | −0.0021      | hurts   |

## Verdict

Team-level overdispersion — measured by a team's own historical goal
variance, SoT-xG variance, or Fano factor — **does not predict future tail
outcomes** and offers no edge over a Poisson extrapolation on Over 3.5 /
Under 1.5. The raw-variance "signal" is entirely mean-confounding, which
the market already prices; the de-confounded signal is absent.

Combined with exp 20 (2.5 line) and the anisotropic-drag result (market
wins H/D/A and O/U), the accumulated evidence is that **EPL match totals
are at Poisson/market efficiency to this data's resolution**, including the
tails. The "state inertia / break-the-Poisson-constraint" avenue is
closed for total-goals markets. There is no dispersion signal to port into
Orbita's geometry — which is exactly what this gate was built to find out
*before* building it.

# The Heavy-Favourite Low-Block Draw Gate — FALSIFIED

**Status:** Model-free premise gate, July 2026. In-repo experiment 27.
Reproducible from `PYTHONPATH=src python3 experiments/27_lowblock_gate.py`
(env: `ORBITA_DIVS`, `ORBITA_FAVCUT`, `ORBITA_MINGAMES`, `ORBITA_BOOT`).

## The hypothesis

Spatial/temporal narrative: an extreme **away favourite** meeting a low-scoring,
compact **home underdog** (a deep low block) should produce a **draw-heavy**
distribution that the closing line consistently **fails to price** — an unpriced
draw edge harvestable as a pre-match scenario.

## The test (no engine, no lever)

The honest money metric: the flat-stake **ROI of backing the draw at the Pinnacle
closing price** in each regime, bootstrap 95% CI (3000 resamples). Positive ROI
with a CI clearing zero = a real unpriced edge. 23,527 matches, 6 leagues × 10
seasons, closing devig as the market's fair belief.

## Result — the market prices it; the edge runs the wrong way

```
BASELINE  back the draw everywhere      n=23527  hit 25.4%  impl 25.6%  ROI  -3.4% [ -5.5, -1.2]

DRAW ROI by favourite strength × side (no bucket clears zero):
  HOME fav impl[0.60,0.70)            n= 2262  ROI  -9.1% [-16.3, -1.3]
  AWAY fav impl[0.60,0.70)            n=  867  ROI -12.3% [-24.2, -0.1]
  AWAY fav impl[0.70,0.85)            n=  526  ROI -17.5% [-35.6, +1.0]

USER'S EXACT REGIME — extreme away favourite (away impl ≥ 0.55):
  all extreme away-fav               n= 2140  hit 18.6%  impl 20.7%  ROI -12.5% [-20.4, -4.3]
  LOW-scoring/compact home           n=  940  hit 19.6%  impl 21.3%  ROI -10.4% [-22.1, +1.6]
  higher-scoring home                n=  931  ROI -15.6% [-27.3, -2.9]

CONTROL — favourite-longshot bias:
  back the FAVOURITE (impl≥0.70)     n= 2646  ROI  -1.0% [ -3.0, +1.0]
  back the UNDERDOG  (impl≥0.70)     n= 2646  ROI -13.4% [-25.8, -0.5]
```

**Verdict: FALSIFIED, decisively.** In the exact regime the hypothesis names —
extreme away favourite — backing the draw returns **−12.5% with a CI that
excludes zero on the negative side**. The realised draw rate (18.6%) is actually
*below* the closing-implied draw prob (20.7%): far from *underpricing* draws
there, the market if anything **overprices** them (draws are mildly overbet when
a strong away favourite invites "shock" money). The leakage-free compact-home
split does **not** rescue it — the low-scoring subgroup (−10.4%) is marginally
less bad but its CI still includes zero; the low block carries no unpriced draw
signal.

## What the control confirms

The classic **favourite–longshot bias** is present in the *direction* the
literature predicts — backing underdogs at impl ≥ 0.70 loses heavily (−13.4%,
CI excludes zero: longshots are overbet) — but backing favourites only recovers
to roughly the vig (−1.0%, CI spans zero). So even the one real behavioural
bias in the data is **not tradeable through the closing line** after margin.

## Conclusion

This reconfirms the campaign's through-line: the closing line is efficient at
every macro level we can test — marginals (exp 10–21), single-outcome
interventions (22–24), joint correlation (25), and now favourite-strength ×
spatial-proxy draw regimes (27). A "low-block away-favourite draw edge" is a
plausible-sounding narrative with **no empirical support**; building a
`low_tempo`/positioning lever to harvest it would be fitting an edge that isn't
there. The engine stays frozen. Spatial/temporal factors keep their **only**
honest role: a pre-match scenario the analyst names from *private* information,
priced through the already-validated levers — never a systematic macro bucket
the market has left on the table.

# Hunting Playbook — which narratives to log (and which to skip)

A driving companion for the human-in-the-loop ledger, not engine work. The
principle behind every call below:

> The closing line prices **public** information efficiently, in proportion to
> market liquidity. Edge therefore needs (a) information that is **private or
> early**, (b) a **thinner market** that absorbs it slowly, and (c) a narrative
> that moves the **result axis** — the only axis Orbita repositions. Scalar /
> totals effects are already priced and the engine cannot move them (exp 21/23/25).

**Venue matters as much as the read.** EPL's close is a sponge; Eliteserien,
Allsvenskan and MLS close far softer. The summer leagues are the sandbox not
just for availability but because a real read has room to be *right before the
line catches up*.

## Tier 1 — log these (validated levers · result-axis · soft markets)

| Narrative | Lever | Intensity | Why there's room |
| --- | --- | --- | --- |
| **Key player out** the line under-reacts to (a pivotal creator or defensive anchor, fresh/local team news) | `injury` — mass transfer to the opponent | single key ~0.10–0.15; keeper / multiple ~0.20–0.30 | Validated mechanism (exp 24); local-language news reaches you before a thin market moves |
| **Heavy squad rotation** (MLS congestion, long-haul travel, midweek cups / Leagues Cup) | `injury` toward the opponent (broad weakening) | ~0.15–0.30 by starters rested | MLS is the ideal venue — brutal travel + congested calendar + soft close |
| **Motivation / stakes asymmetry** (a relegation-desperate side vs a coasting one; a European hangover) | `early_pressure` or a mild `injury` transfer | 0.10–0.25 | Judgment the market under-weights; a directional result-axis tilt |
| **Tactical mismatch** that tilts the result (a relentless press vs a shaky build-from-the-back) | `early_pressure` (momentum, validated primitive) | 0.20–0.40 | Your tactical read is the private signal |

## Tier 2 — in-play only

**Red card** → `red_card` (momentum). Orbita's *strongest*, market-beating lever
(exp 22) — but a card is an in-play event. This is the tool's live-repricing use
during a match, not a pre-match log.

## Tier 3 — skip for the first reads (honest)

- **Severe weather / "it'll be a low-scoring scrap."** This acts on the
  **totals** axis, which the campaign proved efficient (exp 21), and its
  result-axis path is the `low_tempo` favourite-lock — still **unvalidated**
  (exp 23 rejected the scalar; the re-spec has no backtest). Tempting, but a weak
  first test that stakes the ledger's early credibility on the one lever we
  haven't earned.
- **Anything leaning on `low_tempo`.** Same reason — validate it with data
  before betting the ledger on it.

## Testing the engine's bounds

- **Watch the saddle.** Log an *extreme* asymmetry and note where your intensity
  crosses the barrier: a moderate read can strand probability in the **draw**
  before a larger read cleanly flips the result (the exp 22 non-monotonicity).
  Feeling that live is the point of manual settling.
- **Conviction over volume.** 5–10 high-conviction reads teach the ledger more
  than 50 marginal ones — signal-per-settled-game is what the eventual JAX loop
  eats, and noise dilutes it.
- **Log the counterfactual you'd defend out loud.** If the one-sentence
  explanation doesn't sound true, don't log it.

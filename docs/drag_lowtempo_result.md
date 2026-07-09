# Drag / Low-Tempo Interventional Validity — the second lever, rejected

**Status:** Empirical result, July 2026. In-repo experiment 23.
Reproducible from `PYTHONPATH=src python3 experiments/23_drag_lowtempo_validity.py`
(`DIAG=1` for the fast ground-truth diagnostic only).

## Why this test

Red cards (exp 22) validated the **momentum** primitive. Drag is the other
lever — a global friction that bleeds kinetic energy and traps the state.
Its claimed signature: low-tempo games are cagey, so the draw firms up. If
interventional validity is a real metric, it must be able to *reject* a
lever, not just rubber-stamp one. This is that test.

Proxy for low-tempo (no possession data): total shots `HS+AS`, bottom
tercile, used as a post-hoc natural-experiment selector (as with red-card
counts). 10 EPL seasons.

## Ground truth — the intuition is wrong

Outcomes by total-shots tercile (3,800 matches):

| tempo | shots | goals | actual draw% | market draw% | fav win% | market fav% |
| ----- | ----- | ----- | ------------ | ------------ | -------- | ----------- |
| low   | 19.7  | 2.43  | 23.6         | 24.5         | **56.3** | 53.4        |
| mid   | 26.0  | 2.88  | 21.3         | 24.2         | 55.7     | 54.2        |
| high  | 32.6  | 3.32  | **25.5**     | 23.1         | 54.1     | 56.3        |

**Low-tempo does not mean more draws** — low-tempo draw rate (23.6%) is
*below* the market's own draw pricing (24.5%). Draws cluster in **high**
tempo, end-to-end games where both sides score (25.5% vs 23.1% implied).
The only real low-tempo signature is a weak **favourite-hold**: the better
side wins +2.9 pts above market in low-tempo games and −2.2 in high-tempo
(a variance collapse toward the prior leader, not toward the draw).

## Result — the lever fails

Raising the drag multiplier `g` monotonically **worsens** low-tempo H/D/A
Brier; training selects `g = 1.0` (no drag change):

| g (train) | Brier  |
| --------- | ------ |
| **1.0**   | **0.6004** |
| 1.5       | 0.6026 |
| 2.0       | 0.6028 |
| 3.0       | 0.6067 |
| 4.5       | 0.6195 |
| 6.0       | 0.6334 |

Held-out (n = 387): tuned `g = 1.0`, so Δ vs baseline = 0 by construction;
Δ vs market = −0.0008, CI [−0.0064, +0.0048]. **INCONCLUSIVE / rejected.**

## Verdict

The drag / low-tempo lever **does not validate**. Two failures compound:

1. As modelled it lifts the **draw** (its unit test asserts exactly that),
   but reality's low-tempo signal is a favourite-hold, not a draw — the
   lever is aimed at the wrong outcome.
2. Even the favourite-hold it *could* capture is weak (+2.9 pts), and
   global drag over-damps: it collapses the posterior onto the top-mass
   well far harder than +2.9 warrants, so every `g > 1` overshoots and
   hurts calibration.

**This is the metric working.** Interventional validity is discriminating,
not a rubber stamp: momentum (red card) passed and beat the market;
drag (low-tempo) is rejected. A lever earns its place only if it moves the
forecast the way reality does — and this one, in its current form, does
not. The `low_tempo` intervention should be treated as **unvalidated**;
reviving it needs a primitive that expresses "favourite locks in" without
the global over-damping (e.g. an anisotropic drag along the result axis
only, or shrinking the IC spread rather than adding friction) — and it must
clear this same ground-truth bar first.

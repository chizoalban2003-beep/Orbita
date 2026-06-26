"""
06_calibration_review.py

Post-match calibration review for the Norway vs France forecast.

After the final whistle, plug the actual outcome into ``ACTUAL_OUTCOME``
below and run this script. It scores Orbita's pre-kickoff distribution
against the bookmaker consensus using:

    - Brier score (lower is better; ~0 = certain & right, 0.5 = coin flip)
    - Log-loss
    - Whether the modal prediction matched reality

The three ablation scenarios from 05b are scored too, so we can see
which configuration of priors + momentum would have been best
calibrated *in hindsight*. This is the instrument that lets the
v0.1 Brier number land on the README as a real, signed benchmark.

Run with::

    PYTHONPATH=src python3 examples/06_calibration_review.py
"""
from __future__ import annotations

from math import log


# ----- fill these in after the final whistle ------------------------------
ACTUAL_OUTCOME: str | None = None   # one of "france_win", "draw", "norway_win"
ACTUAL_SCORE: str | None = None     # e.g. "1-1", "0-2"
NOTES: str = ""                     # anything we want logged for posterity
# --------------------------------------------------------------------------


# Frozen forecast snapshot — copied verbatim from the runs on 2026-06-26.
# We do NOT recompute these here; the whole point of a post-mortem is to
# score the prediction that was actually published.
ORBITA_HEADLINE = {        # examples/05_world_cup_norway_france.py
    "france_win": 0.433,
    "draw":       0.283,
    "norway_win": 0.283,
}
ORBITA_A = {               # 05b scenario A: bookmaker priors, neutral body
    "france_win": 0.650,
    "draw":       0.283,
    "norway_win": 0.067,
}
ORBITA_B = {               # 05b scenario B: bookmaker priors + momentum
    "france_win": 0.667,
    "draw":       0.233,
    "norway_win": 0.100,
}
BOOKMAKER = {              # Sky / Opta consensus
    "france_win": 0.60,
    "draw":       0.21,
    "norway_win": 0.19,
}


def brier(probs: dict, actual: str) -> float:
    """Multiclass Brier = sum over classes of (p_i - 1{actual==i})^2."""
    s = 0.0
    for label, p in probs.items():
        target = 1.0 if label == actual else 0.0
        s += (p - target) ** 2
    return s


def log_loss(probs: dict, actual: str, eps: float = 1e-9) -> float:
    p = max(eps, probs[actual])
    return -log(p)


def modal(probs: dict) -> str:
    return max(probs, key=probs.get)


def report_row(name: str, probs: dict, actual: str) -> None:
    b = brier(probs, actual)
    ll = log_loss(probs, actual)
    hit = "✓" if modal(probs) == actual else "✗"
    print(f"  {name:<32s}  Brier {b:>5.3f}   logloss {ll:>5.2f}   "
          f"modal={modal(probs):<11s} {hit}")


print("=== Orbita: Norway vs France — calibration review ===")
print(f"Match : Norway vs France, 2026-06-26, World Cup Group I")

if ACTUAL_OUTCOME is None:
    print()
    print("ACTUAL_OUTCOME is not set yet. After the final whistle:")
    print("  1. Set ACTUAL_OUTCOME to 'france_win' | 'draw' | 'norway_win'")
    print("  2. Set ACTUAL_SCORE  to the final score string")
    print("  3. (Optional) Add NOTES on anything notable")
    print("  4. Rerun this script.")
    raise SystemExit(0)

print(f"Score : {ACTUAL_SCORE}")
print(f"Result: {ACTUAL_OUTCOME}")
if NOTES:
    print(f"Notes : {NOTES}")
print()

print("Forecasts (lower Brier / lower logloss = better):")
report_row("Orbita headline (05)",   ORBITA_HEADLINE, ACTUAL_OUTCOME)
report_row("Orbita A (book priors)", ORBITA_A,        ACTUAL_OUTCOME)
report_row("Orbita B (+ momentum)",  ORBITA_B,        ACTUAL_OUTCOME)
report_row("Bookmaker consensus",    BOOKMAKER,       ACTUAL_OUTCOME)
print()

# Headline takeaway: did the public forecast beat the market?
b_orbita = brier(ORBITA_HEADLINE, ACTUAL_OUTCOME)
b_book   = brier(BOOKMAKER,        ACTUAL_OUTCOME)
gap = b_orbita - b_book

print("--- headline summary ---")
if gap < 0:
    print(f"Orbita BEAT the bookmaker by {abs(gap):.3f} Brier points "
          f"({b_orbita:.3f} vs {b_book:.3f}).")
else:
    print(f"Orbita LOST to the bookmaker by {gap:.3f} Brier points "
          f"({b_orbita:.3f} vs {b_book:.3f}).")

# Which configuration would have been the *best* calibrated in hindsight?
ranked = sorted(
    [
        ("headline", ORBITA_HEADLINE),
        ("A",        ORBITA_A),
        ("B",        ORBITA_B),
        ("book",     BOOKMAKER),
    ],
    key=lambda kv: brier(kv[1], ACTUAL_OUTCOME),
)
print(f"Best-calibrated configuration in hindsight: {ranked[0][0]}")
print()
print("Lesson for the next prediction:")
print("  - If 'A' won, the engine + market priors is the right baseline; "
      "author shifts hurt.")
print("  - If 'headline' won, the author's prior shift was justified; "
      "lean into intangible-driven adjustments.")
print("  - If 'book' won, neither configuration of Orbita beat the market — "
      "the prior elicitation pipeline needs work, not the engine.")

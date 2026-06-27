"""
09_today_world_cup.py — pre-kickoff slate for 2026-06-27

Runs the v0.3 Orbita engine against today's 7 World Cup fixtures and
prints the engine's forecast next to the de-vigged sportsbook line.

Inputs are bookmaker moneyline odds collected the morning of 2026-06-27
(bet365 / FanDuel for most; Oddschecker consensus for Algeria-Austria).
De-vigged by dividing each implied probability by the total overround.

Notes:
    - Priors-only run (no rosters). The v0.2 backtest showed roster
      mass-multipliers add a fraction of a Brier point on average and
      need per-player rating sourcing we don't have for these 7 sides.
    - No in-play observations — those land at kickoff. This is the
      kickoff-frozen forecast, equivalent to v0.2 behaviour. Once games
      start, the v0.3 sensor layer takes over (see 08_sensor_layer.py).
    - The engine is the same physics for every match; only the priors
      change. This is the slate the engine actually publishes.

Run with::

    PYTHONPATH=src python3 examples/09_today_world_cup.py
"""
from __future__ import annotations

import warnings
from math import log

import numpy as np

from orbita import Body, build_space, final_well, simulate


warnings.filterwarnings("ignore", message=r".*Renormalizing.*")

N_TRIALS = 60
SEED = 20260627
DT = 0.1


# Today's slate. side_a is the team listed second in the fixture name
# (the "favourite-leaning" side per the market) only by convention; the
# engine is symmetric in the labels.
MATCHES = [
    {
        "fixture": "Panama vs England",
        "side_a": ("england",  0.809),
        "side_b": ("panama",   0.073),
        "draw":   ("draw",     0.118),
    },
    {
        "fixture": "Croatia vs Ghana",
        "side_a": ("croatia",  0.541),
        "side_b": ("ghana",    0.172),
        "draw":   ("draw",     0.287),
    },
    {
        "fixture": "Algeria vs Austria",
        "side_a": ("austria",  0.333),
        "side_b": ("algeria",  0.240),
        "draw":   ("draw",     0.427),
    },
    {
        "fixture": "Jordan vs Argentina",
        "side_a": ("argentina", 0.801),
        "side_b": ("jordan",    0.073),
        "draw":   ("draw",      0.126),
    },
    {
        "fixture": "Colombia vs Portugal",
        "side_a": ("portugal",  0.507),
        "side_b": ("colombia",  0.237),
        "draw":   ("draw",      0.256),
    },
    {
        "fixture": "Uruguay vs Spain",
        "side_a": ("spain",     0.588),
        "side_b": ("uruguay",   0.157),
        "draw":   ("draw",      0.255),
    },
    {
        "fixture": "Uzbekistan vs DR Congo",
        "side_a": ("dr_congo",   0.550),
        "side_b": ("uzbekistan", 0.219),
        "draw":   ("draw",       0.231),
    },
]


def run_engine(space, sim_kwargs) -> dict:
    rng = np.random.default_rng(seed=SEED)
    counts = {a.label: 0 for a in space.attractors}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=[0.3, 0.2], size=2)
        p0 = rng.normal(scale=[0.15, 0.15], size=2)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(space, body=body, dt=DT, **sim_kwargs)
        counts[final_well(sol, space)] += 1
    total = sum(counts.values())
    return {label: c / total for label, c in counts.items()}


def fmt_probs(probs: dict) -> str:
    return "  ".join(f"{k}={v:>5.1%}" for k, v in probs.items())


def total_variation(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def modal(p: dict) -> str:
    return max(p, key=p.get)


print("=== Orbita v0.3: 2026-06-27 World Cup slate ===")
print(f"Monte Carlo N = {N_TRIALS}, seed = {SEED}")
print("Market = de-vigged sportsbook line; Engine = priors-only soccer template.")
print()

biggest_disagreement = (None, 0.0)
flips = []

for m in MATCHES:
    side_a_label, prior_a = m["side_a"]
    side_b_label, prior_b = m["side_b"]
    draw_label,   prior_draw = m["draw"]

    market = {side_a_label: prior_a, draw_label: prior_draw, side_b_label: prior_b}
    space, sim_kwargs = build_space(
        sport="soccer",
        side_a_label=side_a_label, prior_a=prior_a,
        side_b_label=side_b_label, prior_b=prior_b,
        draw_label=draw_label,     prior_draw=prior_draw,
    )
    engine = run_engine(space, sim_kwargs)
    # Sort engine output to match market label order for readability.
    engine = {k: engine[k] for k in (side_a_label, draw_label, side_b_label)}

    tv = total_variation(market, engine)
    if tv > biggest_disagreement[1]:
        biggest_disagreement = (m["fixture"], tv)
    if modal(market) != modal(engine):
        flips.append((m["fixture"], modal(market), modal(engine)))

    print(f"--- {m['fixture']}")
    print(f"    market : {fmt_probs(market)}")
    print(f"    engine : {fmt_probs(engine)}")
    print(f"    Δ (TV) : {tv:.3f}   "
          f"modal: market={modal(market)} engine={modal(engine)}")
    print()


print("=== summary ===")
print(f"Biggest disagreement : {biggest_disagreement[0]} "
      f"(TV={biggest_disagreement[1]:.3f})")
if flips:
    print("Modal flips (engine picks a different winner than the market):")
    for fix, market_pick, engine_pick in flips:
        print(f"  - {fix}: market→{market_pick}, engine→{engine_pick}")
else:
    print("No modal flips — engine agrees with the market on every favourite.")
print()
print("Caveat: this is the priors-only forecast. The v0.2 backtest")
print("showed priors-only over-sharpens vs the market on aggregate")
print("(0.537 Brier vs 0.421). The win is the v0.3 sensor layer once")
print("the matches kick off — that's the part the backtest said was")
print("still missing.")

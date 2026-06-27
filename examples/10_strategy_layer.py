"""
10_strategy_layer.py — replay today's slate as a strategy portfolio

Pulls everything from v0.3 together:

    - #5 multi-market event-spaces (Uruguay vs Spain with W/D/L,
      over/under 2.5, both-teams-to-score in one Match)
    - #6 saddle-point detection feeds per-market confidence into
    - #7 strategy layer's Kelly / confidence-weighted Kelly /
      EV-only views over the full 7-match slate

The bookmaker lines below are the decimal odds collected the morning of
2026-06-27 (same sources as 09_today_world_cup.py). Prop priors for
Uruguay vs Spain are de-vigged from the same morning's market.

Run with::

    PYTHONPATH=src python3 examples/10_strategy_layer.py
"""
from __future__ import annotations

import warnings

import numpy as np

from orbita import Forecast, Market, Match, blend, strategy


warnings.filterwarnings("ignore", message=r".*Renormalizing.*")

N_TRIALS = 60
SEED = 20260627
DT = 0.15
BANKROLL = 1000.0


# ---- Slate ---------------------------------------------------------------

# (label_a, prior_a, label_b, prior_b, draw_label, prior_draw,
#  odds_a, odds_b, odds_draw)
SLATE = [
    ("Panama vs England",
     "england", 0.809, "panama", 0.073, "draw", 0.118,
     1.167, 13.0, 8.0),
    ("Croatia vs Ghana",
     "croatia", 0.541, "ghana", 0.172, "draw", 0.287,
     1.746, 5.50, 3.30),
    ("Algeria vs Austria",
     "austria", 0.333, "algeria", 0.240, "draw", 0.427,
     2.88, 4.00, 2.25),
    ("Jordan vs Argentina",
     "argentina", 0.801, "jordan", 0.073, "draw", 0.126,
     1.182, 13.0, 7.50),
    ("Colombia vs Portugal",
     "portugal", 0.507, "colombia", 0.237, "draw", 0.256,
     1.87, 4.00, 3.70),
    ("Uruguay vs Spain",
     "spain", 0.588, "uruguay", 0.157, "draw", 0.255,
     1.606, 6.00, 3.70),
    ("Uzbekistan vs DR Congo",
     "dr_congo", 0.550, "uzbekistan", 0.219, "draw", 0.231,
     1.725, 4.33, 4.10),
]


def build_match(side_a, prior_a, side_b, prior_b, draw_label, prior_draw,
                *, multi_market=False):
    markets = [Market.from_sport(
        sport="soccer",
        side_a_label=side_a, prior_a=prior_a,
        side_b_label=side_b, prior_b=prior_b,
        draw_label=draw_label, prior_draw=prior_draw,
    )]
    if multi_market:
        # Uruguay vs Spain de-vigged props (FanDuel morning of 2026-06-27)
        markets.append(Market.over_under(line=2.5, prior_over=0.52))
        markets.append(Market.btts(prior_yes=0.494))
    return Match(markets=markets)


def fmt_probs(probs):
    return "  ".join(f"{k}={v:>5.1%}" for k, v in probs.items())


# ---- Multi-market focus: Uruguay vs Spain --------------------------------

print("=" * 78)
print("Multi-market forecast: Uruguay vs Spain")
print("=" * 78)
print()

uru_spa = build_match("spain", 0.588, "uruguay", 0.157,
                      "draw", 0.255, multi_market=True)
forecast = uru_spa.simulate(n_trials=N_TRIALS, seed=SEED, dt=DT)

print("Per-market marginals (engine, priors-only):")
for name in ("win_draw_lose", "over_under_2.5", "btts"):
    print(f"  {name:<18s} {fmt_probs(forecast.market(name))}   "
          f"confidence={forecast.confidence_for(name):.2f}")
print()

print("Joint distribution (win_draw_lose × over_under_2.5):")
joint = forecast.joint(["win_draw_lose", "over_under_2.5"])
for key, p in sorted(joint.items(), key=lambda kv: -kv[1]):
    if p < 0.01:
        continue
    print(f"  {' × '.join(key):<30s}  {p:>5.1%}")
print()

print("This is the value-add the bookmaker can't price by itself —")
print("they offer the marginals independently, but the joint is what")
print("a multi-leg bet (W/D/L + over/under same-game parlay) really is.")
print()


# ---- Portfolio across the full 7-match slate -----------------------------

print("=" * 78)
print("Portfolio: all 7 matches, W/D/L markets only")
print("=" * 78)
print(f"Bankroll = {BANKROLL:.0f}")
print()

# Build one combined Forecast by running each match and merging into a
# single multi-market structure (the strategy layer is market-name-keyed,
# so we just prefix each market by its fixture).
all_probs = {}
all_confidence = {}
all_lines = {}

for (fixture, side_a, prior_a, side_b, prior_b, draw_label, prior_draw,
     odds_a, odds_b, odds_draw) in SLATE:
    match = build_match(side_a, prior_a, side_b, prior_b,
                        draw_label, prior_draw)
    fc = match.simulate(n_trials=N_TRIALS, seed=SEED, dt=DT)
    market_key = fixture.replace(" ", "_")
    all_probs[market_key] = fc.market("win_draw_lose")
    all_confidence[market_key] = fc.confidence_for("win_draw_lose")
    all_lines[market_key] = {side_a: odds_a, side_b: odds_b,
                             draw_label: odds_draw}

combined = Forecast(
    market_names=tuple(all_probs),
    probs=all_probs,
    confidence=all_confidence,
    joint_counts={},
    n_trials=N_TRIALS,
)

# Strategy 1: standard fractional Kelly with edge + confidence gates
kelly_portfolio = strategy.kelly(
    forecast=combined, lines=all_lines, bankroll=BANKROLL,
    fraction=0.25, min_edge=0.03, min_confidence=0.50,
)

# Strategy 2: confidence-weighted Kelly (auto-shrinks on shaky picks)
cw_portfolio = strategy.confidence_weighted_kelly(
    forecast=combined, lines=all_lines, bankroll=BANKROLL,
    fraction=0.25, min_edge=0.03,
)

# Strategy 3: pure +EV listing (no sizing)
ev_portfolio = strategy.ev_only(
    forecast=combined, lines=all_lines, min_edge=0.0,
)


def print_portfolio(name, p):
    print(f"--- {name}")
    if not p.bets:
        print("    (no qualifying bets)")
    for b in p.bets:
        print(f"    {b.market:<24s} {b.selection:<12s} stake={b.stake:>7.2f}  "
              f"odds={b.odds:<5.2f}  edge={b.edge:+.2%}  conf={b.confidence:.2f}")
    print(f"    total_stake={p.total_stake:.2f}  EV={p.expected_value:+.2f}")
    if p.notes:
        for n in p.notes:
            print(f"    note: {n}")
    print()


print_portfolio("Quarter-Kelly (gate: edge ≥3%, confidence ≥0.50)",
                kelly_portfolio)
print_portfolio("Confidence-weighted quarter-Kelly (no hard gate, "
                "stake × confidence)", cw_portfolio)
print_portfolio("+EV listing (all picks engine prices above the line)",
                ev_portfolio)


# Strategy 4: calibrated forecast using the v0.2-fitted alpha.
# The multi-sport backtest collapsed alpha to ~0.005, meaning the engine
# adds essentially no orthogonal signal beyond the bookmaker prior on the
# 13-event panel. Blending at that alpha snaps the engine forecast back
# toward the market — and the Kelly portfolio correctly empties out.
ALPHA = 0.005
calibrated_probs = {}
calibrated_lines = {}
calibrated_confidence = {}
for (fixture, side_a, prior_a, side_b, prior_b, draw_label, prior_draw,
     odds_a, odds_b, odds_draw) in SLATE:
    market_key = fixture.replace(" ", "_")
    prior = {side_a: prior_a, draw_label: prior_draw, side_b: prior_b}
    engine_p = all_probs[market_key]
    calibrated = blend(prior, engine_p, ALPHA)
    calibrated_probs[market_key] = calibrated
    calibrated_lines[market_key] = all_lines[market_key]
    calibrated_confidence[market_key] = all_confidence[market_key]

calibrated_forecast = Forecast(
    market_names=tuple(calibrated_probs),
    probs=calibrated_probs,
    confidence=calibrated_confidence,
    joint_counts={},
    n_trials=N_TRIALS,
)
calibrated_portfolio = strategy.kelly(
    forecast=calibrated_forecast, lines=calibrated_lines, bankroll=BANKROLL,
    fraction=0.25, min_edge=0.03, min_confidence=0.0,
)
print_portfolio(
    f"Calibrated Kelly (alpha={ALPHA} from v0.2 backtest → "
    "engine snaps to market)",
    calibrated_portfolio,
)

print("=" * 78)
print("Reading the result — and why the raw Kelly EV is a mirage")
print("=" * 78)
print()
print("The raw Kelly portfolio above shows +20% expected returns and")
print("looks like a money printer. It isn't. Three things to keep in")
print("mind:")
print()
print("1. The engine over-sharpens its priors (v0.2 backtest finding).")
print("   It says \"England 98%\" against a market line of 86% — and the")
print("   strategy layer correctly reads that as a fat edge, then sizes")
print("   accordingly. But the 98% is not calibrated. The v0.2 panel")
print("   showed engine Brier 0.537 vs market 0.421 — over-sharpening")
print("   loses money on average even when the modal pick is right.")
print()
print("2. Confidence (#6) was 1.00 everywhere here. Soccer's drag drops")
print("   the body deep into a single well, far from any saddle. So the")
print("   confidence-weighted variant produces the same stakes as raw")
print("   Kelly — the signal is genuinely binary on this template.")
print("   The confidence layer is more useful for tennis / MMA (low")
print("   drag, multiple basins of attraction near the boundary).")
print()
print("3. The calibrated portfolio (alpha={:.3f}, fitted on the v0.2".format(ALPHA))
print("   backtest) snaps the engine forecast back to the market. It")
print("   empties out — and that IS the honest recommendation given")
print("   the current state of the engine on this slate. The strategy")
print("   layer is working; the calibration is telling us the engine")
print("   hasn't earned its independence from the bookmaker yet.")
print()
print("The right way to use this today: treat raw Kelly as a \"where")
print("does the engine disagree?\" diagnostic, and use the +EV listing")
print("to see which picks would survive calibration. Use the sensor")
print("layer (#2) to feed in-play data once games start.")

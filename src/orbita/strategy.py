"""Strategy layer (GitHub issue #7).

Reads a :class:`Forecast` (per-market probabilities + per-market confidence
from #6) and bookmaker lines, returns a :class:`Portfolio` of bets sized
by the chosen staking rule. The library never places bets — it returns a
plan object the caller can execute or ignore.

Available strategies:
    - :func:`kelly` — fractional Kelly with edge + confidence gates
    - :func:`confidence_weighted_kelly` — Kelly stake × confidence (the
      practical expression of "hedge against uncertainty")
    - :func:`flat` — unit stake threshold (edge + confidence)
    - :func:`hedge` — explicit two-side hedge when the bookmaker's
      two-way market sums to < 100% implied (rare but worth detecting)
    - :func:`ev_only` — list +EV picks without sizing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .markets import Forecast


@dataclass
class Bet:
    """A single sized recommendation."""

    market: str
    selection: str
    stake: float
    odds: float
    engine_prob: float
    edge: float
    confidence: float
    rationale: str

    @property
    def potential_return(self) -> float:
        """Net winnings if the bet wins (stake × (odds - 1))."""
        return self.stake * (self.odds - 1.0)

    @property
    def expected_value(self) -> float:
        return self.engine_prob * self.potential_return - (1 - self.engine_prob) * self.stake


@dataclass
class HedgeBet:
    """A pair of offsetting bets with a guaranteed return regardless of
    which leg wins. Only emitted when the two-way market actually
    permits it (sum of implied probabilities < 1)."""

    market: str
    legs: List[Bet]
    guaranteed_return: float


@dataclass
class Portfolio:
    """Collection of bets returned by a strategy."""

    bets: List[Bet] = field(default_factory=list)
    hedges: List[HedgeBet] = field(default_factory=list)
    bankroll: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def total_stake(self) -> float:
        single = sum(b.stake for b in self.bets)
        paired = sum(sum(leg.stake for leg in h.legs) for h in self.hedges)
        return single + paired

    @property
    def expected_value(self) -> float:
        return sum(b.expected_value for b in self.bets) + \
               sum(h.guaranteed_return for h in self.hedges)


# ---------- internal helpers ----------------------------------------------

def _kelly_fraction(p: float, odds: float) -> float:
    """Optimal Kelly fraction for a bet with win prob ``p`` and decimal
    odds ``odds``. Negative when the bet is -EV (don't bet)."""
    if odds <= 1.0:
        return 0.0
    return (p * odds - 1.0) / (odds - 1.0)


def _implied_probability(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.0


def _cap_to_bankroll(bets: List[Bet], bankroll: float) -> List[str]:
    notes: List[str] = []
    total = sum(b.stake for b in bets)
    if total > bankroll > 0:
        scale = bankroll / total
        for b in bets:
            b.stake *= scale
        notes.append(
            f"Total Kelly stake {total:.2f} exceeded bankroll {bankroll:.2f}; "
            f"scaled all stakes by {scale:.3f}"
        )
    return notes


# ---------- strategies ----------------------------------------------------

def kelly(
    forecast: Forecast,
    lines: Dict[str, Dict[str, float]],
    bankroll: float,
    fraction: float = 0.25,
    min_edge: float = 0.02,
    min_confidence: float = 0.5,
) -> Portfolio:
    """Fractional Kelly with edge + confidence gates."""
    bets: List[Bet] = []
    notes: List[str] = []
    for market_name, market_odds in lines.items():
        if market_name not in forecast.probs:
            notes.append(f"Skipped {market_name!r}: not in forecast")
            continue
        probs = forecast.market(market_name)
        conf = forecast.confidence_for(market_name)
        if conf < min_confidence:
            notes.append(
                f"Skipped {market_name!r}: confidence {conf:.2f} < "
                f"{min_confidence}"
            )
            continue
        for selection, odds in market_odds.items():
            p = probs.get(selection, 0.0)
            implied = _implied_probability(odds)
            edge = p - implied
            if edge < min_edge:
                continue
            f_kelly = _kelly_fraction(p, odds)
            if f_kelly <= 0:
                continue
            stake = bankroll * fraction * f_kelly
            bets.append(Bet(
                market=market_name, selection=selection,
                stake=stake, odds=odds,
                engine_prob=p, edge=edge, confidence=conf,
                rationale=(f"Kelly f={f_kelly:.3f} × {fraction:.2f} fraction "
                           f"(edge {edge:+.2%})"),
            ))
    notes.extend(_cap_to_bankroll(bets, bankroll))
    return Portfolio(bets=bets, bankroll=bankroll, notes=notes)


def confidence_weighted_kelly(
    forecast: Forecast,
    lines: Dict[str, Dict[str, float]],
    bankroll: float,
    fraction: float = 0.25,
    min_edge: float = 0.02,
) -> Portfolio:
    """Kelly stake × confidence. Naturally shrinks toward zero as the
    forecast's confidence drops — the practical "hedge against
    uncertainty" without explicit hedge legs."""
    bets: List[Bet] = []
    for market_name, market_odds in lines.items():
        if market_name not in forecast.probs:
            continue
        probs = forecast.market(market_name)
        conf = forecast.confidence_for(market_name)
        for selection, odds in market_odds.items():
            p = probs.get(selection, 0.0)
            implied = _implied_probability(odds)
            edge = p - implied
            if edge < min_edge:
                continue
            f_kelly = _kelly_fraction(p, odds)
            if f_kelly <= 0:
                continue
            stake = bankroll * fraction * f_kelly * conf
            bets.append(Bet(
                market=market_name, selection=selection,
                stake=stake, odds=odds,
                engine_prob=p, edge=edge, confidence=conf,
                rationale=(f"Kelly f={f_kelly:.3f} × {fraction:.2f} × "
                           f"conf {conf:.2f} (edge {edge:+.2%})"),
            ))
    notes = _cap_to_bankroll(bets, bankroll)
    return Portfolio(bets=bets, bankroll=bankroll, notes=notes)


def flat(
    forecast: Forecast,
    lines: Dict[str, Dict[str, float]],
    bankroll: float,
    stake_per_bet: float,
    min_edge: float = 0.02,
    min_confidence: float = 0.5,
) -> Portfolio:
    """One unit per qualifying pick — useful for backtesting ROI without
    Kelly's variance compounding."""
    bets: List[Bet] = []
    for market_name, market_odds in lines.items():
        if market_name not in forecast.probs:
            continue
        probs = forecast.market(market_name)
        conf = forecast.confidence_for(market_name)
        if conf < min_confidence:
            continue
        for selection, odds in market_odds.items():
            p = probs.get(selection, 0.0)
            edge = p - _implied_probability(odds)
            if edge < min_edge:
                continue
            bets.append(Bet(
                market=market_name, selection=selection,
                stake=stake_per_bet, odds=odds,
                engine_prob=p, edge=edge, confidence=conf,
                rationale=f"flat {stake_per_bet} (edge {edge:+.2%})",
            ))
    notes = _cap_to_bankroll(bets, bankroll)
    return Portfolio(bets=bets, bankroll=bankroll, notes=notes)


def ev_only(
    forecast: Forecast,
    lines: Dict[str, Dict[str, float]],
    min_edge: float = 0.0,
) -> Portfolio:
    """List every +EV (or above-threshold) pick without sizing. Stake is
    set to 1.0 for accounting; the caller decides how to size."""
    bets: List[Bet] = []
    for market_name, market_odds in lines.items():
        if market_name not in forecast.probs:
            continue
        probs = forecast.market(market_name)
        conf = forecast.confidence_for(market_name)
        for selection, odds in market_odds.items():
            p = probs.get(selection, 0.0)
            edge = p - _implied_probability(odds)
            if edge < min_edge:
                continue
            bets.append(Bet(
                market=market_name, selection=selection,
                stake=1.0, odds=odds,
                engine_prob=p, edge=edge, confidence=conf,
                rationale=f"+EV listing (edge {edge:+.2%})",
            ))
    return Portfolio(bets=bets, bankroll=0.0)


def hedge(
    forecast: Forecast,
    market_name: str,
    selection_a: str,
    selection_b: str,
    odds_a: float,
    odds_b: float,
    total_stake: float,
) -> Optional[HedgeBet]:
    """Compute an offsetting two-side hedge such that the realised return
    is the same regardless of which leg wins.

    Solves
        s_A + s_B = total_stake
        s_A · odds_A = s_B · odds_B
    so the gross payout is identical either way. Returns ``None`` when
    the bookmaker's overround on the two selections makes the hedge
    strictly -EV — better to walk away than to lock in a loss.
    """
    if market_name not in forecast.probs:
        return None
    if odds_a <= 1.0 or odds_b <= 1.0:
        return None
    s_a = total_stake * odds_b / (odds_a + odds_b)
    s_b = total_stake * odds_a / (odds_a + odds_b)
    gross_payout = s_a * odds_a   # equals s_b * odds_b by construction
    guaranteed_return = gross_payout - total_stake

    if guaranteed_return <= 0:
        return None

    probs = forecast.market(market_name)
    conf = forecast.confidence_for(market_name)
    leg_a = Bet(
        market=market_name, selection=selection_a,
        stake=s_a, odds=odds_a,
        engine_prob=probs.get(selection_a, 0.0),
        edge=probs.get(selection_a, 0.0) - _implied_probability(odds_a),
        confidence=conf,
        rationale=f"hedge leg A — guaranteed return {guaranteed_return:.2f}",
    )
    leg_b = Bet(
        market=market_name, selection=selection_b,
        stake=s_b, odds=odds_b,
        engine_prob=probs.get(selection_b, 0.0),
        edge=probs.get(selection_b, 0.0) - _implied_probability(odds_b),
        confidence=conf,
        rationale=f"hedge leg B — guaranteed return {guaranteed_return:.2f}",
    )
    return HedgeBet(market=market_name, legs=[leg_a, leg_b],
                    guaranteed_return=guaranteed_return)

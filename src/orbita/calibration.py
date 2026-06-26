"""Sharpening calibration (GitHub issue #3).

The v0.1 engine systematically over-sharpens its priors. Feed it a 60/40
prior and it returns ~80/20; feed it 80/20 and it returns ~95/5. On the
multi-sport backtest panel this cost +0.16 mean Brier vs the bookmaker
baseline.

The fix is a single learnable parameter ``alpha`` that geometrically
blends the prior with the engine's output:

    log_blend = (1 - alpha) * log(prior) + alpha * log(engine_output)
    calibrated = softmax(log_blend)

``alpha == 0`` recovers the bookmaker exactly. ``alpha == 1`` recovers
the raw engine. The optimum is fit by minimising aggregate Brier on a
backtest panel.

Diagnostic interpretation of the fitted value:

    alpha → 1   : engine is well-calibrated; trust it.
    alpha → 0.5 : engine adds partial signal; blend.
    alpha → 0   : engine adds NO signal beyond the prior — the gravity-
                  well physics is the bottleneck, not the post-hoc knob.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


PriorEngineActual = Tuple[Dict[str, float], Dict[str, float], str]


def blend(prior: Dict[str, float],
          engine: Dict[str, float],
          alpha: float) -> Dict[str, float]:
    """Geometric mix of ``prior`` and ``engine`` controlled by ``alpha``.

    Both inputs must share the same keys. Output is a proper distribution
    (sums to 1.0, non-negative).
    """
    labels = list(prior.keys())
    if set(engine.keys()) != set(labels):
        raise ValueError("prior and engine must share the same label set")
    eps = 1e-9
    log_p = np.log([max(prior[l], eps) for l in labels])
    log_e = np.log([max(engine[l], eps) for l in labels])
    log_mix = (1.0 - alpha) * log_p + alpha * log_e
    log_mix -= log_mix.max()
    e = np.exp(log_mix)
    e /= e.sum()
    return dict(zip(labels, [float(x) for x in e]))


def brier(probs: Dict[str, float], actual: str) -> float:
    """Multiclass Brier score for one observation."""
    return float(sum(
        (p - (1.0 if label == actual else 0.0)) ** 2
        for label, p in probs.items()
    ))


def aggregate_brier(records: Sequence[PriorEngineActual], alpha: float) -> float:
    """Mean Brier across ``records`` under the given alpha-blend."""
    if not records:
        return 0.0
    return float(np.mean([
        brier(blend(p, e, alpha), actual) for p, e, actual in records
    ]))


def fit_alpha(records: Sequence[PriorEngineActual],
              grid: int = 201) -> Tuple[float, float]:
    """Find the alpha in [0, 1] that minimises mean Brier.

    Grid search over ``grid`` values. Returns ``(alpha, brier_at_alpha)``.

    With small panels (n ≲ 20) this is in-sample fit. Use
    :func:`loocv_alpha` for an out-of-sample estimate.
    """
    alphas = np.linspace(0.0, 1.0, grid)
    scores = np.array([aggregate_brier(records, float(a)) for a in alphas])
    i = int(np.argmin(scores))
    return float(alphas[i]), float(scores[i])


def loocv_alpha(records: Sequence[PriorEngineActual],
                grid: int = 201) -> Tuple[List[float], float]:
    """Leave-one-out cross-validation for alpha.

    For each held-out record, fit alpha on the other (n-1) records and
    score that held-out record under the fitted alpha. Returns the list
    of held-out alphas and the mean held-out Brier.
    """
    held_alphas = []
    held_briers = []
    n = len(records)
    for i in range(n):
        train = [records[j] for j in range(n) if j != i]
        a, _ = fit_alpha(train, grid=grid)
        test_brier = brier(blend(records[i][0], records[i][1], a),
                           records[i][2])
        held_alphas.append(a)
        held_briers.append(test_brier)
    return held_alphas, float(np.mean(held_briers))

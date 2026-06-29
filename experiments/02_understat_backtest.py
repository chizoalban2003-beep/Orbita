"""02_understat_backtest.py — does the v0.3.3 win generalize?

The 13-event hand-curated panel (`data/backtest_matches.toml`) is too
small to be conclusive. This script pulls a larger soccer panel from
understat.com via :class:`orbita.UnderstatMatchProvider` and re-runs the
v0.3.3 engine (sport-specific IC scale + soft Plummer posterior) against
the bookmaker priors that Understat itself publishes.

If the v0.3.3 engine still beats the market on aggregate Brier across a
panel an order of magnitude larger, we have an honest signal. If it
doesn't, the original 13-event win was a small-sample fluke and we have
to keep digging.

The match panel is cached on disk under ``~/.cache/orbita/understat`` so
re-runs are offline and reproducible. Default panel is a 35-match
EPL window from Oct-Nov 2024 (matchdays 8-12, IDs 26680-26714).

Run with::

    PYTHONPATH=src python3 experiments/02_understat_backtest.py
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import (  # noqa: E402
    Body,
    UnderstatMatchProvider,
    aggregate_brier,
    blend,
    build_space,
    final_well_posterior,
    fit_alpha,
    loocv_alpha,
    simulate,
)

warnings.filterwarnings("ignore", message=r".*Renormalizing.*")

# Default panel: 35-match EPL window (Oct 19 – Nov 25, 2024, matchdays
# 8-12). Chosen as a contiguous block to avoid look-ahead cherry-picking.
DEFAULT_MATCH_IDS = [str(i) for i in range(26680, 26715)]

N_TRIALS = 200
SEED = 20260627
DT = 0.1
SOFT_ALPHA = 2.0


def brier(probs: dict, actual: str) -> float:
    return sum((p - (1.0 if l == actual else 0.0)) ** 2
               for l, p in probs.items())


def log_loss(probs: dict, actual: str, eps: float = 1e-9) -> float:
    return -log(max(eps, probs[actual]))


def modal(probs: dict) -> str:
    return max(probs, key=probs.get)


def run_engine(space, sim_kwargs) -> dict:
    """Same Monte-Carlo loop as examples/07 — kept inline so this
    experiment stays self-contained."""
    rng = np.random.default_rng(seed=SEED)
    ic_scale = sim_kwargs.get("ic_scale", 1.0)
    q_scale = np.array([0.3, 0.2]) * ic_scale
    p_scale = np.array([0.15, 0.15]) * ic_scale
    acc = {a.label: 0.0 for a in space.attractors}
    for _ in range(N_TRIALS):
        q0 = rng.normal(scale=q_scale)
        p0 = rng.normal(scale=p_scale)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(space, body=body, dt=DT, **sim_kwargs)
        probs = final_well_posterior(sol, space, alpha=SOFT_ALPHA)
        for label, p in probs.items():
            acc[label] += p
    total = sum(acc.values())
    return {label: v / total for label, v in acc.items()}


def score_one(match_dict: dict) -> tuple:
    side_a = match_dict["side_a_label"]
    side_b = match_dict["side_b_label"]
    draw_label = match_dict["draw_label"]
    actual = match_dict["actual"]
    priors = {
        side_a:     match_dict["prior_a"],
        draw_label: match_dict["prior_draw"],
        side_b:     match_dict["prior_b"],
    }
    space, sim_kwargs = build_space(
        sport="soccer",
        side_a_label=side_a,
        side_b_label=side_b,
        prior_a=match_dict["prior_a"],
        prior_b=match_dict["prior_b"],
        prior_draw=match_dict["prior_draw"],
        draw_label=draw_label,
    )
    engine_probs = run_engine(space, sim_kwargs)
    return priors, engine_probs, actual


def main() -> None:
    prov = UnderstatMatchProvider()
    print("=== Orbita: Understat panel backtest (v0.3.3) ===")
    print(f"Provider cache : {prov.cache_dir}")
    print(f"Panel size     : {len(DEFAULT_MATCH_IDS)} matches")
    print(f"Monte Carlo N  : {N_TRIALS}  (seed={SEED})")
    print()

    matches = []
    for mid in DEFAULT_MATCH_IDS:
        m = prov.fetch_match(mid)
        matches.append(m.as_backtest_dict())

    records = []
    full_results = []
    for i, m in enumerate(matches, 1):
        print(f"[{i}/{len(matches)}] {m['date']} {m['event']}", flush=True)
        priors, engine_probs, actual = score_one(m)
        records.append((priors, engine_probs, actual))
        full_results.append((m, priors, engine_probs, actual))

    # alpha calibration (in-sample + LOOCV).
    alpha_is, brier_is = fit_alpha(records)
    loocv_alphas, loocv_brier = loocv_alpha(records)
    print()
    print(f"Fitted alpha (in-sample)     : {alpha_is:.3f}  "
          f"(mean Brier @ alpha = {brier_is:.3f})")
    print(f"LOOCV alpha (per-fold)       : "
          f"min={min(loocv_alphas):.3f}  max={max(loocv_alphas):.3f}  "
          f"median={float(np.median(loocv_alphas)):.3f}")
    print(f"LOOCV mean Brier             : {loocv_brier:.3f}")
    print()

    # Per-match and aggregate scoring.
    overall = defaultdict(list)
    hits = defaultdict(int)
    for m, priors, engine_p, actual in full_results:
        calibrated = blend(priors, engine_p, alpha_is)
        overall["bookmaker"].append(brier(priors, actual))
        overall["orbita_priors"].append(brier(engine_p, actual))
        overall["orbita_calibrated"].append(brier(calibrated, actual))
        for col, probs in (("bookmaker", priors),
                           ("orbita_priors", engine_p),
                           ("orbita_calibrated", calibrated)):
            if modal(probs) == actual:
                hits[col] += 1

    n = len(full_results)
    print("=== aggregate Brier (mean — lower is better) ===")
    cols = ("bookmaker", "orbita_priors", "orbita_calibrated")
    print(f"{'model':<22s}  {'N':>3s}  {'mean Brier':>12s}  {'modal hit-rate':>16s}")
    for c in cols:
        mean_brier = float(np.mean(overall[c]))
        hit_rate = hits[c] / n
        print(f"{c:<22s}  {n:>3d}  {mean_brier:>12.3f}  {hit_rate:>15.1%}")
    print()

    mb = float(np.mean(overall["bookmaker"]))
    mp = float(np.mean(overall["orbita_priors"]))
    mc = float(np.mean(overall["orbita_calibrated"]))

    print("=== verdicts (delta = engine - bookmaker, negative = engine wins) ===")
    print(f"priors-only vs market    : {mp - mb:+.3f}  "
          f"({'engine wins' if mp < mb else 'market wins'})")
    print(f"calibrated vs market     : {mc - mb:+.3f}  "
          f"({'calibrated engine wins' if mc < mb else 'market wins'})")
    print(f"LOOCV vs market          : {loocv_brier - mb:+.3f}  "
          f"({'calibrated engine wins' if loocv_brier < mb else 'market wins'})")
    print()

    # Reproducibility footer.
    print(f"Panel: Understat match IDs {DEFAULT_MATCH_IDS[0]}-{DEFAULT_MATCH_IDS[-1]}")
    dates = sorted({m['date'] for m, *_ in full_results})
    print(f"Date range: {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    main()

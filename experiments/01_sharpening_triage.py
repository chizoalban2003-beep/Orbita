"""
01_sharpening_triage.py

Diagnose and triage the over-sharpening that caps the v0.2 backtest at
Brier 0.537 (vs bookmaker 0.421) on soccer + NBA.

Hypothesis: ``final_well`` hard-classifies each MC trial to the nearest
attractor. With narrow initial conditions and N=40 trials, most trials
funnel into the modal well -> the engine's posterior is much sharper
than the bookmaker's. The fix is to replace hard-classify with the
physically correct soft posterior: weight each trial's contribution to
well k by the mass-weighted Plummer attraction at the trial's final
state.

What this script does
---------------------
For every match in data/backtest_matches.toml, run the priors-only
engine under several configurations:

    A. CURRENT       — hard-classify, N=40, narrow IC (v0.2 default)
    B. SOFT(alpha)   — soft Boltzmann assignment with alpha in {1, 2, 3}
                       (alpha = exponent on inverse-Plummer distance)
    C. WIDE_IC       — q0/p0 scale x2.5, hard-classify
    D. SOFT + WIDE   — combine B with widened IC

Report per-sport mean Brier and the engine-vs-market delta. Pick the
combo that wins on soccer/NBA without breaking tennis/MMA.

Run with::

    PYTHONPATH=src python3 experiments/01_sharpening_triage.py
"""
from __future__ import annotations

import sys
import tomllib
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbita import Body, build_space, simulate
from orbita.forces import SOFTENING


warnings.filterwarnings("ignore", message=r".*Renormalizing.*")

DATA_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "backtest_matches.toml"
)

SEED = 20260626
DT = 0.1


# ---------- assignment kernels --------------------------------------------

def hard_assignment(q_end: np.ndarray, space) -> dict:
    """Original v0.2 behaviour: 100% mass on the nearest well."""
    dists = [float(np.linalg.norm(q_end - a.position)) for a in space.attractors]
    idx = int(np.argmin(dists))
    return {
        a.label: (1.0 if i == idx else 0.0)
        for i, a in enumerate(space.attractors)
    }


def soft_assignment(q_end: np.ndarray, space, alpha: float = 2.0) -> dict:
    """Mass-weighted Plummer assignment.

    Each trial's final state contributes
        P(well_k | q_end) = m_k / r_k^alpha
    normalised across wells. r_k is the Plummer-softened distance to
    well k. alpha controls sharpness:
        alpha = 0  -> just the mass prior (no info from q_end)
        alpha = 1  -> proportional to gravitational potential magnitude
        alpha = 2  -> proportional to gravitational force magnitude
        alpha = inf -> hard nearest-well
    """
    weights = {}
    for a in space.attractors:
        d = float(np.sqrt(np.sum((q_end - a.position) ** 2) + SOFTENING ** 2))
        weights[a.label] = a.mass / (d ** alpha)
    total = sum(weights.values())
    return {label: w / total for label, w in weights.items()}


# ---------- MC engine runners ---------------------------------------------

def run_engine(space, sim_kwargs, *, mode: str, alpha: float = 2.0,
               n_trials: int = 40, ic_scale: float = 1.0) -> dict:
    """Run N MC trials and combine per-trial soft (or hard) outputs."""
    rng = np.random.default_rng(seed=SEED)
    acc = {a.label: 0.0 for a in space.attractors}
    q_scale = np.array([0.3, 0.2]) * ic_scale
    p_scale = np.array([0.15, 0.15]) * ic_scale
    for _ in range(n_trials):
        q0 = rng.normal(scale=q_scale)
        p0 = rng.normal(scale=p_scale)
        body = Body(mass=1.0, q0=q0, p0=p0)
        sol = simulate(space, body=body, dt=DT, **sim_kwargs)
        q_end = sol["q"][-1]
        if mode == "hard":
            probs = hard_assignment(q_end, space)
        elif mode == "soft":
            probs = soft_assignment(q_end, space, alpha=alpha)
        else:
            raise ValueError(mode)
        for label, p in probs.items():
            acc[label] += p
    total = sum(acc.values())
    return {label: v / total for label, v in acc.items()}


def brier(probs: dict, actual: str) -> float:
    return sum((p - (1.0 if l == actual else 0.0)) ** 2
               for l, p in probs.items())


def modal(probs: dict) -> str:
    return max(probs, key=probs.get)


# ---------- driver ---------------------------------------------------------

CONFIGS = [
    # name,              mode,  alpha, n_trials, ic_scale
    ("A_current",        "hard", 2.0,  40,       1.0),
    ("B_soft_a1",        "soft", 1.0,  40,       1.0),
    ("B_soft_a2",        "soft", 2.0,  40,       1.0),
    ("B_soft_a3",        "soft", 3.0,  40,       1.0),
    ("C_wide_IC",        "hard", 2.0,  40,       2.5),
    ("D_soft_wide_a2",   "soft", 2.0,  40,       2.5),
    ("E_soft_a2_N200",   "soft", 2.0,  200,      1.0),
    ("F_soft_a2_wide_N200", "soft", 2.0, 200,    2.5),
]


def fmt_probs(probs: dict) -> str:
    return "  ".join(f"{k[:14]}={v:.0%}" for k, v in probs.items())


def main() -> None:
    if not DATA_FILE.exists():
        print(f"ERROR: data file not found at {DATA_FILE}", flush=True)
        sys.exit(1)
    with DATA_FILE.open("rb") as fh:
        data = tomllib.load(fh)
    matches = data.get("match", [])

    print("=== Orbita: over-sharpening triage ===")
    print(f"Data file       : {DATA_FILE.name}  ({len(matches)} matches)")
    print(f"Configs         : {len(CONFIGS)}")
    print(f"DT              : {DT}, seed={SEED}")
    print()

    # per_config[name][sport] = list[brier]; also per_config[name]["ALL"]
    per_config: dict = {name: defaultdict(list) for name, *_ in CONFIGS}
    bookmaker: dict = defaultdict(list)
    # collect engine output probs for a few sample upset matches
    upsets: dict = defaultdict(list)
    upset_match_idxs = []

    for i, m in enumerate(matches):
        sport = m["sport"]
        side_a = m["side_a_label"]
        side_b = m["side_b_label"]
        draw_label = m.get("draw_label")
        actual = m["actual"]

        if "prior_draw" in m:
            priors = {
                side_a:     m["prior_a"],
                draw_label: m["prior_draw"],
                side_b:     m["prior_b"],
            }
        else:
            priors = {side_a: m["prior_a"], side_b: m["prior_b"]}

        # detect upset: actual is not the modal prior
        if modal(priors) != actual:
            upset_match_idxs.append(i)

        # bookmaker = priors as-is
        b = brier(priors, actual)
        bookmaker[sport].append(b)
        bookmaker["ALL"].append(b)

        space, sim_kwargs = build_space(
            sport=sport,
            side_a_label=side_a,
            side_b_label=side_b,
            prior_a=m["prior_a"],
            prior_b=m["prior_b"],
            prior_draw=m.get("prior_draw"),
            draw_label=draw_label,
        )

        print(f"[{i+1}/{len(matches)}] {sport} {m.get('date', '?')} "
              f"actual={actual} priors={fmt_probs(priors)} "
              f"bookmaker_brier={b:.3f}", flush=True)

        for name, mode, alpha, n_trials, ic_scale in CONFIGS:
            engine = run_engine(
                space, sim_kwargs,
                mode=mode, alpha=alpha,
                n_trials=n_trials, ic_scale=ic_scale,
            )
            br = brier(engine, actual)
            per_config[name][sport].append(br)
            per_config[name]["ALL"].append(br)
            if i in upset_match_idxs:
                upsets[name].append((i, engine, br))

    print()
    print("=== aggregate Brier per config (mean; lower is better) ===")
    sports_seen = [s for s in bookmaker if s != "ALL"]
    header = f"{'config':<22s} " + "  ".join(
        f"{s:>10s}" for s in sports_seen) + f"  {'ALL':>10s}"
    print(header)
    book_row = f"{'BOOKMAKER':<22s} " + "  ".join(
        f"{np.mean(bookmaker[s]):>10.3f}" for s in sports_seen)
    book_row += f"  {np.mean(bookmaker['ALL']):>10.3f}"
    print(book_row)
    for name, *_ in CONFIGS:
        row = f"{name:<22s} " + "  ".join(
            f"{np.mean(per_config[name][s]):>10.3f}" for s in sports_seen)
        row += f"  {np.mean(per_config[name]['ALL']):>10.3f}"
        print(row)

    print()
    print("=== delta vs bookmaker (engine - bookmaker; negative = engine wins) ===")
    print(header)
    for name, *_ in CONFIGS:
        row = f"{name:<22s} " + "  ".join(
            f"{np.mean(per_config[name][s]) - np.mean(bookmaker[s]):>+10.3f}"
            for s in sports_seen)
        row += (f"  {np.mean(per_config[name]['ALL']) - np.mean(bookmaker['ALL']):>+10.3f}")
        print(row)

    print()
    print("=== upset diagnostic (matches where bookmaker modal was wrong) ===")
    for i in upset_match_idxs:
        m = matches[i]
        print(f"  {m['sport']:<8s} {m.get('date','?')} "
              f"event={m.get('event','?')[:60]} actual={m['actual']}")
        for name, *_ in CONFIGS:
            tup = next(t for t in upsets[name] if t[0] == i)
            _, engine, br = tup
            print(f"    {name:<22s} brier={br:.3f}  [{fmt_probs(engine)}]")


if __name__ == "__main__":
    main()

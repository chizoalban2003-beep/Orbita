"""16_reliability_calibration.py — Are engine posteriors calibrated?

Exps 10-15 established that on Brier, the engine ties the market at
best on public pre-match features. But Brier collapses *sharpness*
and *calibration* into one scalar. A model that ties Brier while
being better-calibrated (its stated 60% events happen ~60% of the
time; the market's 60% events happen ~70% of the time because the
market is over-confident on favourites) is a different, publishable
kind of result.

This experiment:
  1. Runs the anisotropic baseline (0.00, 0.16) on the full 380-match
     EPL 2024/25 season, unbiased p0, N=30 trials.
  2. For each match, saves engine + market posteriors and actual
     outcome to a CSV so downstream analyses (alpha-blend, cross-market
     comparison) can be post-hoc and instant.
  3. Bins predictions into deciles and computes:
       - reliability curves (predicted vs observed frequency per bin)
       - Expected Calibration Error (ECE)
       - Maximum Calibration Error (MCE)
     For both engine and market, on H/D/A (3-way) and O/U (binary).
  4. Bootstrap 90% CI on the engine-market ECE gap.

Interpretation:
  - ECE_engine < ECE_market: engine is better calibrated. Even if
    Brier is tied, this is a real, orthogonal result.
  - Slope of the reliability curve: closer to 1.0 = better. Overshoot
    = over-confident; undershoot = under-confident.

Run with::

    PYTHONPATH=src python3 experiments/16_reliability_calibration.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import EventSpace  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e05", ROOT / "experiments" / "05_player_attractor_panel.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)

spec2 = importlib.util.spec_from_file_location(
    "e14", ROOT / "experiments" / "14_momentum_ic_features.py")
e14 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(e14)

e05.N_TRIALS = 30
SEED = 20260702
BEST_CD_OU = np.array([0.00, 0.16])
CACHE_PATH = ROOT / "experiments" / "_cache_16_posteriors.csv"


def score_and_save(matches: list, out_path: Path) -> list:
    """Runs anisotropic baseline over all matches, saves posteriors."""
    rows = []
    hda_cols = ["home", "draw", "away"]
    ou_cols = ["over", "under"]
    for i, m in enumerate(matches, 1):
        wells = e05.multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"],
            m["p_over"], m["p_under"],
        )
        space = EventSpace(wells)
        rng = np.random.default_rng(seed=e05.SEED)
        q_scale = np.array([0.3, 0.2]) * e05.IC_SCALE
        p_scale = np.array([0.15, 0.15]) * e05.IC_SCALE
        f_pos, f_mass, _ = e05._stack(space)
        o_pos, o_mass, o_labels = e05._stack(space)
        acc = {l: 0.0 for l in o_labels}
        for _ in range(e05.N_TRIALS):
            q0 = rng.normal(scale=q_scale)
            p0 = rng.normal(scale=p_scale)
            q_end = e05.fast_final_q(f_pos, f_mass, q0, p0,
                                     duration=e05.DURATION, dt=e05.DT,
                                     C_d=BEST_CD_OU)
            probs = e05.fast_posterior(q_end, o_pos, o_mass, o_labels,
                                       alpha=e05.SOFT_ALPHA)
            for label, p in probs.items():
                acc[label] += p
        total = sum(acc.values())
        joint = {l: v / total for l, v in acc.items()}
        hda_e = e05.marginal_hda(joint, m["home_label"], m["away_label"])
        ou_e = e05.marginal_ou(joint)
        actual_hda_label = m["actual_hda"]
        actual_hda_side = ("home" if actual_hda_label == m["home_label"]
                           else "away" if actual_hda_label == m["away_label"]
                           else "draw")
        rows.append({
            "match_id": i,
            "date": m["date"],
            "event": m["event"],
            "actual_hda": actual_hda_side,
            "actual_ou": m["actual_ou"],
            "eng_h": hda_e[m["home_label"]],
            "eng_d": hda_e["draw"],
            "eng_a": hda_e[m["away_label"]],
            "eng_o": ou_e["over"],
            "eng_u": ou_e["under"],
            "mkt_h": m["p_h"],
            "mkt_d": m["p_d"],
            "mkt_a": m["p_a"],
            "mkt_o": m["p_over"],
            "mkt_u": m["p_under"],
        })
        if i % 50 == 0:
            print(f"  [{i:>3d}/{len(matches)}] scored", flush=True)
    fields = list(rows[0].keys())
    with out_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved posteriors -> {out_path}", flush=True)
    return rows


def collect_pairs(rows, engine: bool):
    """Yields (predicted_prob, observed_1_or_0) for every class of
    every match. This is the raw material for a reliability curve."""
    prefix = "eng" if engine else "mkt"
    hda_pairs = []
    for r in rows:
        for side, key in [("home", "h"), ("draw", "d"), ("away", "a")]:
            p = r[f"{prefix}_{key}"]
            y = 1.0 if r["actual_hda"] == side else 0.0
            hda_pairs.append((p, y))
    ou_pairs = []
    for r in rows:
        for side, key in [("over", "o"), ("under", "u")]:
            p = r[f"{prefix}_{key}"]
            y = 1.0 if r["actual_ou"] == side else 0.0
            ou_pairs.append((p, y))
    return (np.array(hda_pairs, dtype=float),
            np.array(ou_pairs, dtype=float))


def reliability_bins(pairs, n_bins=10):
    """Equal-width binning on predicted probability. Returns
    (bin_centers, mean_predicted, mean_observed, counts)."""
    p, y = pairs[:, 0], pairs[:, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mp = np.full(n_bins, np.nan)
    mo = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        cnt[b] = int(mask.sum())
        if cnt[b] > 0:
            mp[b] = float(p[mask].mean())
            mo[b] = float(y[mask].mean())
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, mp, mo, cnt


def ece_mce(pairs, n_bins=10):
    centers, mp, mo, cnt = reliability_bins(pairs, n_bins=n_bins)
    total = cnt.sum()
    if total == 0:
        return 0.0, 0.0
    weights = cnt / total
    per_bin_err = np.where(cnt > 0, np.abs(mo - mp), 0.0)
    ece = float((weights * per_bin_err).sum())
    mce = float(per_bin_err[cnt > 0].max()) if (cnt > 0).any() else 0.0
    return ece, mce


def bootstrap_ci_delta_ece(pairs_a, pairs_b, n_boot=2000, n_bins=10):
    """Bootstrap on the per-match ECE difference ECE(a) − ECE(b).
    Both inputs are (n*classes, 2) arrays where every `classes`
    consecutive rows correspond to one match."""
    rng = np.random.default_rng(SEED)
    n = pairs_a.shape[0]
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_a = pairs_a[idx]
        sample_b = pairs_b[idx]
        ea, _ = ece_mce(sample_a, n_bins=n_bins)
        eb, _ = ece_mce(sample_b, n_bins=n_bins)
        means[i] = ea - eb
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def brier_from_row(r, engine: bool, market: str) -> float:
    prefix = "eng" if engine else "mkt"
    if market == "hda":
        p_h = r[f"{prefix}_h"]
        p_d = r[f"{prefix}_d"]
        p_a = r[f"{prefix}_a"]
        y_h = 1.0 if r["actual_hda"] == "home" else 0.0
        y_d = 1.0 if r["actual_hda"] == "draw" else 0.0
        y_a = 1.0 if r["actual_hda"] == "away" else 0.0
        return (p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2
    else:
        p_o = r[f"{prefix}_o"]
        p_u = r[f"{prefix}_u"]
        y_o = 1.0 if r["actual_ou"] == "over" else 0.0
        y_u = 1.0 if r["actual_ou"] == "under" else 0.0
        return (p_o - y_o) ** 2 + (p_u - y_u) ** 2


def print_reliability(name, pairs, n_bins=10):
    centers, mp, mo, cnt = reliability_bins(pairs, n_bins=n_bins)
    print(f"  {name}")
    print(f"  {'bin center':>10s} | {'n':>4s} | {'pred':>6s} | {'obs':>6s} | "
          f"{'gap':>6s}")
    for c, p, o, k in zip(centers, mp, mo, cnt):
        if k == 0:
            continue
        gap = o - p
        print(f"  {c:>10.2f} | {k:>4d} | {p:>6.3f} | {o:>6.3f} | {gap:>+6.3f}")


def main() -> None:
    matches = e14.load_full_season_with_goals()
    print("=== Orbita: reliability + calibration audit ===")
    print(f"Panel : {len(matches)} matches (full 2024/25 EPL)")
    print(f"Trials: {e05.N_TRIALS}/match  | drag = {BEST_CD_OU}")
    print()

    if CACHE_PATH.exists():
        print(f"Loading cached posteriors from {CACHE_PATH}", flush=True)
        with CACHE_PATH.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for k in list(r.keys()):
                if k in ("match_id", "date", "event",
                          "actual_hda", "actual_ou"):
                    continue
                r[k] = float(r[k])
            r["match_id"] = int(r["match_id"])
    else:
        print("--- Scoring anisotropic baseline + saving posteriors ---",
              flush=True)
        rows = score_and_save(matches, CACHE_PATH)

    hda_e, ou_e = collect_pairs(rows, engine=True)
    hda_m, ou_m = collect_pairs(rows, engine=False)

    print()
    print("=== HDA reliability curves ===")
    print_reliability("ENGINE HDA", hda_e)
    print()
    print_reliability("MARKET HDA", hda_m)
    print()
    print("=== O/U reliability curves ===")
    print_reliability("ENGINE O/U", ou_e)
    print()
    print_reliability("MARKET O/U", ou_m)
    print()

    ece_e_h, mce_e_h = ece_mce(hda_e)
    ece_m_h, mce_m_h = ece_mce(hda_m)
    ece_e_o, mce_e_o = ece_mce(ou_e)
    ece_m_o, mce_m_o = ece_mce(ou_m)

    print("=== Calibration errors (lower is better) ===")
    print(f"HDA  engine  ECE={ece_e_h:.4f}  MCE={mce_e_h:.4f}")
    print(f"HDA  market  ECE={ece_m_h:.4f}  MCE={mce_m_h:.4f}")
    print(f"     delta ECE = engine − market = {ece_e_h - ece_m_h:+.4f}")
    print(f"O/U  engine  ECE={ece_e_o:.4f}  MCE={mce_e_o:.4f}")
    print(f"O/U  market  ECE={ece_m_o:.4f}  MCE={mce_m_o:.4f}")
    print(f"     delta ECE = engine − market = {ece_e_o - ece_m_o:+.4f}")
    print()

    print("=== Bootstrap 90% CI on delta ECE (engine − market) ===")
    d_h, lo_h, hi_h = bootstrap_ci_delta_ece(hda_e, hda_m)
    d_o, lo_o, hi_o = bootstrap_ci_delta_ece(ou_e, ou_m)
    v_h = ("engine BETTER calibrated" if hi_h < 0 else
           "market BETTER calibrated" if lo_h > 0 else "tied (CI ∋ 0)")
    v_o = ("engine BETTER calibrated" if hi_o < 0 else
           "market BETTER calibrated" if lo_o > 0 else "tied (CI ∋ 0)")
    print(f"HDA  d={d_h:+.4f}  CI=[{lo_h:+.4f}, {hi_h:+.4f}]  {v_h}")
    print(f"O/U  d={d_o:+.4f}  CI=[{lo_o:+.4f}, {hi_o:+.4f}]  {v_o}")
    print()

    print("=== Brier sanity check on cached rows ===")
    b_eng_h = np.mean([brier_from_row(r, True, "hda") for r in rows])
    b_mkt_h = np.mean([brier_from_row(r, False, "hda") for r in rows])
    b_eng_o = np.mean([brier_from_row(r, True, "ou") for r in rows])
    b_mkt_o = np.mean([brier_from_row(r, False, "ou") for r in rows])
    print(f"HDA  engine  Brier={b_eng_h:.4f}")
    print(f"HDA  market  Brier={b_mkt_h:.4f}  delta={b_eng_h - b_mkt_h:+.4f}")
    print(f"O/U  engine  Brier={b_eng_o:.4f}")
    print(f"O/U  market  Brier={b_mkt_o:.4f}  delta={b_eng_o - b_mkt_o:+.4f}")


if __name__ == "__main__":
    main()

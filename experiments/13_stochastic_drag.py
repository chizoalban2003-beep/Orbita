"""13_stochastic_drag.py — Ornstein-Uhlenbeck noise on the drag coefficient.

Point estimate drag treats every match as if intangibles evolve
deterministically. Real match dynamics have persistent random shocks:
a red card 63' spikes fatigue on one side; a substitution 78' relaxes
pressing intensity for 5 minutes then re-tightens.

An OU process is the simplest physically-motivated way to model that
noise structure. Given long-run mean `C_mean`, reversion rate `theta`,
and volatility `sigma`, the coefficient wanders around `C_mean` with
autocorrelation timescale `1/theta`.

For the H/D/L event-space we hold x-drag at 0 (from experiment 10) and
put OU noise only on y-drag around the mean 0.16 (experiment 11 winner).

We compare three configurations on the 50-match panel:
  A) constant anisotropic: (0.00, 0.16) — reproduces experiment 11.
  B) OU low-noise:  theta=0.5/DURATION,  sigma=0.05
  C) OU high-noise: theta=1.0/DURATION,  sigma=0.15

Higher sigma diffuses the drag more; higher theta reverts to mean faster.
If OU noise beats deterministic on the same 50-match panel, we have a
real robustness effect (each MC trial samples a different drag realisation
— the posterior is naturally hedged).

Run with::

    PYTHONPATH=src python3 experiments/13_stochastic_drag.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orbita import EventSpace, ornstein_uhlenbeck_schedule  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e05", ROOT / "experiments" / "05_player_attractor_panel.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)

e05.N_TRIALS = 30
SEED = 20260701

CONFIGS = {
    "constant (0.00, 0.16)": {"kind": "const", "C": np.array([0.0, 0.16])},
    "OU low-noise":  {"kind": "ou", "C_mean": np.array([0.0, 0.16]),
                      "theta": 0.5 / e05.DURATION, "sigma": 0.05},
    "OU high-noise": {"kind": "ou", "C_mean": np.array([0.0, 0.16]),
                      "theta": 1.0 / e05.DURATION, "sigma": 0.15},
}


def score_config(matches, cfg, base_seed) -> tuple:
    """Return (hda_briers, ou_briers) arrays for Config B under cfg."""
    hda = []
    ou = []
    for m_i, m in enumerate(matches):
        wells_m = e05.multimarket_wells(
            m["home_label"], m["away_label"],
            m["p_h"], m["p_d"], m["p_a"],
            m["p_over"], m["p_under"],
        )
        space_m = EventSpace(wells_m)
        rng = np.random.default_rng(seed=e05.SEED)
        f_pos, f_mass, _ = e05._stack(space_m)
        o_pos, o_mass, o_labels = e05._stack(space_m)
        acc = {l: 0.0 for l in o_labels}
        q_scale = np.array([0.3, 0.2]) * e05.IC_SCALE
        p_scale = np.array([0.15, 0.15]) * e05.IC_SCALE
        for trial in range(e05.N_TRIALS):
            q0 = rng.normal(scale=q_scale)
            p0 = rng.normal(scale=p_scale)
            if cfg["kind"] == "const":
                C_d = cfg["C"]
            else:
                # Per-trial OU realisation seeded deterministically.
                C_d = ornstein_uhlenbeck_schedule(
                    cfg["C_mean"], cfg["theta"], cfg["sigma"], e05.DT,
                    seed=base_seed + m_i * 10000 + trial,
                )
            q_end = e05.fast_final_q(f_pos, f_mass, q0, p0,
                                    duration=e05.DURATION, dt=e05.DT, C_d=C_d)
            probs = e05.fast_posterior(q_end, o_pos, o_mass, o_labels,
                                       alpha=e05.SOFT_ALPHA)
            for label, p in probs.items():
                acc[label] += p
        total = sum(acc.values())
        joint = {l: v / total for l, v in acc.items()}
        hda_post = e05.marginal_hda(joint, m["home_label"], m["away_label"])
        ou_post = e05.marginal_ou(joint)
        hda.append(e05.brier(hda_post, m["actual_hda"]))
        ou.append(e05.brier(ou_post, m["actual_ou"]))
    return np.array(hda), np.array(ou)


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000):
    rng = np.random.default_rng(SEED)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(deltas, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(means.mean()), float(lo), float(hi)


def main() -> None:
    matches = e05.load_matches_with_ou()
    print("=== Orbita: stochastic OU drag on Config B ===")
    print(f"Panel : {len(matches)} matches (2024/25 EPL Oct–Nov)")
    print(f"Trials: {e05.N_TRIALS}/match (per-trial OU realisation)")
    print()

    mkt_hda = []
    mkt_ou = []
    for m in matches:
        p_mkt_hda = {m["home_label"]: m["p_h"], "draw": m["p_d"],
                     m["away_label"]: m["p_a"]}
        p_mkt_ou = {"over": m["p_over"], "under": m["p_under"]}
        mkt_hda.append(e05.brier(p_mkt_hda, m["actual_hda"]))
        mkt_ou.append(e05.brier(p_mkt_ou, m["actual_ou"]))
    mkt_hda = np.array(mkt_hda)
    mkt_ou = np.array(mkt_ou)
    print(f"Market: HDA={mkt_hda.mean():.4f}  O/U={mkt_ou.mean():.4f}")
    print()

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"--- Scoring: {name} ---", flush=True)
        hda, ou = score_config(matches, cfg, base_seed=SEED)
        results[name] = (hda, ou)
        print(f"  engine HDA={hda.mean():.4f}  O/U={ou.mean():.4f}")
        print()

    print("=== summary + bootstrap 90% CIs (engine − market) ===")
    for name, (hda, ou) in results.items():
        d_hda = hda - mkt_hda
        d_ou = ou - mkt_ou
        h_mean, h_lo, h_hi = bootstrap_ci(d_hda)
        o_mean, o_lo, o_hi = bootstrap_ci(d_ou)
        h_v = ("engine BEATS" if h_hi < 0 else
               "market BEATS" if h_lo > 0 else "tied (CI ∋ 0)")
        o_v = ("engine BEATS" if o_hi < 0 else
               "market BEATS" if o_lo > 0 else "tied (CI ∋ 0)")
        print(f"{name}")
        print(f"  HDA d={h_mean:+.4f}  CI=[{h_lo:+.4f}, {h_hi:+.4f}]  {h_v}")
        print(f"  O/U d={o_mean:+.4f}  CI=[{o_lo:+.4f}, {o_hi:+.4f}]  {o_v}")


if __name__ == "__main__":
    main()

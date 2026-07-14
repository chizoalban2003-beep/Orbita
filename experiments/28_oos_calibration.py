"""28_oos_calibration.py — out-of-sample robustness of the lever calibrations.

exp26 fit the three lever magnitudes (red_card 0.145, injury 0.126,
early_pressure 0.205) on six big-5 leagues (E0,E1,D1,SP1,I1,F1). Are those
constants real physics, or artifacts of those leagues? This refits each lever
FROM A FLAT PRIOR on a disjoint held-out set — Netherlands, Belgium, Portugal,
Turkey, Greece (N1,B1,P1,T1,G1) — and checks whether the in-sample constant
falls inside the independent held-out posterior.

This is out-of-sample across LEAGUES (not across data-source or instrument): same
football-data.co.uk feed, same natural experiments (HR/AR cards, Pinnacle
open->close drift, narrow HT leads), different competitions. A pass means the
magnitudes generalise beyond the leagues they were tuned on; it says nothing
about beating the market (the campaign settled that separately).

Run:  PYTHONPATH=src python3 experiments/28_oos_calibration.py
      env: ORBITA_OOS_DIVS (default N1,B1,P1,T1,G1), ORBITA_NCAL (default 80)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OOS_DIVS = os.environ.get("ORBITA_OOS_DIVS", "N1,B1,P1,T1,G1")
os.environ["ORBITA_DIVS"] = OOS_DIVS          # exp26 loaders read this at import
NCAL = int(os.environ.get("ORBITA_NCAL", 80))
NTR = int(os.environ.get("ORBITA_NTRIALS", 60))

from orbita.calibrate import (LeverPosterior, _default_grid,  # noqa: E402
                              calibrate_from_matches)

# reuse exp26's natural-experiment loaders (its main() is guarded, not run)
_spec = importlib.util.spec_from_file_location(
    "exp26", ROOT / "experiments" / "26_precalibrate.py")
_exp26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp26)

INSAMPLE = {"red_card": 0.145, "injury": 0.126, "early_pressure": 0.205}
LOADERS = {"red_card": _exp26.load_redcards,
           "injury": _exp26.load_injury_drift,
           "early_pressure": _exp26.load_early_pressure}


def flat_prior(lever: str) -> LeverPosterior:
    g = _default_grid(lever)
    return LeverPosterior(lever, "magnitude", g, np.zeros_like(g), "FLAT (OOS)", 0)


def main():
    print(f"OUT-OF-SAMPLE lever calibration on {OOS_DIVS} "
          f"(held out from exp26's big-5)\n")
    print(f"  {'lever':<15}{'in-samp':>9}{'OOS mean':>10}{'OOS 90% CI':>20}"
          f"{'n':>6}  verdict")
    rng = np.random.default_rng(0)
    for lever in ("red_card", "injury", "early_pressure"):
        ms = LOADERS[lever]()
        if len(ms) > NCAL:
            ms = [ms[i] for i in rng.choice(len(ms), NCAL, replace=False)]
        post = calibrate_from_matches(lever, ms, start=flat_prior(lever),
                                      n_trials=NTR)
        lo, hi = post.ci()
        ins = INSAMPLE[lever]
        verdict = ("REPLICATES (in-samp in CI)" if lo <= ins <= hi
                   else "DRIFTS (in-samp outside CI)")
        print(f"  {lever:<15}{ins:>9.3f}{post.mean():>10.3f}"
              f"{f'[{lo:.3f},{hi:.3f}]':>20}{len(ms):>6}  {verdict}")
    print("\n(flat-prior fit on leagues never used in the original calibration)")


if __name__ == "__main__":
    main()

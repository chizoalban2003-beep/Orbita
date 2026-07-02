"""12_inplay_kick.py — In-play event injection via a momentum kick.

Static pre-match models can only re-price at halftime by re-running
a Bayesian update. The engine's natural mechanism is different: an
in-play event is a *physical* perturbation to the body's momentum,
and the remaining minutes are simply the deterministic (or
stochastic) evolution of the perturbed state.

This example:

  1. Builds a 3-well H/D/A event-space priced to Arsenal (H) 55%,
     draw 25%, Chelsea (A) 20%.
  2. Runs the pre-match simulation to minute 60 with anisotropic
     drag (0.00, 0.16) — the empirical best from exp 10/11.
  3. Snapshots ``(q, p)`` at minute 60.
  4. Applies a **goal for the home side** as an instantaneous
     momentum kick toward the home well: ``kick(p, [+0.6, 0.0])``.
  5. Continues the simulation to minute 90 from the perturbed state.
  6. Reports the soft posterior before and after the kick.

Run with::

    PYTHONPATH=src python3 examples/12_inplay_kick.py
"""
from __future__ import annotations

import numpy as np

from orbita import (
    Attractor,
    Body,
    EventSpace,
    final_well_posterior,
    kick,
    simulate,
    simulate_from_state,
)

# Match uses the same simulation-time convention as experiment 05:
# total DURATION=300 (dimensionless) maps to 90 match minutes.
FULL_DURATION = 300.0
MINUTES = FULL_DURATION / 90.0
CD = 0.04  # isotropic — light drag so a kick isn't fully absorbed
           # before the match ends.
DT = 0.1
IC_SCALE = 2.5
N_TRIALS = 30
SEED = 20260702


def _mc_posterior(space, trials, sim_kwargs, ic_start=None,
                   kick_dp=None, kick_t=None):
    """Run N MC trials, return averaged soft posterior.

    If ic_start is given, use (q0, p0) from it. If kick_dp/kick_t are
    given, split the simulation at kick_t, apply the kick, and continue.
    """
    rng = np.random.default_rng(SEED)
    q_scale = np.array([0.3, 0.2]) * IC_SCALE
    p_scale = np.array([0.15, 0.15]) * IC_SCALE
    labels = [a.label for a in space.attractors]
    acc = {l: 0.0 for l in labels}
    for _ in range(trials):
        q0 = rng.normal(scale=q_scale) + np.array([0.0, 0.0])
        p0 = rng.normal(scale=p_scale) + np.array([+0.35, +0.05])
        if kick_t is not None:
            sol_pre = simulate_from_state(
                space, q0=q0, p0=p0, body_mass=1.0,
                duration=kick_t, C_d=CD, dt=DT, n_saves=50,
            )
            q_k = sol_pre["q"][-1]
            p_k = kick(sol_pre["p"][-1], kick_dp)
            sol = simulate_from_state(
                space, q0=q_k, p0=p_k, body_mass=1.0,
                duration=sim_kwargs["duration"] - kick_t,
                C_d=CD, dt=DT, n_saves=50, t_start=kick_t,
            )
        else:
            sol = simulate_from_state(
                space, q0=q0, p0=p0, body_mass=1.0,
                duration=sim_kwargs["duration"], C_d=CD, dt=DT, n_saves=50,
            )
        posterior = final_well_posterior(sol, space)
        for l, p in posterior.items():
            acc[l] += p
    total = sum(acc.values())
    return {l: v / total for l, v in acc.items()}


def main() -> None:
    space = EventSpace([
        Attractor("arsenal_win", [+5.0, 0.0], 0.55),
        Attractor("draw",        [ 0.0, 0.0], 0.25),
        Attractor("chelsea_win", [-5.0, 0.0], 0.20),
    ])

    print(f"=== In-play kick demo (Arsenal vs Chelsea, N={N_TRIALS} trials) ===")
    print(f"Convention: sim time {FULL_DURATION:.0f} = 90 match minutes.")
    print()

    print("--- No kick (control) ---")
    pre = _mc_posterior(space, N_TRIALS,
                         sim_kwargs={"duration": FULL_DURATION})
    for label, p in sorted(pre.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<20s} {p:.3f}")
    print()

    kick_t = 60.0 * MINUTES
    dp_home = np.array([+1.5, 0.0])
    print(f"--- Home goal at 60' (dp={dp_home.tolist()}) ---")
    post = _mc_posterior(space, N_TRIALS,
                          sim_kwargs={"duration": FULL_DURATION},
                          kick_dp=dp_home, kick_t=kick_t)
    for label, p in sorted(post.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<20s} {p:.3f}")
    print()

    print("=== Posterior shift induced by the kick ===")
    for label in sorted(pre):
        delta = post[label] - pre[label]
        print(f"  {label:<20s} {pre[label]:.3f} -> {post[label]:.3f}  "
              f"({delta:+.3f})")


if __name__ == "__main__":
    main()

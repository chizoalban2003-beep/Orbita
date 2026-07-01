"""11_three_axis_event_space.py — n-D event space demo.

As of v0.3.6 the integrator, gravity kernel, and Attractor validation
all accept positions in any dimension >= 2. This example builds a
3D H/D/A × O/U × BTTS event-space with 12 wells and simulates an
orbiting body, showing that the same physics generalises past 2D
without any code changes.

Physical read for the third axis: BTTS_yes = z + 2, BTTS_no = z − 2.
A body drifting +z is expressing that both sides are creating
chances; a body drifting −z is expressing a shut-out dynamic.

Run with::

    python3 examples/11_three_axis_event_space.py
"""
from __future__ import annotations

import numpy as np

from orbita import (
    Attractor,
    Body,
    EventSpace,
    final_well,
    final_well_posterior,
    simulate,
)

WIN_X = 5.0
OVER_Y = 3.0
UNDER_Y = -3.0
BTTS_Z = 2.0


def build_3d_space(p_h: float, p_d: float, p_a: float,
                   p_over: float, p_under: float,
                   p_btts: float, p_no_btts: float,
                   home_label: str, away_label: str) -> EventSpace:
    wells = []
    for x, outcome, p_match in [
        (WIN_X,  home_label, p_h),
        (0.0,    "draw",     p_d),
        (-WIN_X, away_label, p_a),
    ]:
        for y, ou_side, p_ou in [(OVER_Y, "over", p_over),
                                   (UNDER_Y, "under", p_under)]:
            for z, btts_side, p_btts_side in [
                (BTTS_Z,  "btts",    p_btts),
                (-BTTS_Z, "no_btts", p_no_btts),
            ]:
                mass = p_match * p_ou * p_btts_side
                wells.append(Attractor(
                    label=f"{outcome}_{ou_side}_{btts_side}",
                    position=[x, y, z],
                    mass=mass,
                ))
    return EventSpace(wells)


def main() -> None:
    # Arsenal (H) vs Chelsea (A) — pretend priors.
    space = build_3d_space(
        p_h=0.55, p_d=0.25, p_a=0.20,
        p_over=0.55, p_under=0.45,
        p_btts=0.60, p_no_btts=0.40,
        home_label="arsenal_win", away_label="chelsea_win",
    )

    body = Body(
        mass=1.0,
        q0=np.array([0.0, 0.0, 0.0]),
        p0=np.array([0.4, 0.1, 0.2]),  # lean home, slight over, slight btts
    )

    print("=== 3D event space demo: H/D/A × O/U × BTTS ===")
    print(f"Wells       : {len(space.attractors)} (3 × 2 × 2)")
    print(f"Position dim: {space.attractors[0].position.shape[0]}D")
    print()

    sol = simulate(space, body=body, duration=300.0, C_d=np.array([0.0, 0.16, 0.10]))
    q_end = sol["q"][-1]
    print(f"Final position: ({q_end[0]:+.2f}, {q_end[1]:+.2f}, {q_end[2]:+.2f})")
    print(f"Hard nearest well: {final_well(sol, space)}")
    print()
    print("Soft posterior (top 5):")
    posterior = final_well_posterior(sol, space)
    for label, p in sorted(posterior.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {label:<40s} {p:.3f}")


if __name__ == "__main__":
    main()

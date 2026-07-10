"""Durable, append-only prediction ledger — Orbita's training corpus.

The human-in-the-loop repricer is a forward pass: an analyst's private read
becomes an intervention, the engine kinematically turns it into a probability.
This ledger is the *loss layer*. It records each prediction before kickoff and
its outcome after, so the app's real performance against the market can be
measured — and, eventually, so the physics itself can be tuned to the record.

Design principles (why the schema looks the way it does):

* **Event-sourced + immutable.** Two append-only record types: an ``[[entry]]``
  (the prediction) and a later ``[[settlement]]`` (the outcome, referencing the
  entry id). Nothing already written is ever mutated — the file is a clean audit
  log, and git gives it versioned durability. localStorage in the browser tool
  is disposable; this is the permanent record.

* **Replayable / gradient-ready.** Each entry caches the *exact physical state*
  the forecast used — post-intervention well ``masses``, the ``momentum``
  offset, ``C_d``, the ``lock`` spec — plus the full ``constants`` snapshot
  (geometry, softening, alpha, dt, duration, seed, n_trials) and the market
  ``priors``. This is deliberate: final probabilities are the *output* of the
  forward pass and cannot yield a gradient w.r.t. a constant. The state vectors
  are the inputs a differentiable Orbita would compute ∂L/∂k, ∂L/∂M, ∂L/∂C_d
  against. Freezing the constants *per entry* keeps history uncorrupted once the
  constants start evolving. ``scenario``/params are kept for per-lever
  calibration (e.g. "red-card reads under-price momentum by 2% → nudge k").

Read with :func:`load` (stdlib ``tomllib``); appended with a tiny built-in TOML
emitter so the module stays numpy-only.
"""
from __future__ import annotations

import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .forces import SOFTENING
from .interventions import (_C_D, _DURATION, _IC_SCALE, _POS, _apply,
                            _resolve_lock, forecast, Intervention)

ENGINE_VERSION = "0.3.7"
_LAB = ("home", "draw", "away")
_ALPHA = 2.0
_DT = 0.1
DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ledger.toml"

# retrospective magnitude sweep (auto-run at settlement)
_SWEEP_MAX = 0.5          # θ grid ceiling (user spec); early_pressure widened below
_SWEEP_STEP = 0.01
_SWEEP_NTRIALS = 200      # per-θ MC trials; common random numbers keep the curve smooth


# --------------------------------------------------------------------------
# tiny TOML emitter (controlled schema: scalars, arrays, inline tables)
# --------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        v = round(v, 8)
        s = repr(v)
        if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
            s += ".0"
        return s
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} = {_fmt(val)}" for k, val in v.items()) + "}"
    raise TypeError(f"unserializable: {type(v)}")


def _emit(array_name: str, d: Dict) -> str:
    lines = [f"[[{array_name}]]"]
    lines += [f"{k} = {_fmt(v)}" for k, v in d.items()]
    return "\n".join(lines) + "\n\n"


def _append(path: Path, array_name: str, d: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_emit(array_name, d))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def devig(odds: Dict[str, float]) -> Dict[str, float]:
    inv = {k: 1.0 / float(odds[k]) for k in _LAB}
    s = sum(inv.values())
    return {k: inv[k] / s for k in _LAB}


def brier(probs: Dict[str, float], result: str) -> float:
    return float(sum((probs[k] - (1.0 if k == result else 0.0)) ** 2 for k in _LAB))


def _physical_state(priors, iv, seed, n_trials) -> Dict:
    """The exact state vectors the forecast used — the gradient-ready block."""
    post = _apply(priors, iv) if iv else dict(priors)
    masses = [float(post[k]) for k in _LAB]
    mom = (iv.momentum.tolist() if (iv and iv.momentum is not None)
           else [0.0, 0.0])
    lk = _resolve_lock(priors, iv)
    lock = ({"on": True, "pos": [float(x) for x in lk[0]], "strength": float(lk[1])}
            if lk else {"on": False})
    C_d = _C_D * (iv.drag_scale if iv else 1.0)
    constants = {
        "well_home": [float(x) for x in _POS["home"]],
        "well_draw": [float(x) for x in _POS["draw"]],
        "well_away": [float(x) for x in _POS["away"]],
        "softening": float(SOFTENING), "alpha": _ALPHA, "ic_scale": _IC_SCALE,
        "dt": _DT, "duration": _DURATION, "seed": int(seed),
        "n_trials": int(n_trials), "engine_version": ENGINE_VERSION,
    }
    return {"masses": masses, "momentum": [float(x) for x in mom],
            "C_d": float(C_d), "lock": lock, "constants": constants}


def project(market: Dict[str, float], engine_base: Dict[str, float],
            engine_cf: Dict[str, float]) -> Dict[str, float]:
    """The priced line: apply the engine's directional intervention DELTA to the
    market's calibrated marginals. The market owns the marginals (exp 22-24); the
    engine contributes only the shift (exp 25). Using the raw engine forecast
    instead would pay a marginal-distortion penalty unrelated to the read.
    Clipped and renormalised to a valid distribution."""
    raw = {k: max(0.0, market[k] + (engine_cf[k] - engine_base[k])) for k in _LAB}
    s = sum(raw.values()) or 1.0
    return {k: raw[k] / s for k in _LAB}


def _logged_theta(entry_id: str, path: Path = DEFAULT_PATH) -> float:
    """The magnitude the analyst actually logged for an entry (for MLE contrast)."""
    e = next((e for e in load(path)["entry"] if e["id"] == entry_id), None)
    if e is None:
        return float("nan")
    r = e["read"]
    for k in ("severity", "pressure", "strength"):
        if k in r:
            return float(r[k])
    return float("nan")


def _tv(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total-variation distance between two H/D/A distributions, in [0,1]."""
    return 0.5 * sum(abs(float(p[k]) - float(q[k])) for k in _LAB)


def information_alpha(market_open: Dict[str, float], market_close: Dict[str, float],
                     projection: Dict[str, float]) -> float:
    """Iα = d(open, orbita) − d(close, orbita), total-variation.

    Positive ⇒ the market drifted *toward* our priced line by kickoff — sharp
    money validating the private read independently of the 90-minute result.
    This isolates execution edge (did we beat the pricing mechanism?) from
    single-sample outcome variance (a 90th-minute penalty ruining one Brier)."""
    return _tv(market_open, projection) - _tv(market_close, projection)


def magnitude_sweep(entry: Dict, result: str, *, n_trials: int = _SWEEP_NTRIALS,
                    step: float = _SWEEP_STEP, grid_max: Optional[float] = None
                    ) -> Optional[Dict]:
    """Retrospective θ-likelihood profile for a settled entry.

    Re-runs the *same* row's projection across a θ grid, holding the market
    marginals and the entry's seed fixed (common random numbers ⇒ smooth curve),
    and scores each against the realised ``result``. Returns two aligned arrays:

    * ``likelihood`` = 1 − Brier_orbita(θ)  — the display/MLE surface (argmax =
      the magnitude that would have minimised Brier for this match).
    * ``prob_outcome`` = P_projection(result | θ) — the proper per-match
      likelihood the Bayesian/JAX fold multiplies across matches.

    θ=0 is the market anchor (no-op intervention ⇒ projection==market), so a
    curve decaying monotonically to 0 means "the lever should not have fired".
    Returns ``None`` for baseline (no-read) rows and unknown levers."""
    from .calibrate import make_iv_for
    lever = entry["read"].get("scenario", "none")
    if lever in ("none", ""):
        return None
    side = entry["read"].get("side") or "home"
    try:
        mk = make_iv_for(lever, side)
    except ValueError:
        return None
    if grid_max is None:                       # don't clip the momentum lever's range
        grid_max = 1.0 if lever == "early_pressure" else _SWEEP_MAX
    priors = entry["market"]["priors"]
    seed = int(entry["state"]["constants"].get("seed", 42))
    base = forecast(priors, None, n_trials=n_trials, seed=seed)   # θ-independent: once
    n = int(round(grid_max / step)) + 1
    grid = [round(i * step, 4) for i in range(n)]
    like, pout = [], []
    for t in grid:
        iv = mk(float(t))
        cf = forecast(priors, iv, n_trials=n_trials, seed=seed) if iv else dict(base)
        proj = project(priors, base, cf)
        like.append(round(1.0 - brier(proj, result), 6))
        pout.append(round(float(proj[result]), 6))
    mle = grid[int(np.argmax(like))]
    return {"lever": lever, "side": side, "result": result,
            "grid_min": 0.0, "grid_max": float(grid_max), "grid_step": float(step),
            "n_trials": int(n_trials), "mle_theta": float(mle),
            "likelihood": like, "prob_outcome": pout}


def _read_summary(iv: Optional[Intervention], side: str) -> Dict:
    scenario = iv.name.split(":")[0] if iv else "none"
    out = {"scenario": scenario, "side": side,
           "description": iv.description if iv else "market baseline (no read)"}
    if iv and iv.mass_transfer is not None:
        out["severity"] = float(iv.mass_transfer[2])
    if iv and iv.momentum is not None:
        out["pressure"] = float(np.linalg.norm(iv.momentum))
    if iv and iv.lock is not None:
        out["strength"] = float(iv.lock[1])
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def log_read(match: str, odds: Dict[str, float], iv: Optional[Intervention],
             *, side: str = "", kickoff: str = "", source: str = "manual",
             seed: int = 42, n_trials: int = 300,
             path: Path = DEFAULT_PATH) -> str:
    """Append an immutable prediction. Runs the base and counterfactual
    forecasts, freezes the physical state, returns the entry id."""
    priors = devig(odds)
    base = forecast(priors, None, n_trials=n_trials, seed=seed)
    cf = forecast(priors, iv, n_trials=n_trials, seed=seed) if iv else dict(base)
    proj = project(priors, base, cf) if iv else dict(priors)
    slug = "".join(c if c.isalnum() else "-" for c in match.lower())[:32].strip("-")
    entry_id = f"{_now()}-{slug}"
    entry = {
        "id": entry_id, "logged_at": _now(), "match": match,
        "kickoff": kickoff, "status": "open",
        "market": {"odds": {k: float(odds[k]) for k in _LAB},
                   "priors": {k: round(float(priors[k]), 6) for k in _LAB},
                   "source": source},
        "read": _read_summary(iv, side),
        "state": _physical_state(priors, iv, seed, n_trials),
        "forecast": {"engine_base": {k: round(float(base[k]), 6) for k in _LAB},
                     "engine_cf": {k: round(float(cf[k]), 6) for k in _LAB},
                     "projection": {k: round(float(proj[k]), 6) for k in _LAB}},
    }
    _append(path, "entry", entry)
    return entry_id


def settle(entry_id: str, result: str, *, score: str = "",
           close_odds: Optional[Dict[str, float]] = None, sweep: bool = True,
           sweep_ntrials: int = _SWEEP_NTRIALS, path: Path = DEFAULT_PATH) -> Dict:
    """Append an immutable settlement for a logged entry and return its metrics.

    Always emits the retrospective :func:`magnitude_sweep` (``sweep=True``) so
    every settlement — not just the memorable ones — writes its exact
    θ-likelihood profile into the corpus (no selective-observation bias). Pass
    ``close_odds`` (the closing 1X2 line) to also log Information-Alpha."""
    if result not in _LAB:
        raise ValueError(f"result must be one of {_LAB}, got {result!r}")
    data = load(path)
    entry = next((e for e in data["entry"] if e["id"] == entry_id), None)
    if entry is None:
        raise KeyError(f"no entry with id {entry_id!r}")
    mkt = entry["market"]["priors"]
    fc = entry["forecast"]
    b_mkt = brier(mkt, result)
    b_base = brier(fc["engine_base"], result)
    b_orb = brier(fc["projection"], result)          # the priced line = market + engine delta
    rec = {"entry_id": entry_id, "result": result, "score": score,
           "brier_market": round(b_mkt, 6), "brier_base": round(b_base, 6),
           "brier_orbita": round(b_orb, 6),
           "edge_vs_market": round(b_mkt - b_orb, 6), "settled_at": _now()}
    if close_odds is not None:
        close = devig(close_odds)
        rec["market_close"] = {k: round(float(close[k]), 6) for k in _LAB}
        rec["information_alpha"] = round(
            information_alpha(mkt, close, fc["projection"]), 6)
    if sweep:
        sw = magnitude_sweep(entry, result, n_trials=sweep_ntrials)
        if sw is not None:
            rec["magnitude_sweep"] = sw
    _append(path, "settlement", rec)
    return rec


def import_entries(entries, path: Path = DEFAULT_PATH) -> int:
    """Fold browser-exported entries (from the Artifact) into the durable
    ledger, appending only ids not already present. The Artifact computes the
    same gradient-ready schema client-side, so imported entries are faithful,
    not lossy. Returns the number newly appended."""
    have = {e["id"] for e in load(path)["entry"]}
    n = 0
    for e in entries:
        eid = e.get("id")
        if not eid or eid in have:
            continue
        _append(path, "entry", e)
        have.add(eid)
        n += 1
    return n


def load(path: Path = DEFAULT_PATH) -> Dict:
    if not Path(path).exists():
        return {"entry": [], "settlement": []}
    with Path(path).open("rb") as fh:
        d = tomllib.load(fh)
    d.setdefault("entry", [])
    d.setdefault("settlement", [])
    return d


def report(path: Path = DEFAULT_PATH) -> Dict:
    """Join settlements to entries; aggregate overall + per-scenario edge."""
    data = load(path)
    settled = {s["entry_id"]: s for s in data["settlement"]}
    rows = [(e, settled[e["id"]]) for e in data["entry"] if e["id"] in settled]
    agg: Dict[str, Dict] = {}
    for e, s in rows:
        for key in ("__all__", e["read"]["scenario"]):
            a = agg.setdefault(key, {"n": 0, "mkt": 0.0, "orb": 0.0, "base": 0.0,
                                     "win": 0, "ia": 0.0, "n_ia": 0})
            a["n"] += 1
            a["mkt"] += s["brier_market"]; a["orb"] += s["brier_orbita"]
            a["base"] += s["brier_base"]
            a["win"] += 1 if s["brier_orbita"] < s["brier_market"] else 0
            if "information_alpha" in s:
                a["ia"] += s["information_alpha"]; a["n_ia"] += 1
    for a in agg.values():
        if a["n"]:
            a["mkt"] /= a["n"]; a["orb"] /= a["n"]; a["base"] /= a["n"]
            a["edge"] = a["mkt"] - a["orb"]
            a["info_alpha"] = a["ia"] / a["n_ia"] if a["n_ia"] else None
    return {"n_entries": len(data["entry"]), "n_settled": len(rows), "agg": agg}


def _print_report(path: Path = DEFAULT_PATH) -> None:
    r = report(path)
    print(f"Ledger: {r['n_entries']} logged, {r['n_settled']} settled  ({path})")
    if not r["n_settled"]:
        print("  (nothing settled yet — log reads before kickoff, settle after)")
        return
    print(f"\n  {'scenario':<14}{'n':>4}{'Brier mkt':>11}{'Brier orb':>11}"
          f"{'edge':>9}{'win%':>7}{'InfoA':>9}")
    order = ["__all__"] + sorted(k for k in r["agg"] if k != "__all__")
    for k in order:
        a = r["agg"][k]
        name = "ALL" if k == "__all__" else k
        ia = f"{a['info_alpha']:+.4f}" if a.get("info_alpha") is not None else "   —"
        print(f"  {name:<14}{a['n']:>4}{a['mkt']:>11.4f}{a['orb']:>11.4f}"
              f"{a['edge']:>+9.4f}{a['win']/a['n']*100:>6.0f}%{ia:>9}")
    e = r["agg"]["__all__"]["edge"]
    print(f"\n  {'human+tool BEATS market (edge>0, small sample)' if e > 0 else 'human+tool trails the market — the honest default'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv=None):
    import argparse
    from . import interventions as itv

    p = argparse.ArgumentParser(description="Orbita durable prediction ledger")
    sub = p.add_subparsers(dest="cmd", required=True)

    lg = sub.add_parser("log", help="log a prediction before kickoff")
    lg.add_argument("--match", required=True)
    lg.add_argument("--home", type=float, required=True)
    lg.add_argument("--draw", type=float, required=True)
    lg.add_argument("--away", type=float, required=True)
    lg.add_argument("--scenario", choices=["injury", "red", "low", "early", "none"],
                    default="none")
    lg.add_argument("--side", choices=["home", "away"], default="home")
    lg.add_argument("--amount", type=float, default=0.2,
                    help="severity / pressure / lock strength")
    lg.add_argument("--kickoff", default="")

    st = sub.add_parser("settle", help="settle a logged entry")
    st.add_argument("--id", required=True)
    st.add_argument("--result", choices=list(_LAB), required=True)
    st.add_argument("--score", default="")
    st.add_argument("--close-home", type=float, help="closing line (for Info-Alpha)")
    st.add_argument("--close-draw", type=float)
    st.add_argument("--close-away", type=float)
    st.add_argument("--no-sweep", action="store_true",
                    help="skip the retrospective magnitude sweep")

    im = sub.add_parser("import", help="fold Artifact-exported entries (JSON) into the ledger")
    im.add_argument("--file", required=True, help="JSON file exported from the repricer tool")

    sub.add_parser("report", help="show performance vs the market")

    args = p.parse_args(argv)
    if args.cmd == "log":
        iv = None
        if args.scenario == "injury":
            iv = itv.injury(args.side, args.amount)
        elif args.scenario == "red":
            iv = itv.red_card(args.side, args.amount)
        elif args.scenario == "low":
            iv = itv.low_tempo(args.amount)
        elif args.scenario == "early":
            iv = itv.early_pressure(args.side, args.amount)
        eid = log_read(args.match, {"home": args.home, "draw": args.draw, "away": args.away},
                       iv, side=args.side, kickoff=args.kickoff)
        print(f"logged {eid}")
    elif args.cmd == "settle":
        close = None
        if None not in (args.close_home, args.close_draw, args.close_away):
            close = {"home": args.close_home, "draw": args.close_draw,
                     "away": args.close_away}
        rec = settle(args.id, args.result, score=args.score, close_odds=close,
                     sweep=not args.no_sweep)
        print(f"settled: market {rec['brier_market']:.4f}  orbita {rec['brier_orbita']:.4f}  "
              f"edge {rec['edge_vs_market']:+.4f}")
        if "magnitude_sweep" in rec:
            sw = rec["magnitude_sweep"]
            print(f"  sweep: MLE θ̂ = {sw['mle_theta']:.2f}  "
                  f"(logged read used {_logged_theta(rec['entry_id']):.2f})")
        if "information_alpha" in rec:
            ia = rec["information_alpha"]
            verdict = "market drifted TOWARD us" if ia > 0 else "market drifted away"
            print(f"  Info-Alpha: {ia:+.4f}  ({verdict})")
    elif args.cmd == "import":
        import json
        raw = json.loads(Path(args.file).read_text())
        entries = raw["entry"] if isinstance(raw, dict) else raw
        n = import_entries(entries)
        print(f"imported {n} new entr{'y' if n == 1 else 'ies'} (of {len(entries)})")
    elif args.cmd == "report":
        _print_report()


if __name__ == "__main__":
    _main()

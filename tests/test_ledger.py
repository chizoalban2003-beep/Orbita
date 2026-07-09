"""Durable append-only prediction ledger."""
import tomllib

from orbita import ledger
from orbita.interventions import injury, red_card, low_tempo


ODDS = {"home": 2.0, "draw": 3.7, "away": 4.35}


def test_entry_is_replayable_and_gradient_ready(tmp_path):
    p = tmp_path / "l.toml"
    eid = ledger.log_read("Test A", ODDS, injury("home", 0.25), side="home", path=p)
    d = ledger.load(p)
    assert len(d["entry"]) == 1
    e = d["entry"][0]
    assert e["id"] == eid
    st = e["state"]
    # the physical state vectors are cached (not just probabilities)
    assert len(st["masses"]) == 3 and abs(sum(st["masses"]) - 1.0) < 1e-6
    pri = e["market"]["priors"]
    assert st["masses"][0] < pri["home"]          # home cut by the injury transfer
    assert st["masses"][2] > pri["away"]          # opponent gained
    # full constants snapshot present for replay / future gradients
    for k in ("softening", "alpha", "dt", "duration", "seed", "n_trials",
              "well_home", "engine_version"):
        assert k in st["constants"]
    assert "base" in e["forecast"] and "counterfactual" in e["forecast"]
    assert e["read"]["scenario"] == "injury" and e["read"]["severity"] == 0.25


def test_settle_computes_brier_and_edge(tmp_path):
    p = tmp_path / "l.toml"
    eid = ledger.log_read("Test B", ODDS, injury("home", 0.3), side="home", path=p)
    rec = ledger.settle(eid, "away", path=p)
    e = ledger.load(p)["entry"][0]
    assert rec["brier_market"] == round(ledger.brier(e["market"]["priors"], "away"), 6)
    assert rec["brier_orbita"] == round(ledger.brier(e["forecast"]["counterfactual"], "away"), 6)
    assert rec["edge_vs_market"] == round(rec["brier_market"] - rec["brier_orbita"], 6)


def test_append_only_immutable(tmp_path):
    p = tmp_path / "l.toml"
    e1 = ledger.log_read("M1", ODDS, red_card("away"), side="away", path=p)
    ledger.log_read("M2", ODDS, low_tempo(), path=p)
    before = p.read_text()
    ledger.settle(e1, "home", path=p)
    after = p.read_text()
    # settling only appends; the earlier text is untouched
    assert after.startswith(before)
    assert after.count("[[entry]]") == 2 and after.count("[[settlement]]") == 1


def test_report_aggregates_per_scenario(tmp_path):
    p = tmp_path / "l.toml"
    a = ledger.log_read("A", ODDS, injury("home", 0.3), side="home", path=p)
    b = ledger.log_read("B", ODDS, red_card("away"), side="away", path=p)
    ledger.settle(a, "away", path=p)
    ledger.settle(b, "home", path=p)
    r = ledger.report(p)
    assert r["n_settled"] == 2
    assert r["agg"]["__all__"]["n"] == 2
    assert "injury" in r["agg"] and "red_card" in r["agg"]


def test_emitted_toml_is_valid(tmp_path):
    p = tmp_path / "l.toml"
    ledger.log_read("Round Trip", ODDS, low_tempo(0.2), path=p)
    # parses cleanly with the stdlib reader — no emitter corruption
    with p.open("rb") as fh:
        tomllib.load(fh)

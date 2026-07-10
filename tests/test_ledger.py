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
    assert all(k in e["forecast"] for k in ("engine_base", "engine_cf", "projection"))
    # projection = market + engine delta: home cut moves the priced line off market
    assert e["forecast"]["projection"]["away"] > e["market"]["priors"]["away"]
    assert e["read"]["scenario"] == "injury" and e["read"]["severity"] == 0.25


def test_settle_computes_brier_and_edge(tmp_path):
    p = tmp_path / "l.toml"
    eid = ledger.log_read("Test B", ODDS, injury("home", 0.3), side="home", path=p)
    rec = ledger.settle(eid, "away", path=p)
    e = ledger.load(p)["entry"][0]
    assert rec["brier_market"] == round(ledger.brier(e["market"]["priors"], "away"), 6)
    assert rec["brier_orbita"] == round(ledger.brier(e["forecast"]["projection"], "away"), 6)
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


def test_import_entries_dedups(tmp_path):
    src = tmp_path / "src.toml"
    ledger.log_read("Imported", ODDS, red_card("away"), side="away", path=src)
    entry = ledger.load(src)["entry"][0]
    dst = tmp_path / "dst.toml"
    assert ledger.import_entries([entry], path=dst) == 1        # appended
    assert ledger.import_entries([entry], path=dst) == 0        # deduped by id
    d = ledger.load(dst)
    assert len(d["entry"]) == 1 and d["entry"][0]["id"] == entry["id"]


def test_emitted_toml_is_valid(tmp_path):
    p = tmp_path / "l.toml"
    ledger.log_read("Round Trip", ODDS, low_tempo(0.2), path=p)
    # parses cleanly with the stdlib reader — no emitter corruption
    with p.open("rb") as fh:
        tomllib.load(fh)


def test_magnitude_sweep_is_auto_written_and_anchored(tmp_path):
    p = tmp_path / "l.toml"
    eid = ledger.log_read("Sweep", ODDS, injury("home", 0.25), side="home",
                          n_trials=80, path=p)
    rec = ledger.settle(eid, "away", sweep_ntrials=80, path=p)
    sw = rec["magnitude_sweep"]
    grid = [round(i * sw["grid_step"], 4) for i in range(len(sw["likelihood"]))]
    # aligned arrays over the full grid, both surfaces present
    assert len(sw["likelihood"]) == len(sw["prob_outcome"]) == len(grid)
    assert sw["lever"] == "injury" and sw["result"] == "away"
    # θ=0 is the market anchor: L(0) == 1 - Brier_market exactly
    assert abs(sw["likelihood"][0] - (1.0 - rec["brier_market"])) < 1e-6
    # MLE is the argmax of the likelihood surface, and is on the grid
    assert sw["mle_theta"] == grid[max(range(len(grid)), key=lambda i: sw["likelihood"][i])]
    # survives the tomllib round-trip (long inline array inside the settlement)
    d = ledger.load(p)
    assert len(d["settlement"][0]["magnitude_sweep"]["likelihood"]) == len(grid)


def test_sweep_mle_decays_to_zero_when_read_was_wrong(tmp_path):
    # transfer home->away, but home actually wins: more transfer only hurts,
    # so the likelihood decays and the MLE pins at the bottom of the grid
    # (θ=0.00/0.01 are a statistical tie, so allow the noise neighbour).
    p = tmp_path / "l.toml"
    eid = ledger.log_read("Wrong", ODDS, injury("home", 0.3), side="home",
                          n_trials=80, path=p)
    sw = ledger.settle(eid, "home", sweep_ntrials=80, path=p)["magnitude_sweep"]
    assert sw["mle_theta"] <= 0.02
    assert sw["likelihood"][0] > sw["likelihood"][-1]


def test_information_alpha_sign(tmp_path):
    p = tmp_path / "l.toml"
    # read pushes home->away; a close that SHORTENS away (raises away prob) and
    # lengthens home is a drift TOWARD our priced line ⇒ Iα > 0.
    eid = ledger.log_read("IA", ODDS, injury("home", 0.3), side="home",
                          n_trials=80, path=p)
    toward = ledger.settle(eid, "away", close_odds={"home": 2.4, "draw": 3.7, "away": 3.6},
                           sweep=False, path=p)
    assert toward["information_alpha"] > 0
    # no closing line => no Info-Alpha key
    eid2 = ledger.log_read("noIA", ODDS, injury("home", 0.3), side="home",
                           n_trials=80, path=p)
    plain = ledger.settle(eid2, "away", sweep=False, path=p)
    assert "information_alpha" not in plain


def test_baseline_row_has_no_sweep(tmp_path):
    p = tmp_path / "l.toml"
    eid = ledger.log_read("Flat", ODDS, None, path=p)
    rec = ledger.settle(eid, "home", path=p)
    assert "magnitude_sweep" not in rec

"""Tests for the Understat live adapter (GitHub issue #9).

Tests run entirely offline against a saved HTML fixture; no network
access is required. The fixture is a real Understat page snapshot for
match 26694 (Newcastle United 1-0 Arsenal, 2024-11-02, EPL) — chosen
because it's a recent EPL fixture with a clear home win and non-trivial
priors (home was the underdog).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orbita.providers import (
    UnderstatMatch,
    UnderstatMatchProvider,
    _slug,
)

FIXTURE = Path(__file__).parent / "fixtures" / "understat_match_26694.html"


def _load_fixture(match_id: str) -> str:
    return FIXTURE.read_text(encoding="utf-8")


# ---- _slug ---------------------------------------------------------------

def test_slug_lowercases_and_underscores() -> None:
    assert _slug("Newcastle United") == "newcastle_united"


def test_slug_handles_ampersand() -> None:
    assert _slug("Brighton & Hove Albion") == "brighton_and_hove_albion"


def test_slug_strips_trailing_separators() -> None:
    assert _slug("  Arsenal!  ") == "arsenal"


# ---- fetch_match (offline via html_loader) -------------------------------

@pytest.fixture
def provider() -> UnderstatMatchProvider:
    return UnderstatMatchProvider(html_loader=_load_fixture)


def test_fetch_match_returns_understat_match(provider) -> None:
    m = provider.fetch_match("26694")
    assert isinstance(m, UnderstatMatch)


def test_fetch_match_id_and_date(provider) -> None:
    m = provider.fetch_match("26694")
    assert m.match_id == "26694"
    assert m.date == "2024-11-02"


def test_fetch_match_sport_is_soccer(provider) -> None:
    m = provider.fetch_match("26694")
    assert m.sport == "soccer"


def test_fetch_match_labels_are_slugged(provider) -> None:
    m = provider.fetch_match("26694")
    assert m.side_a_label == "newcastle_united_win"
    assert m.side_b_label == "arsenal_win"
    assert m.draw_label == "draw"


def test_fetch_match_priors_sum_to_one(provider) -> None:
    m = provider.fetch_match("26694")
    total = m.prior_a + m.prior_b + m.prior_draw
    assert total == pytest.approx(1.0, abs=1e-2)


def test_fetch_match_actual_is_home_win(provider) -> None:
    # Newcastle 1, Arsenal 0 → home win.
    m = provider.fetch_match("26694")
    assert m.actual == m.side_a_label
    assert m.h_goals == 1
    assert m.a_goals == 0


def test_fetch_match_carries_xg(provider) -> None:
    m = provider.fetch_match("26694")
    assert m.h_xg > 0
    assert m.a_xg > 0


def test_as_backtest_dict_has_expected_keys(provider) -> None:
    m = provider.fetch_match("26694")
    d = m.as_backtest_dict()
    assert set(d) == {
        "sport", "event", "date",
        "side_a_label", "side_b_label", "draw_label",
        "prior_a", "prior_b", "prior_draw",
        "actual",
    }


def test_as_backtest_dict_actual_matches_a_label(provider) -> None:
    """The 'actual' field must be one of the three labels — this is what
    the backtest harness asserts."""
    m = provider.fetch_match("26694")
    d = m.as_backtest_dict()
    assert d["actual"] in {d["side_a_label"], d["side_b_label"], d["draw_label"]}


# ---- parser robustness ---------------------------------------------------

def test_parse_match_info_raises_when_blob_missing() -> None:
    prov = UnderstatMatchProvider(html_loader=lambda mid: "<html>no json here</html>")
    with pytest.raises(ValueError, match="match_info"):
        prov.fetch_match("0")


def test_disk_cache_round_trip(tmp_path) -> None:
    """If html_loader is None and a cached file exists, it is read
    directly without any network call."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "match_26694.html").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    prov = UnderstatMatchProvider(cache_dir=cache_dir)
    m = prov.fetch_match("26694")
    assert m.match_id == "26694"

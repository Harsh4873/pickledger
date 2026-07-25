"""Smoke tests for the in-house tennis model.

The properties asserted here are the ones that would silently produce a
plausible-looking but wrong model: as-of feature construction (no leakage),
rating-update symmetry, the published Weighted-Elo margin weights, prediction
antisymmetry, and the shape of the emitted picks (which the tennis grader
matches on).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TennisPredictionModel.tennis_core import (  # noqa: E402
    ANTISYMMETRIC_FEATURES,
    FEATURE_NAMES,
    INITIAL_RATING,
    SYMMETRIC_FEATURES,
    EloConfig,
    RatingEngine,
    elo_probability,
    mirror_vector,
    symmetrise,
    to_vector,
)
from TennisPredictionModel.tennis_data import (  # noqa: E402
    Match,
    _normalise_series,
    _repair_date,
    player_key,
)
from TennisPredictionModel.tennis_infer import TennisModel  # noqa: E402


def make_match(
    *,
    date: str = "2024-05-01",
    winner: str = "Alpha A.",
    loser: str = "Bravo B.",
    winner_games: int = 12,
    loser_games: int = 6,
    surface: str = "Clay",
    status: str = "completed",
    best_of: int = 3,
    tour: str = "ATP",
) -> Match:
    return Match(
        date=date,
        tour=tour,
        season=int(date[:4]),
        tournament="Test Open",
        location="Testville",
        series="ATP250",
        tier=2,
        court="Outdoor",
        surface=surface,
        round="1st Round",
        round_order=1,
        best_of=best_of,
        winner=winner,
        loser=loser,
        winner_key=player_key(winner),
        loser_key=player_key(loser),
        winner_rank=10,
        loser_rank=20,
        winner_points=2000,
        loser_points=1000,
        winner_games=winner_games,
        loser_games=loser_games,
        winner_sets=2,
        loser_sets=0,
        status=status,
    )


# --------------------------------------------------------------------------
# player identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("archive_form", "display_form"),
    [
        ("Alcaraz C.", "Carlos Alcaraz"),
        ("Ugo Carabelli C.", "Camilo Ugo Carabelli"),
        ("Auger-Aliassime F.", "Felix Auger-Aliassime"),
        ("Bautista Agut R.", "Roberto Bautista Agut"),
        ("De Minaur A.", "Alex De Minaur"),
    ],
)
def test_player_key_joins_archive_and_espn_spellings(archive_form: str, display_form: str) -> None:
    """The daily slate is ESPN display names; the ratings are archive names."""
    assert player_key(archive_form) == player_key(display_form)
    assert player_key(archive_form)


def test_player_key_folds_spelling_variants_but_separates_players() -> None:
    assert player_key("Ferrero J.") == player_key("Ferrero J.C.")
    assert player_key("van Lottum J.") == player_key("Van Lottum J.")
    assert player_key("Zverev A.") != player_key("Zverev M.")
    assert player_key("Williams S.") != player_key("Williams V.")


# --------------------------------------------------------------------------
# source-data repair
# --------------------------------------------------------------------------


def test_repair_date_keeps_legitimate_season_spill() -> None:
    # The tour season opens in the last days of December.
    assert _repair_date("2013-12-30", 2014) == "2013-12-30"
    assert _repair_date("2014-06-01", 2014) == "2014-06-01"


def test_repair_date_fixes_typo_years() -> None:
    # The 2026 Iasi Open final is stamped 2029 in the archive.
    assert _repair_date("2029-07-20", 2026) == "2026-07-20"
    assert _repair_date("", 2026) == ""


def test_normalise_series_folds_tier_typos() -> None:
    assert _normalise_series("WTA253") == "WTA250"
    assert _normalise_series("WTA1000") == "WTA1000"
    assert _normalise_series("Grand Slam") == "Grand Slam"


# --------------------------------------------------------------------------
# rating engine
# --------------------------------------------------------------------------


def test_features_are_as_of_and_never_see_the_match_being_predicted() -> None:
    """The first meeting must look like a blank slate, and the replay must
    emit the row before folding the result in."""
    engine = RatingEngine()
    matches = [
        make_match(date="2024-05-01"),
        make_match(date="2024-05-08"),
    ]
    records = engine.replay(matches)
    assert len(records) == 2
    first, second = records
    # Nothing was known before the first match.
    assert first["features"]["elo_diff"] == 0.0
    assert first["features"]["h2h_matches"] == 0.0
    # The second row sees exactly one prior meeting, not two.
    assert second["features"]["h2h_matches"] == 1.0
    assert second["features"]["elo_diff"] != 0.0


def test_replay_emits_before_updating_state() -> None:
    engine = RatingEngine()
    match = make_match()
    records = engine.replay([match])
    emitted = records[0]["features"]["elo_diff"]
    after = engine.players[match.winner_key].elo - engine.players[match.loser_key].elo
    assert emitted == 0.0
    assert after > 0.0


def test_elo_update_is_zero_sum_between_equally_experienced_players() -> None:
    engine = RatingEngine()
    match = make_match()
    engine.update(match)
    winner = engine.players[match.winner_key]
    loser = engine.players[match.loser_key]
    assert winner.elo - INITIAL_RATING == pytest.approx(INITIAL_RATING - loser.elo)
    assert winner.welo - INITIAL_RATING == pytest.approx(INITIAL_RATING - loser.welo)


@pytest.mark.parametrize(
    ("winner_games", "loser_games", "expected_weight"),
    [
        (12, 0, 1.00),   # 6-0 6-0
        (14, 12, 0.54),  # 7-6 7-6
        (14, 18, 0.44),  # 0-6 7-6 7-6
    ],
)
def test_welo_margin_weight_matches_the_published_examples(
    winner_games: int, loser_games: int, expected_weight: float
) -> None:
    """Table 1 of Angelini, Candila & De Angelis (EJOR 297(1), 2022)."""
    engine = RatingEngine()
    match = make_match(winner_games=winner_games, loser_games=loser_games)
    engine.update(match)
    winner = engine.players[match.winner_key]
    scale = engine.config.k_scale / (engine.config.k_offset ** engine.config.k_shape)
    observed_weight = (winner.welo - INITIAL_RATING) / (scale * 0.5)
    assert observed_weight == pytest.approx(expected_weight, abs=0.01)


def test_walkovers_never_move_the_ratings() -> None:
    engine = RatingEngine()
    engine.update(make_match(status="walkover"))
    assert not engine.players


def test_retirements_rate_with_a_neutral_margin() -> None:
    """Someone did win, but a truncated scoreline says nothing about dominance."""
    engine = RatingEngine()
    engine.update(make_match(status="retired", winner_games=12, loser_games=2))
    winner = engine.players[player_key("Alpha A.")]
    scale = engine.config.k_scale / (engine.config.k_offset ** engine.config.k_shape)
    assert (winner.welo - INITIAL_RATING) / (scale * 0.5) == pytest.approx(0.5, abs=0.01)


def test_surface_ratings_are_independent() -> None:
    engine = RatingEngine()
    engine.replay([make_match(surface="Clay"), make_match(date="2024-06-01", surface="Clay")])
    winner = engine.players[player_key("Alpha A.")]
    assert winner.surface_elo["Clay"] > INITIAL_RATING
    assert winner.surface_elo["Grass"] == INITIAL_RATING


def test_snapshot_round_trips_through_json() -> None:
    engine = RatingEngine(EloConfig(surface_blend=0.62))
    engine.replay([make_match(), make_match(date="2024-06-01")])
    restored = RatingEngine.from_snapshot(json.loads(json.dumps(engine.snapshot())))
    assert restored.config.surface_blend == pytest.approx(0.62)
    assert restored.last_date == engine.last_date
    for key, state in engine.players.items():
        assert restored.players[key].elo == pytest.approx(state.elo, abs=1e-3)
        assert restored.players[key].matches == state.matches
    assert restored.h2h == engine.h2h


def test_prune_keeps_active_players_and_drops_stale_ones() -> None:
    engine = RatingEngine()
    engine.replay([
        make_match(date="2019-05-01", winner="Old A.", loser="Older B."),
        make_match(date="2026-05-01", winner="Alpha A.", loser="Bravo B."),
    ])
    summary = engine.prune(active_within_days=365)
    assert summary["players_after"] == 2
    assert player_key("Alpha A.") in engine.players
    assert player_key("Old A.") not in engine.players


# --------------------------------------------------------------------------
# symmetry
# --------------------------------------------------------------------------


def test_feature_lists_partition_cleanly() -> None:
    assert FEATURE_NAMES == ANTISYMMETRIC_FEATURES + SYMMETRIC_FEATURES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_mirroring_negates_only_the_antisymmetric_block() -> None:
    engine = RatingEngine()
    engine.replay([make_match()])
    forward = engine.features(
        player_key("Alpha A."), player_key("Bravo B."),
        surface="Clay", best_of=3, tier=2, round_order=1, indoor=False,
        tour="ATP", ordinal=739000, event_key="2024:ATP:Test Open",
    )
    vector = to_vector(forward)
    mirrored = mirror_vector(vector)
    for index, name in enumerate(FEATURE_NAMES):
        if name in ANTISYMMETRIC_FEATURES:
            assert mirrored[index] == pytest.approx(-vector[index])
        else:
            assert mirrored[index] == pytest.approx(vector[index])


def test_swapping_the_players_negates_the_rating_gaps() -> None:
    engine = RatingEngine()
    engine.replay([make_match()])
    common = dict(
        surface="Clay", best_of=3, tier=2, round_order=1, indoor=False,
        tour="ATP", ordinal=739000, event_key="2024:ATP:Test Open",
    )
    forward = engine.features(player_key("Alpha A."), player_key("Bravo B."), **common)
    reverse = engine.features(player_key("Bravo B."), player_key("Alpha A."), **common)
    for name in ANTISYMMETRIC_FEATURES:
        assert forward[name] == pytest.approx(-reverse[name])


def test_symmetrise_makes_the_two_sides_sum_to_one() -> None:
    assert symmetrise(0.7, 0.4) == pytest.approx(0.65)
    assert symmetrise(0.7, 0.4) + symmetrise(0.4, 0.7) == pytest.approx(1.0)


def test_elo_probability_is_the_standard_logistic() -> None:
    assert elo_probability(1500, 1500) == pytest.approx(0.5)
    assert elo_probability(1900, 1500) == pytest.approx(1.0 / (1.0 + 10 ** -1.0))
    assert elo_probability(1500, 1900) + elo_probability(1900, 1500) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# exported scorer
# --------------------------------------------------------------------------


def _linear_artifact() -> dict:
    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[FEATURE_NAMES.index("elo_diff")] = 1.0
    return {
        "kind": "linear",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": coefficients,
        "intercept": 0.0,
        "means": [0.0] * len(FEATURE_NAMES),
        "scales": [1.0] * len(FEATURE_NAMES),
        "calibration": {"x": [0.0, 0.5, 1.0], "y": [0.05, 0.5, 0.95]},
        "market_blend": {"intercept": 0.0, "model_weight": 0.5, "market_weight": 0.5},
        "metadata": {"model_version": "test"},
    }


def test_exported_scorer_is_antisymmetric() -> None:
    model = TennisModel(_linear_artifact())
    vector = [0.0] * len(FEATURE_NAMES)
    vector[FEATURE_NAMES.index("elo_diff")] = 0.8
    flipped = mirror_vector(vector)
    assert model.predict(vector) + model.predict(flipped) == pytest.approx(1.0, abs=1e-9)


def test_calibration_interpolates_between_knots() -> None:
    model = TennisModel(_linear_artifact())
    assert model.calibrate(0.5) == pytest.approx(0.5)
    assert model.calibrate(0.25) == pytest.approx(0.275)
    # Out-of-range inputs clamp to the end knots rather than extrapolating.
    assert model.calibrate(-1.0) == pytest.approx(0.05)
    assert model.calibrate(2.0) == pytest.approx(0.95)


def test_load_model_refuses_an_artifact_with_a_drifted_feature_contract(tmp_path: Path) -> None:
    """Positional indexing means a stale artifact mispredicts silently."""
    from TennisPredictionModel.tennis_infer import load_model

    payload = _linear_artifact()
    payload["feature_names"] = FEATURE_NAMES[:-1]
    stale = tmp_path / "tennis_model.json"
    stale.write_text(json.dumps(payload), encoding="utf-8")
    assert load_model(stale) is None

    payload["feature_names"] = list(FEATURE_NAMES)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(payload), encoding="utf-8")
    assert load_model(good) is not None


def test_market_blend_falls_back_to_the_model_without_a_price() -> None:
    model = TennisModel(_linear_artifact())
    assert model.blend_with_market(0.62, None) == pytest.approx(0.62)
    blended = model.blend_with_market(0.62, 0.55)
    assert 0.55 < blended < 0.62


def test_shipped_artifact_loads_and_matches_the_feature_contract() -> None:
    """The committed artifact must agree with the code's feature list."""
    from TennisPredictionModel.tennis_infer import load_model

    model = load_model()
    if model is None:
        pytest.skip("tennis artifacts not present in this checkout")
    assert model.feature_names == FEATURE_NAMES
    assert model.metadata.get("model_version")
    probability = model.predict([0.0] * len(FEATURE_NAMES))
    assert 0.0 < probability < 1.0
    # A blank matchup is a coin flip by construction: every difference is zero.
    assert probability == pytest.approx(0.5, abs=0.05)


def test_shipped_ratings_snapshot_is_sane() -> None:
    from TennisPredictionModel.tennis_core import RATINGS_PATH

    engine = RatingEngine.load(RATINGS_PATH)
    if engine is None:
        pytest.skip("tennis ratings snapshot not present in this checkout")
    assert engine.players
    assert engine.last_date
    top = engine.leaderboard(limit=5, tour="ATP")
    assert top and all(row["elo"] > INITIAL_RATING for row in top)
    assert all(state.matches > 0 for state in engine.players.values())


# --------------------------------------------------------------------------
# serving payload
# --------------------------------------------------------------------------


def _espn_payload(status: str = "STATUS_SCHEDULED") -> dict:
    return {
        "events": [
            {
                "name": "Generali Open",
                "venue": {"displayName": "Kitzb\u00fchel, Austria"},
                "groupings": [
                    {
                        "grouping": {"displayName": "Men's Singles"},
                        "competitions": [
                            {
                                "date": "2026-07-24T12:00Z",
                                "round": {"displayName": "Quarterfinals"},
                                "status": {"type": {"name": status}},
                                "competitors": [
                                    {"homeAway": "away", "athlete": {"displayName": "Carlos Alcaraz"}},
                                    {"homeAway": "home", "athlete": {"displayName": "Jannik Sinner"}},
                                ],
                            }
                        ],
                    },
                    {
                        "grouping": {"displayName": "Men's Doubles"},
                        "competitions": [
                            {
                                "date": "2026-07-24T12:00Z",
                                "status": {"type": {"name": status}},
                                "competitors": [{"homeAway": "away"}, {"homeAway": "home"}],
                            }
                        ],
                    },
                ],
            }
        ]
    }


def test_generate_tennis_picks_emits_gradeable_rows() -> None:
    pytest.importorskip("requests")
    from scripts.scrapers.tennis_scraper import is_tennis_pick
    from TennisPredictionModel.tennis_infer import load_model
    from TennisPredictionModel.tennis_model import generate_tennis_picks

    if load_model() is None:
        pytest.skip("tennis artifacts not present in this checkout")

    result = generate_tennis_picks(
        "2026-07-24",
        fetch_json=lambda url: _espn_payload(),
        download=False,
    )
    assert result["ok"] is True
    assert result["picks"], "expected the singles match to be rated"
    assert len(result["picks"]) == 1, "doubles rows must be dropped"

    pick = result["picks"][0]
    assert is_tennis_pick(pick)
    assert pick["sport"] == "Tennis"
    assert pick["selected_player"] in {"Carlos Alcaraz", "Jannik Sinner"}
    assert pick["away_team"] == "Carlos Alcaraz"
    assert pick["home_team"] == "Jannik Sinner"
    assert pick["market_type"] == "tennis_moneyline"
    assert pick["grade_supported"] is True
    assert pick["decision"] in {"BET", "LEAN", "PASS"}
    assert 0.5 <= pick["probability"] <= 1.0
    # Unpriced by design: nothing may imply an edge against a price we never saw.
    assert pick["odds"] is None
    assert pick["edge"] is None
    assert pick["pricing_type"] == "unpriced"
    assert pick["pick"].startswith(pick["selected_player"])


def test_event_key_uses_the_archive_tournament_name() -> None:
    """Serving must build the same event key the replay stamped.

    The replay keys on the archive's tournament name; ESPN publishes the
    current sponsor name. Keying off ESPN's would make every event look new and
    silently zero `event_games_diff` across the whole slate.
    """
    from datetime import date

    from TennisPredictionModel.tennis_core import RatingEngine
    from TennisPredictionModel.tennis_model import _tournament_meta

    engine = RatingEngine()
    engine.replay([
        make_match(date="2026-07-20", winner="Alpha A.", loser="Bravo B.", winner_games=12, loser_games=6),
    ])
    archive_key = "2026:ATP:Test Open"
    alpha = engine.players[player_key("Alpha A.")]
    assert alpha.event_key == archive_key
    assert alpha.event_games == 18

    index = {
        "tournaments": {"ATP|test open": {**_clay("Test Open", "Testville"), "date": "2026-07-20"}},
        "locations": {"ATP|testville": {**_clay("Test Open", "Testville"), "date": "2026-07-20"}},
        "week_surface": {},
    }
    meta = _tournament_meta(index, "ATP", "Sponsor Cup presented by Someone", date(2026, 7, 24), "Testville, Nowhere")
    assert meta["tournament"] == "Test Open"
    resolved_key = f"2026:ATP:{meta['tournament']}"
    assert resolved_key == archive_key

    engine.features(
        player_key("Alpha A."), player_key("Bravo B."),
        surface="Clay", best_of=3, tier=2, round_order=5, indoor=False, tour="ATP",
        ordinal=date(2026, 7, 24).toordinal(), event_key=resolved_key,
    )
    assert alpha.event_games == 18, "the resolved key must not reset the event counters"


def test_generate_tennis_picks_reports_an_unresolved_slate() -> None:
    pytest.importorskip("requests")
    from TennisPredictionModel.tennis_infer import load_model
    from TennisPredictionModel.tennis_model import generate_tennis_picks

    if load_model() is None:
        pytest.skip("tennis artifacts not present in this checkout")

    def failing_fetch(url: str):
        raise RuntimeError("boom")

    result = generate_tennis_picks("2026-07-24", fetch_json=failing_fetch, download=False)
    assert result["ok"] is False
    assert "did not resolve" in result["error"]


def _clay(name: str, location: str, tour: str = "ATP") -> dict:
    return {
        "date": "2025-05-01", "surface": "Clay", "tier": 2, "series": "ATP250",
        "court": "Outdoor", "best_of": 3, "tour": tour, "tournament": name, "location": location,
    }


def test_tournament_metadata_resolves_by_name_then_venue_then_containment() -> None:
    from datetime import date

    from TennisPredictionModel.tennis_model import _tournament_meta

    index = {
        "tournaments": {
            "ATP|estoril open": _clay("Estoril Open", "Estoril"),
            "WTA|internazionali femminili di palermo": _clay("Internazionali Femminili di Palermo", "Palermo", "WTA"),
        },
        "locations": {
            "ATP|estoril": _clay("Estoril Open", "Estoril"),
            "WTA|palermo": _clay("Internazionali Femminili di Palermo", "Palermo", "WTA"),
        },
    }
    # ESPN's "Millennium Estoril Open" finds the archive's "Estoril Open".
    assert _tournament_meta(index, "ATP", "Millennium Estoril Open", date(2026, 5, 1))["surface"] == "Clay"
    # A full sponsor rename resolves only through the venue city — this is the
    # case that a name-only index gets wrong, and it is not cosmetic: the wrong
    # surface feeds the surface-Elo features.
    resolved = _tournament_meta(index, "WTA", "Palermo Ladies Open", date(2026, 7, 24), "Palermo, Italy")
    assert resolved["surface"] == "Clay"
    assert not resolved.get("assumed")
    # Accents in the ESPN venue must not defeat the join.
    index["locations"]["ATP|kitzbuhel"] = _clay("Generali Open", "Kitzbuhel")
    assert _tournament_meta(index, "ATP", "Unknown Cup", date(2026, 7, 24), "Kitzbühel, Austria")["surface"] == "Clay"


def test_tournament_metadata_calendar_fallback_knows_the_swings() -> None:
    from datetime import date

    from TennisPredictionModel.tennis_model import _tournament_meta

    empty: dict = {"tournaments": {}, "locations": {}, "week_surface": {}}
    # Grass is a narrow window between the French Open and Wimbledon…
    assert _tournament_meta(empty, "ATP", "New Event", date(2026, 6, 25))["surface"] == "Grass"
    # …and the two weeks after Wimbledon the tour is back on clay (Hamburg,
    # Palermo, Kitzbühel), which a month-granularity "July means grass" rule
    # gets wrong. The archive says week 30 is 60% clay.
    assert _tournament_meta(empty, "ATP", "New Event", date(2026, 7, 24))["surface"] == "Clay"
    assert _tournament_meta(empty, "ATP", "New Event", date(2026, 5, 10))["surface"] == "Clay"
    assert _tournament_meta(empty, "ATP", "New Event", date(2026, 2, 10))["surface"] == "Hard"
    assert _tournament_meta(empty, "ATP", "New Event", date(2026, 2, 10))["assumed"] is True
    # A measured table in the committed index overrides the built-in default.
    measured = {**empty, "week_surface": {"30": "Hard"}}
    assert _tournament_meta(measured, "ATP", "New Event", date(2026, 7, 24))["surface"] == "Hard"


def test_best_of_is_not_inherited_across_tours() -> None:
    """Men play five sets at the Slams; women never do."""
    from datetime import date

    from TennisPredictionModel.tennis_model import _tournament_meta

    slam = {**_clay("Wimbledon", "London"), "best_of": 5, "surface": "Grass"}
    index = {"tournaments": {"ATP|wimbledon": slam}, "locations": {"ATP|london": slam}}
    resolved = _tournament_meta(index, "WTA", "Wimbledon", date(2026, 7, 1))
    assert resolved["surface"] == "Grass"
    assert resolved["best_of"] == 3


def test_round_order_reads_espn_round_names() -> None:
    from TennisPredictionModel.tennis_model import _round_order

    assert _round_order("Final") == 7
    assert _round_order("Semifinals") == 6
    assert _round_order("Quarterfinals") == 5
    assert _round_order("Round of 32") == 3
    assert _round_order("Qualifying 1st Round") == 0
    assert _round_order("something unparseable") == 2


def test_backtest_kelly_and_devig_helpers() -> None:
    from TennisPredictionModel.tennis_backtest import kelly_stake
    from TennisPredictionModel.tennis_train import no_vig, shin_no_vig

    # A fair two-way book with no margin devigs to its own implied prices.
    assert no_vig(2.0, 2.0) == pytest.approx(0.5)
    assert shin_no_vig(2.0, 2.0) == pytest.approx(0.5, abs=1e-6)
    # With margin, Shin shades the favourite higher than proportional devig,
    # which is the favourite-longshot correction.
    assert shin_no_vig(1.2, 4.5) > no_vig(1.2, 4.5)
    # No edge, no stake.
    assert kelly_stake(0.4, 2.0) == 0.0
    assert 0.0 < kelly_stake(0.6, 2.0) <= 1.0


def test_metrics_agree_with_hand_computed_values() -> None:
    from TennisPredictionModel.tennis_train import brier, log_loss

    assert brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))

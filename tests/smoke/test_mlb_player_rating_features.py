"""Regression coverage for the isolated MLB player-lineup research layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "MLBPredictionModel"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from backtest_player_rating_oracle import _walk_forward_folds
from player_rating_oracle_data import PlayerRatingOracleDatasetBuilder
from player_rating_features import (
    LEAGUE_BATTING_RATING,
    lineup_batting_rating,
    lineup_batting_rating_from_boxscore,
    matchup_player_rating_features,
    pregame_batting_stat,
)


def _batting_stat(
    *,
    pa: int = 100,
    ab: int = 90,
    hits: int = 28,
    doubles: int = 6,
    triples: int = 0,
    home_runs: int = 4,
    walks: int = 8,
    hbp: int = 1,
    total_bases: int = 46,
    games: int = 24,
) -> dict[str, int]:
    return {
        "plateAppearances": pa,
        "atBats": ab,
        "hits": hits,
        "doubles": doubles,
        "triples": triples,
        "homeRuns": home_runs,
        "baseOnBalls": walks,
        "hitByPitch": hbp,
        "totalBases": total_bases,
        "sacFlies": 1,
        "sacBunts": 0,
        "runs": 10,
        "rbi": 12,
        "gamesPlayed": games,
    }


def _add_stat(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    keys = set(left) | set(right)
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in keys}


def _lineup(player: dict[str, int], *, star_at: int = 0) -> list[dict[str, object]]:
    ordinary = _batting_stat()
    players = [{"pregame_batting": ordinary} for _ in range(9)]
    players[star_at] = {"pregame_batting": player}
    return players


def test_current_game_subtraction_recovers_the_pregame_line():
    pregame = _batting_stat(pa=100, ab=90, hits=30, doubles=7, home_runs=5, total_bases=52)
    game = _batting_stat(pa=4, ab=4, hits=2, doubles=1, home_runs=0, walks=0, hbp=0, total_bases=3, games=1)
    season_after = _add_stat(pregame, game)

    recovered = pregame_batting_stat(season_after, game)

    for key in ("plateAppearances", "atBats", "hits", "doubles", "homeRuns", "totalBases"):
        assert recovered[key] == pregame[key]
    assert recovered["gamesPlayed"] == pregame["gamesPlayed"]


def test_one_plate_appearance_extreme_is_heavily_shrunk():
    moonshot = _batting_stat(
        pa=1,
        ab=1,
        hits=1,
        doubles=0,
        home_runs=1,
        walks=0,
        hbp=0,
        total_bases=4,
        games=1,
    )
    lineup = lineup_batting_rating(
        [{"pregame_batting": moonshot} for _ in range(9)], source="test"
    )

    # A single HR cannot turn a lineup into an elite one; it stays near the
    # league prior after empirical-Bayes shrinkage.
    assert LEAGUE_BATTING_RATING < lineup["batting_rating"] < 0.35
    assert lineup["reliability"] < 0.1


def test_batting_order_weights_reward_putting_the_strong_hitter_first():
    star = _batting_stat(
        pa=500,
        ab=420,
        hits=155,
        doubles=35,
        home_runs=35,
        walks=70,
        hbp=5,
        total_bases=295,
        games=120,
    )
    first = lineup_batting_rating(_lineup(star, star_at=0), source="test")
    ninth = lineup_batting_rating(_lineup(star, star_at=8), source="test")

    assert first["available"] == 1.0
    assert first["batting_rating"] > ninth["batting_rating"]


def test_home_away_matchup_features_reverse_cleanly():
    strong = lineup_batting_rating(
        _lineup(_batting_stat(pa=450, ab=390, hits=145, home_runs=32, total_bases=270)),
        source="test",
    )
    weak = lineup_batting_rating(
        _lineup(_batting_stat(pa=450, ab=410, hits=95, home_runs=8, total_bases=140)),
        source="test",
    )

    forward = matchup_player_rating_features(strong, weak)
    reverse = matchup_player_rating_features(weak, strong)

    assert forward["lineup_batting_rating_adv"] == -reverse["lineup_batting_rating_adv"]
    assert forward["lineup_power_rating_adv"] == -reverse["lineup_power_rating_adv"]
    assert forward["lineup_rating_reliability_adv"] == -reverse["lineup_rating_reliability_adv"]


def test_missing_batting_order_is_unavailable_not_a_roster_fallback():
    result = lineup_batting_rating_from_boxscore(
        {"players": {"ID1": {"seasonStats": {"batting": _batting_stat()}}}},
        {},
        subtract_current_game=False,
        source="test",
    )

    assert result["available"] == 0.0
    assert result["player_count"] == 0.0
    assert result["coverage"] == 0.0


def test_historical_and_live_pregame_inputs_match_exactly():
    game = _batting_stat(pa=4, ab=4, hits=1, doubles=0, home_runs=0, walks=0, hbp=0, total_bases=1, games=1)
    player_rows = {}
    live_rows = {}
    for player_id in range(1, 10):
        pregame = _batting_stat(pa=100 + player_id, hits=25 + player_id)
        player_rows[f"ID{player_id}"] = {
            "seasonStats": {"batting": _add_stat(pregame, game)},
            "stats": {"batting": game},
        }
        live_rows[f"ID{player_id}"] = {"seasonStats": {"batting": pregame}, "stats": {"batting": {}}}
    historic = lineup_batting_rating_from_boxscore(
        {"players": player_rows, "battingOrder": list(range(1, 10))},
        {},
        subtract_current_game=True,
        source="historic",
    )
    live = lineup_batting_rating_from_boxscore(
        {"players": live_rows, "battingOrder": list(range(1, 10))},
        {},
        subtract_current_game=False,
        source="live",
    )

    for key in ("batting_rating", "power_rating", "reliability", "coverage", "available"):
        assert historic[key] == live[key]


def test_oracle_folds_never_mix_a_first_pitch_day_between_train_and_validation():
    rows = []
    for day in range(1, 11):
        for game in range(2):
            rows.append(
                {
                    "game_pk": day * 10 + game,
                    "game_start_utc": f"2026-04-{day:02d}T{12 + game}:00:00+00:00",
                    "home_win": game % 2,
                }
            )
    frame = pd.DataFrame(rows)
    folds = _walk_forward_folds(frame, folds=2, initial_train_fraction=0.6)

    assert len(folds) == 2
    for train, validation, _ in folds:
        assert train["game_start_utc"].max() < validation["game_start_utc"].min()


def test_oracle_builder_deduplicates_repeated_schedule_game_pks():
    class FakeClient:
        def __init__(self):
            self.prefetched: list[int] = []

        def get_schedule_for_season(self, _season):
            return [
                {"game_id": "123", "status": "Final", "game_datetime": "2026-04-01T18:00:00Z"},
                {"game_id": "123", "status": "Final", "game_datetime": "2026-04-01T18:00:00Z"},
            ]

        def prefetch_game_feeds(self, game_pks, max_workers):
            self.prefetched = list(game_pks)

    builder = object.__new__(PlayerRatingOracleDatasetBuilder)
    builder.client = FakeClient()
    games = builder._season_games(2026)

    assert [game["game_pk"] for game in games] == [123]
    assert builder.client.prefetched == [123]

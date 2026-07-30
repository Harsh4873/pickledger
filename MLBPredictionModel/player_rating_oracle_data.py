"""Strict historical data builder for the MLB player-lineup research test.

StatsAPI's historical game feeds retain the lineup and starter that ultimately
played, but not a timestamped lineup announcement.  This module therefore
builds an explicitly *oracle-lineup* dataset: player and team statistics are
updated only after each game's first-pitch group is scored, while lineup and
starter identity are known retrospectively.  It is useful for measuring the
upside of a player layer, never as evidence that it is safe to publish live.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from market_mechanics import remove_vig
from mlb_api import HistoricalOddsArchive, StatsAPIClient
from player_rating_features import (
    RATING_SCHEMA,
    lineup_batting_rating,
    matchup_player_rating_features,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_PATH = DATA_DIR / "mlb_player_rating_oracle_dataset.csv"
ORACLE_DATASET_SCHEMA = "mlb_player_rating_oracle_dataset_v1"

_BATTING_UPDATE_FIELDS = (
    "runs",
    "doubles",
    "triples",
    "homeRuns",
    "strikeOuts",
    "baseOnBalls",
    "hits",
    "hitByPitch",
    "atBats",
    "stolenBases",
    "plateAppearances",
    "totalBases",
    "rbi",
    "sacBunts",
    "sacFlies",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-", "-.--"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _innings_to_float(value: Any) -> float:
    if value in (None, "", "-", "-.--"):
        return 0.0
    whole, _, remainder = str(value).partition(".")
    innings = _as_float(whole)
    if remainder == "1":
        innings += 1.0 / 3.0
    elif remainder == "2":
        innings += 2.0 / 3.0
    return innings


def _plate_appearances(stat: Mapping[str, Any]) -> float:
    pa = _as_float(stat.get("plateAppearances"))
    if pa > 0:
        return pa
    return max(
        0.0,
        _as_float(stat.get("atBats"))
        + _as_float(stat.get("baseOnBalls"))
        + _as_float(stat.get("hitByPitch"))
        + _as_float(stat.get("sacFlies"))
        + _as_float(stat.get("sacBunts")),
    )


def _final_status(game: Mapping[str, Any]) -> bool:
    status = str(game.get("status") or "").lower()
    return "final" in status or "completed early" in status


def _team_name(meta: Mapping[str, Any]) -> str:
    return str(meta.get("name") or meta.get("teamName") or "")


def _team_short_name(meta: Mapping[str, Any]) -> str:
    return str(meta.get("abbreviation") or meta.get("teamName") or "").upper()


def _lineup_from_state(
    team_box: Mapping[str, Any],
    batting_state: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    batting_order = list(team_box.get("battingOrder") or [])[:9]
    players = []
    for player_id in batting_order:
        try:
            player_key = int(player_id)
        except (TypeError, ValueError):
            continue
        players.append({"pregame_batting": dict(batting_state.get(player_key) or {})})
    return lineup_batting_rating(players, source="oracle_final_boxscore_lineup")


def _team_profile(state: Mapping[str, float]) -> dict[str, float]:
    games = float(state.get("games", 0.0))
    if games <= 0:
        return {"win_pct": 0.5, "run_diff_per_game": 0.0, "games": 0.0}
    return {
        "win_pct": float(state.get("wins", 0.0)) / games,
        "run_diff_per_game": (
            float(state.get("runs_scored", 0.0)) - float(state.get("runs_allowed", 0.0))
        )
        / games,
        "games": games,
    }


def _starter_era(player_id: int, pitching_state: Mapping[int, Mapping[str, float]]) -> float:
    stat = pitching_state.get(player_id) or {}
    innings = float(stat.get("innings", 0.0))
    if innings <= 0:
        return 4.20
    return 9.0 * float(stat.get("earned_runs", 0.0)) / innings


def _update_batting_state(
    state: dict[int, dict[str, float]],
    team_box: Mapping[str, Any],
) -> None:
    players = team_box.get("players") or {}
    for player_id in team_box.get("batters") or []:
        try:
            normalized_id = int(player_id)
        except (TypeError, ValueError):
            continue
        player = players.get(f"ID{normalized_id}") or {}
        game_batting = (player.get("stats") or {}).get("batting") or {}
        if _plate_appearances(game_batting) <= 0:
            continue
        target = state.setdefault(normalized_id, defaultdict(float))
        for field in _BATTING_UPDATE_FIELDS:
            target[field] = float(target.get(field, 0.0)) + _as_float(game_batting.get(field))


def _update_pitching_state(
    state: dict[int, dict[str, float]],
    team_box: Mapping[str, Any],
) -> None:
    players = team_box.get("players") or {}
    for player_id in team_box.get("pitchers") or []:
        try:
            normalized_id = int(player_id)
        except (TypeError, ValueError):
            continue
        player = players.get(f"ID{normalized_id}") or {}
        game_pitching = (player.get("stats") or {}).get("pitching") or {}
        innings = _innings_to_float(game_pitching.get("inningsPitched"))
        if innings <= 0:
            continue
        target = state.setdefault(normalized_id, defaultdict(float))
        target["innings"] = float(target.get("innings", 0.0)) + innings
        target["earned_runs"] = float(target.get("earned_runs", 0.0)) + _as_float(
            game_pitching.get("earnedRuns")
        )


def _update_team_state(
    state: dict[int, dict[str, float]],
    team_id: int,
    *,
    won: bool,
    runs_scored: int,
    runs_allowed: int,
) -> None:
    target = state.setdefault(team_id, defaultdict(float))
    target["games"] = float(target.get("games", 0.0)) + 1.0
    target["wins"] = float(target.get("wins", 0.0)) + float(won)
    target["runs_scored"] = float(target.get("runs_scored", 0.0)) + float(runs_scored)
    target["runs_allowed"] = float(target.get("runs_allowed", 0.0)) + float(runs_allowed)


class PlayerRatingOracleDatasetBuilder:
    """Build a date-safe player-state dataset from StatsAPI game feeds."""

    def __init__(self, seasons: list[int], *, include_odds: bool = True) -> None:
        self.seasons = sorted(set(seasons))
        self.client = StatsAPIClient()
        self.odds = HistoricalOddsArchive() if include_odds else None

    def _season_games(self, season: int) -> list[dict[str, Any]]:
        # The StatsAPI schedule can repeat a rescheduled game entry.  Keep one
        # gamePk only; otherwise the duplicate would be emitted and would also
        # update the rolling player/team state twice.
        games: list[dict[str, Any]] = []
        seen_game_pks: set[int] = set()
        for game in self.client.get_schedule_for_season(season):
            if not _final_status(game) or not game.get("game_id"):
                continue
            game_pk = int(game["game_id"])
            if game_pk in seen_game_pks:
                continue
            seen_game_pks.add(game_pk)
            games.append(game)
        game_pks = [int(game.get("game_id")) for game in games if game.get("game_id")]
        self.client.prefetch_game_feeds(game_pks, max_workers=24)
        output: list[dict[str, Any]] = []
        for game in games:
            game_pk = int(game.get("game_id"))
            # Schedule timestamps are available before we load a multi-megabyte
            # game feed.  Keeping only this small reference list lets the
            # builder replay multi-season data without retaining thousands of
            # full JSON payloads in memory.
            datetime_text = str(game.get("game_datetime") or game.get("game_date") or "")
            try:
                game_start = pd.Timestamp(datetime.fromisoformat(datetime_text.replace("Z", "+00:00")))
            except ValueError:
                game_start = pd.Timestamp(datetime_text)
            output.append({"game_pk": game_pk, "game_start": game_start})
        return sorted(output, key=lambda game: (game["game_start"], game["game_pk"]))

    def _market_row(
        self,
        game_date: str,
        away_meta: Mapping[str, Any],
        home_meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.odds is None:
            return {}
        odds = self.odds.lookup_moneyline(
            game_date,
            _team_short_name(away_meta),
            _team_short_name(home_meta),
        )
        home_ml = odds.get("home_moneyline")
        away_ml = odds.get("away_moneyline")
        try:
            market_home_probability, _ = remove_vig(int(home_ml), int(away_ml))
        except (TypeError, ValueError):
            market_home_probability = 0.5
        return {
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "market_home_vigfree_prob": float(market_home_probability),
            "market_available": int(home_ml is not None and away_ml is not None),
        }

    def build(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for season in self.seasons:
            batting_state: dict[int, dict[str, float]] = {}
            pitching_state: dict[int, dict[str, float]] = {}
            team_state: dict[int, dict[str, float]] = {}
            games = self._season_games(season)

            # Exact same-start games form one pregame snapshot.  Nothing from
            # one of those games can affect another in the group.
            grouped: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
            for game in games:
                grouped[game["game_start"]].append(game)

            for game_start in sorted(grouped):
                pending_updates: list[dict[str, Any]] = []
                for game in grouped[game_start]:
                    # Fetch one cached feed at a time.  It is released after
                    # this same-start group has been scored and then updates
                    # the rolling state, preserving the no-lookahead contract.
                    payload = self.client.get_game_feed(game["game_pk"])
                    game_data = payload.get("gameData") or {}
                    teams = game_data.get("teams") or {}
                    away_meta = teams.get("away") or {}
                    home_meta = teams.get("home") or {}
                    live = payload.get("liveData") or {}
                    boxes = (live.get("boxscore") or {}).get("teams") or {}
                    away_box = boxes.get("away") or {}
                    home_box = boxes.get("home") or {}
                    line_score = (live.get("linescore") or {}).get("teams") or {}
                    away_score = int(_as_float((line_score.get("away") or {}).get("runs")))
                    home_score = int(_as_float((line_score.get("home") or {}).get("runs")))
                    away_team_id = int(_as_float(away_meta.get("id")))
                    home_team_id = int(_as_float(home_meta.get("id")))
                    away_starter = int(_as_float((away_box.get("pitchers") or [0])[0]))
                    home_starter = int(_as_float((home_box.get("pitchers") or [0])[0]))

                    away_lineup = _lineup_from_state(away_box, batting_state)
                    home_lineup = _lineup_from_state(home_box, batting_state)
                    away_team = _team_profile(team_state.get(away_team_id, {}))
                    home_team = _team_profile(team_state.get(home_team_id, {}))
                    game_date = game_start.date().isoformat()
                    row = {
                        "dataset_schema": ORACLE_DATASET_SCHEMA,
                        "player_rating_schema": RATING_SCHEMA,
                        "oracle_lineup_known": 1,
                        "oracle_starter_known": 1,
                        "lineup_source": "historical_final_boxscore_actual_lineup",
                        "starter_source": "historical_final_boxscore_actual_starter",
                        "game_pk": game["game_pk"],
                        "season": season,
                        "game_date": game_date,
                        "game_start_utc": game_start.isoformat(),
                        "away_team": _team_name(away_meta),
                        "home_team": _team_name(home_meta),
                        "away_team_id": away_team_id,
                        "home_team_id": home_team_id,
                        "away_score": away_score,
                        "home_score": home_score,
                        "home_win": int(home_score > away_score),
                        "home_team_win_pct": home_team["win_pct"],
                        "away_team_win_pct": away_team["win_pct"],
                        "team_win_pct_adv": home_team["win_pct"] - away_team["win_pct"],
                        "home_team_run_diff_per_game": home_team["run_diff_per_game"],
                        "away_team_run_diff_per_game": away_team["run_diff_per_game"],
                        "team_run_diff_per_game_adv": home_team["run_diff_per_game"]
                        - away_team["run_diff_per_game"],
                        "team_games_adv": home_team["games"] - away_team["games"],
                        "home_starter_era": _starter_era(home_starter, pitching_state),
                        "away_starter_era": _starter_era(away_starter, pitching_state),
                        "starter_era_adv": _starter_era(away_starter, pitching_state)
                        - _starter_era(home_starter, pitching_state),
                        "home_lineup_batting_rating": home_lineup["batting_rating"],
                        "away_lineup_batting_rating": away_lineup["batting_rating"],
                        "home_lineup_power_rating": home_lineup["power_rating"],
                        "away_lineup_power_rating": away_lineup["power_rating"],
                        "home_lineup_rating_reliability": home_lineup["reliability"],
                        "away_lineup_rating_reliability": away_lineup["reliability"],
                        "home_lineup_rating_coverage": home_lineup["coverage"],
                        "away_lineup_rating_coverage": away_lineup["coverage"],
                        "home_lineup_rating_available": home_lineup["available"],
                        "away_lineup_rating_available": away_lineup["available"],
                    }
                    row.update(matchup_player_rating_features(home_lineup, away_lineup))
                    row.update(self._market_row(game_date, away_meta, home_meta))
                    rows.append(row)
                    pending_updates.append(
                        {
                            "away_box": away_box,
                            "home_box": home_box,
                            "away_team_id": away_team_id,
                            "home_team_id": home_team_id,
                            "away_score": away_score,
                            "home_score": home_score,
                        }
                    )

                for update in pending_updates:
                    _update_batting_state(batting_state, update["away_box"])
                    _update_batting_state(batting_state, update["home_box"])
                    _update_pitching_state(pitching_state, update["away_box"])
                    _update_pitching_state(pitching_state, update["home_box"])
                    _update_team_state(
                        team_state,
                        update["away_team_id"],
                        won=update["away_score"] > update["home_score"],
                        runs_scored=update["away_score"],
                        runs_allowed=update["home_score"],
                    )
                    _update_team_state(
                        team_state,
                        update["home_team_id"],
                        won=update["home_score"] > update["away_score"],
                        runs_scored=update["home_score"],
                        runs_allowed=update["away_score"],
                    )

        return pd.DataFrame(rows).sort_values(["game_start_utc", "game_pk"]).reset_index(drop=True)


def build_oracle_dataset(
    seasons: list[int],
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    include_odds: bool = True,
) -> pd.DataFrame:
    frame = PlayerRatingOracleDatasetBuilder(seasons, include_odds=include_odds).build()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict, oracle-lineup MLB player-rating research dataset."
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=[2026])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--skip-odds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_oracle_dataset(
        args.seasons,
        output_path=Path(args.output),
        include_odds=not args.skip_odds,
    )
    print(
        json.dumps(
            {
                "rows": int(len(frame)),
                "date_start": str(frame["game_date"].min()) if not frame.empty else None,
                "date_end": str(frame["game_date"].max()) if not frame.empty else None,
                "priced_rows": int(frame.get("market_available", pd.Series(dtype=int)).sum()),
                "schema": ORACLE_DATASET_SCHEMA,
                "oracle_lineup_known": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

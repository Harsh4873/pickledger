from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_api import HistoricalOddsArchive, StatsAPIClient, ensure_data_dirs
from park_factors import get_park_factor


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "mlb_historical_dataset.csv"


def innings_to_float(value: Any) -> float:
    if value in (None, "", "-", "-.--"):
        return 0.0
    whole, _, frac = str(value).partition(".")
    out = float(whole or 0)
    if frac == "1":
        out += 1.0 / 3.0
    elif frac == "2":
        out += 2.0 / 3.0
    return out


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rate_or_default(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return numerator / denominator


def compute_fip_constant(season_stats: dict[int, dict[str, Any]]) -> float:
    total_er = 0.0
    total_ip = 0.0
    total_hr = 0.0
    total_bb = 0.0
    total_hbp = 0.0
    total_k = 0.0

    for split in season_stats.values():
        stat = split.get("stat") or {}
        ip = innings_to_float(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        total_er += safe_float(stat.get("earnedRuns"))
        total_ip += ip
        total_hr += safe_float(stat.get("homeRuns"))
        total_bb += safe_float(stat.get("baseOnBalls"))
        total_hbp += safe_float(stat.get("hitByPitch")) or safe_float(stat.get("hitBatsmen"))
        total_k += safe_float(stat.get("strikeOuts"))

    if total_ip <= 0:
        return 3.2

    league_era = 9.0 * total_er / total_ip
    raw_component = (13.0 * total_hr + 3.0 * (total_bb + total_hbp) - 2.0 * total_k) / total_ip
    return round(league_era - raw_component, 3)


def compute_estimated_fip(stat: dict[str, Any], constant: float) -> float:
    innings = innings_to_float(stat.get("inningsPitched"))
    if innings <= 0:
        return round(constant, 3)
    home_runs = safe_float(stat.get("homeRuns"))
    walks = safe_float(stat.get("baseOnBalls"))
    hit_by_pitch = safe_float(stat.get("hitByPitch")) or safe_float(stat.get("hitBatsmen"))
    strikeouts = safe_float(stat.get("strikeOuts"))
    raw = (13.0 * home_runs + 3.0 * (walks + hit_by_pitch) - 2.0 * strikeouts) / innings
    return round(raw + constant, 3)


def recompute_era(earned_runs: float, innings: float) -> float:
    if innings <= 0:
        return 4.2
    return round(9.0 * earned_runs / innings, 3)


def recompute_whip(hits: float, walks: float, innings: float) -> float:
    if innings <= 0:
        return 1.3
    return round((hits + walks) / innings, 3)


def subtract_pitching_game_from_season(
    season_stat: dict[str, Any],
    game_stat: dict[str, Any],
    *,
    was_starter: bool,
) -> dict[str, Any]:
    pre = dict(season_stat)
    subtract_keys = (
        "runs",
        "doubles",
        "triples",
        "homeRuns",
        "strikeOuts",
        "baseOnBalls",
        "hits",
        "atBats",
        "stolenBases",
        "numberOfPitches",
        "earnedRuns",
        "pitchesThrown",
        "strikes",
        "hitByPitch",
        "hitBatsmen",
        "balls",
        "battersFaced",
        "outs",
        "sacBunts",
        "sacFlies",
        "pickoffs",
        "wildPitches",
        "balks",
    )
    for key in subtract_keys:
        pre[key] = safe_float(season_stat.get(key)) - safe_float(game_stat.get(key))

    pre["gamesPlayed"] = max(0, safe_int(season_stat.get("gamesPlayed")) - 1)
    pre["gamesPitched"] = max(0, safe_int(season_stat.get("gamesPitched")) - 1)
    pre["gamesStarted"] = max(0, safe_int(season_stat.get("gamesStarted")) - (1 if was_starter else 0))

    season_ip = innings_to_float(season_stat.get("inningsPitched"))
    game_ip = innings_to_float(game_stat.get("inningsPitched"))
    pre_ip = max(0.0, season_ip - game_ip)
    pre["inningsPitched"] = pre_ip
    pre["era"] = recompute_era(pre["earnedRuns"], pre_ip)
    pre["whip"] = recompute_whip(pre["hits"], pre["baseOnBalls"], pre_ip)
    return pre


def subtract_batting_game_from_season(
    season_stat: dict[str, Any],
    game_stat: dict[str, Any],
) -> dict[str, Any]:
    pre = dict(season_stat)
    subtract_keys = (
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
        "groundIntoDoublePlay",
        "groundIntoTriplePlay",
        "plateAppearances",
        "totalBases",
        "rbi",
        "leftOnBase",
        "sacBunts",
        "sacFlies",
        "catchersInterference",
        "pickoffs",
        "flyOuts",
        "groundOuts",
        "airOuts",
        "popOuts",
        "lineOuts",
    )
    for key in subtract_keys:
        pre[key] = max(0.0, safe_float(season_stat.get(key)) - safe_float(game_stat.get(key)))

    games_played = max(0, safe_int(season_stat.get("gamesPlayed")) - 1)
    pre["gamesPlayed"] = games_played

    hits = pre["hits"]
    walks = pre["baseOnBalls"]
    hbp = pre["hitByPitch"]
    at_bats = pre["atBats"]
    total_bases = pre["totalBases"]
    sac_flies = pre["sacFlies"]
    obp_denom = at_bats + walks + hbp + sac_flies
    pre["obp"] = rate_or_default(hits + walks + hbp, obp_denom)
    pre["slg"] = rate_or_default(total_bases, at_bats)
    pre["ops"] = pre["obp"] + pre["slg"]
    pre["avg"] = rate_or_default(hits, at_bats)
    return pre


def parse_weather(game_weather: dict[str, Any], roof_type: str) -> dict[str, Any]:
    roof = str(roof_type or "").lower()
    is_dome = roof in {"dome", "fixed", "closed"}

    temp = safe_float(game_weather.get("temp"))
    wind_text = str(game_weather.get("wind") or "")
    wind_speed = 0.0
    wind_direction = "unknown"
    if "mph" in wind_text:
        try:
            wind_speed = safe_float(wind_text.split("mph", 1)[0].strip().split()[-1])
        except IndexError:
            wind_speed = 0.0

    lowered = wind_text.lower()
    if "out to" in lowered:
        wind_direction = "out"
    elif "in from" in lowered:
        wind_direction = "in"
    elif lowered:
        wind_direction = "cross"

    if is_dome:
        wind_speed = 0.0
        wind_direction = "dome"

    return {
        "temperature_f": temp,
        "wind_speed_mph": wind_speed,
        "wind_direction": wind_direction,
        "is_dome": int(is_dome),
    }


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.756
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def team_short_name(feed_team: dict[str, Any]) -> str:
    return str(feed_team.get("abbreviation") or feed_team.get("teamName") or "").upper()


@dataclass
class TeamHistoryRecord:
    game_date: date
    venue_lat: float
    venue_lon: float
    timezone_offset: int
    won: int
    bullpen_pitches: int
    bullpen_innings: float
    bullpen_earned_runs: int
    runs_scored: int
    runs_allowed: int


class HistoricalDatasetBuilder:
    def __init__(self, seasons: list[int], include_odds: bool = True) -> None:
        ensure_data_dirs()
        self.seasons = seasons
        self.include_odds = include_odds
        self.client = StatsAPIClient()
        self.odds = HistoricalOddsArchive() if include_odds else None
        self.pitching_stats_by_season: dict[int, dict[int, dict[str, Any]]] = {}
        self.hitting_stats_by_season: dict[int, dict[int, dict[str, Any]]] = {}
        self.pitch_hand_cache: dict[int, str] = {}
        self.fip_constants: dict[int, float] = {}
        self.final_team_win_pct: dict[int, dict[int, float]] = {}

    def _load_static_context(self) -> None:
        seasons_to_pull = set(self.seasons)
        seasons_to_pull.update({season - 1 for season in self.seasons if season > 2000})
        for season in sorted(seasons_to_pull):
            pitching = self.client.get_season_player_stats(season, "pitching")
            hitting = self.client.get_season_player_stats(season, "hitting")
            self.pitching_stats_by_season[season] = pitching
            self.hitting_stats_by_season[season] = hitting
            self.fip_constants[season] = compute_fip_constant(pitching)
            self.final_team_win_pct[season] = self._compute_team_win_pct_for_season(season)

    def _compute_team_win_pct_for_season(self, season: int) -> dict[int, float]:
        games = self.client.get_schedule_for_season(season)
        wins = defaultdict(int)
        losses = defaultdict(int)
        for game in games:
            away_id = safe_int(game.get("away_id"))
            home_id = safe_int(game.get("home_id"))
            away_score = safe_int(game.get("away_score"))
            home_score = safe_int(game.get("home_score"))
            status = str(game.get("status") or "")
            if "final" not in status.lower():
                continue
            if home_score > away_score:
                wins[home_id] += 1
                losses[away_id] += 1
            elif away_score > home_score:
                wins[away_id] += 1
                losses[home_id] += 1

        out: dict[int, float] = {}
        for team_id in set(wins) | set(losses):
            games_played = wins[team_id] + losses[team_id]
            out[team_id] = rate_or_default(wins[team_id], games_played, 0.5)
        return out

    def build(self) -> pd.DataFrame:
        self._load_static_context()
        rows: list[dict[str, Any]] = []
        for season in sorted(self.seasons):
            season_games = sorted(
                self.client.get_schedule_for_season(season),
                key=lambda game: str(game.get("game_datetime") or game.get("game_date") or ""),
            )
            final_game_pks = [
                safe_int(game.get("game_id"))
                for game in season_games
                if "final" in str(game.get("status") or "").lower()
            ]
            self.client.prefetch_game_feeds(final_game_pks, max_workers=24)

            for game in season_games:
                status = str(game.get("status") or "")
                if "final" not in status.lower():
                    continue
                row = self._build_game_row(season, safe_int(game.get("game_id")))
                if row is None:
                    continue
                rows.append(row)
        frame = pd.DataFrame(rows).sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)
        return frame

    def _build_game_row(self, season: int, game_pk: int) -> dict[str, Any] | None:
        payload = self.client.get_game_feed(game_pk)
        game_data = payload.get("gameData") or {}
        live_box = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
        away_box = live_box.get("away") or {}
        home_box = live_box.get("home") or {}
        if not away_box or not home_box:
            return None

        away_team_meta = (game_data.get("teams") or {}).get("away") or {}
        home_team_meta = (game_data.get("teams") or {}).get("home") or {}
        venue = game_data.get("venue") or {}
        location = venue.get("location") or {}
        field_info = venue.get("fieldInfo") or {}
        weather = parse_weather(game_data.get("weather") or {}, field_info.get("roofType", ""))

        game_date = datetime.fromisoformat(game_data.get("datetime", {}).get("dateTime", "").replace("Z", "+00:00")).date()
        venue_lat = safe_float((location.get("defaultCoordinates") or {}).get("latitude"))
        venue_lon = safe_float((location.get("defaultCoordinates") or {}).get("longitude"))
        timezone_offset = safe_int((venue.get("timeZone") or {}).get("offsetAtGameTime"))

        away_score = safe_int(payload.get("liveData", {}).get("linescore", {}).get("teams", {}).get("away", {}).get("runs"))
        home_score = safe_int(payload.get("liveData", {}).get("linescore", {}).get("teams", {}).get("home", {}).get("runs"))

        away_pitcher_id = safe_int((away_box.get("pitchers") or [0])[0])
        home_pitcher_id = safe_int((home_box.get("pitchers") or [0])[0])
        if not away_pitcher_id or not home_pitcher_id:
            return None

        away_starter = away_box.get("players", {}).get(f"ID{away_pitcher_id}") or {}
        home_starter = home_box.get("players", {}).get(f"ID{home_pitcher_id}") or {}
        away_pitching_pre = subtract_pitching_game_from_season(
            away_starter.get("seasonStats", {}).get("pitching", {}),
            away_starter.get("stats", {}).get("pitching", {}),
            was_starter=True,
        )
        home_pitching_pre = subtract_pitching_game_from_season(
            home_starter.get("seasonStats", {}).get("pitching", {}),
            home_starter.get("stats", {}).get("pitching", {}),
            was_starter=True,
        )

        away_lineup = self._lineup_proxy(away_box, season - 1)
        home_lineup = self._lineup_proxy(home_box, season - 1)

        away_record = self._pregame_team_record(away_team_meta, away_score > home_score)
        home_record = self._pregame_team_record(home_team_meta, home_score > away_score)

        away_bullpen = self._extract_bullpen_game_stats(away_box)
        home_bullpen = self._extract_bullpen_game_stats(home_box)

        previous_season = season - 1
        away_prior_pitching = self.pitching_stats_by_season.get(previous_season, {}).get(away_pitcher_id, {})
        home_prior_pitching = self.pitching_stats_by_season.get(previous_season, {}).get(home_pitcher_id, {})

        away_prior_stat = away_prior_pitching.get("stat") or {}
        home_prior_stat = home_prior_pitching.get("stat") or {}
        away_prior_fip = compute_estimated_fip(
            away_prior_stat,
            self.fip_constants.get(previous_season, 3.2),
        )
        home_prior_fip = compute_estimated_fip(
            home_prior_stat,
            self.fip_constants.get(previous_season, 3.2),
        )

        away_abbrev = team_short_name(away_team_meta)
        home_abbrev = team_short_name(home_team_meta)

        row = {
            "game_pk": game_pk,
            "season": season,
            "game_date": game_date.isoformat(),
            "away_team_id": safe_int(away_team_meta.get("id")),
            "home_team_id": safe_int(home_team_meta.get("id")),
            "away_team": away_team_meta.get("name"),
            "home_team": home_team_meta.get("name"),
            "away_abbrev": away_abbrev,
            "home_abbrev": home_abbrev,
            "venue_name": venue.get("name"),
            "venue_id": safe_int(venue.get("id")),
            "park_factor_runs": get_park_factor(str(venue.get("name") or "")),
            "venue_lat": venue_lat,
            "venue_lon": venue_lon,
            "venue_timezone_offset": timezone_offset,
            "temperature_f": weather["temperature_f"],
            "wind_speed_mph": weather["wind_speed_mph"],
            "wind_direction": weather["wind_direction"],
            "is_dome": weather["is_dome"],
            "away_score": away_score,
            "home_score": home_score,
            "home_win": int(home_score > away_score),
            "total_runs": away_score + home_score,
            "away_season_win_pct": away_record["win_pct"],
            "home_season_win_pct": home_record["win_pct"],
            "away_games_played": away_record["games_played"],
            "home_games_played": home_record["games_played"],
            "away_prior_season_win_pct": self.final_team_win_pct.get(previous_season, {}).get(
                safe_int(away_team_meta.get("id")),
                0.5,
            ),
            "home_prior_season_win_pct": self.final_team_win_pct.get(previous_season, {}).get(
                safe_int(home_team_meta.get("id")),
                0.5,
            ),
            "away_starter_id": away_pitcher_id,
            "home_starter_id": home_pitcher_id,
            "away_starter_name": (away_starter.get("person") or {}).get("fullName"),
            "home_starter_name": (home_starter.get("person") or {}).get("fullName"),
            "away_starter_hand": self._pitch_hand(away_pitcher_id),
            "home_starter_hand": self._pitch_hand(home_pitcher_id),
            "away_starter_era": safe_float(away_pitching_pre.get("era")),
            "home_starter_era": safe_float(home_pitching_pre.get("era")),
            "away_starter_fip": compute_estimated_fip(
                away_pitching_pre,
                self.fip_constants.get(season, 3.2),
            ),
            "home_starter_fip": compute_estimated_fip(
                home_pitching_pre,
                self.fip_constants.get(season, 3.2),
            ),
            "away_starter_whip": safe_float(away_pitching_pre.get("whip")),
            "home_starter_whip": safe_float(home_pitching_pre.get("whip")),
            "away_starter_ip": innings_to_float(away_pitching_pre.get("inningsPitched")),
            "home_starter_ip": innings_to_float(home_pitching_pre.get("inningsPitched")),
            "away_starter_starts": safe_int(away_pitching_pre.get("gamesStarted")),
            "home_starter_starts": safe_int(home_pitching_pre.get("gamesStarted")),
            "away_prior_starter_era": safe_float(away_prior_stat.get("era")),
            "home_prior_starter_era": safe_float(home_prior_stat.get("era")),
            "away_prior_starter_fip": away_prior_fip,
            "home_prior_starter_fip": home_prior_fip,
            "away_prior_starter_ip": innings_to_float(away_prior_stat.get("inningsPitched")),
            "home_prior_starter_ip": innings_to_float(home_prior_stat.get("inningsPitched")),
            "away_lineup_ops_proxy": away_lineup["ops"],
            "home_lineup_ops_proxy": home_lineup["ops"],
            "away_lineup_obp_proxy": away_lineup["obp"],
            "home_lineup_obp_proxy": home_lineup["obp"],
            "away_lineup_slg_proxy": away_lineup["slg"],
            "home_lineup_slg_proxy": home_lineup["slg"],
            "away_lineup_sample_games": away_lineup["sample_games"],
            "home_lineup_sample_games": home_lineup["sample_games"],
            "away_bullpen_pitches_game": away_bullpen["pitches"],
            "home_bullpen_pitches_game": home_bullpen["pitches"],
            "away_bullpen_ip_game": away_bullpen["innings"],
            "home_bullpen_ip_game": home_bullpen["innings"],
            "away_bullpen_er_game": away_bullpen["earned_runs"],
            "home_bullpen_er_game": home_bullpen["earned_runs"],
            # TODO: lineup-level handedness splits need a dedicated archived lineup source.
            # This OPS/OBP/SLG blend is an explicit proxy, not a full projected wRC+ model.
        }

        odds_row = {}
        if self.odds is not None:
            odds_row = self.odds.lookup_moneyline(game_date.isoformat(), away_abbrev, home_abbrev)
        row.update(odds_row)
        return row

    def _pregame_team_record(self, team_meta: dict[str, Any], won_today: bool) -> dict[str, float]:
        record = team_meta.get("record") or {}
        wins = safe_int(record.get("wins"))
        losses = safe_int(record.get("losses"))
        if won_today:
            wins = max(0, wins - 1)
        else:
            losses = max(0, losses - 1)
        games_played = wins + losses
        return {"games_played": games_played, "win_pct": rate_or_default(wins, games_played, 0.5)}

    def _pitch_hand(self, player_id: int) -> str:
        if player_id in self.pitch_hand_cache:
            return self.pitch_hand_cache[player_id]
        person = self.client.get_person(player_id)
        pitch_hand = ((person.get("pitchHand") or {}).get("code")) or "U"
        self.pitch_hand_cache[player_id] = pitch_hand
        return pitch_hand

    def _lineup_proxy(self, team_box: dict[str, Any], previous_season: int) -> dict[str, float]:
        players = team_box.get("players") or {}
        lineup = (team_box.get("battingOrder") or [])[:9]
        ops_values: list[float] = []
        obp_values: list[float] = []
        slg_values: list[float] = []
        sample_games: list[int] = []
        for pid in lineup:
            player = players.get(f"ID{pid}") or {}
            season_stat = (player.get("seasonStats") or {}).get("batting", {})
            game_stat = (player.get("stats") or {}).get("batting", {})
            pre = subtract_batting_game_from_season(season_stat, game_stat)
            games_played = safe_int(pre.get("gamesPlayed"))

            if games_played <= 0:
                prior_split = self.hitting_stats_by_season.get(previous_season, {}).get(safe_int(pid), {})
                prior_stat = prior_split.get("stat") or {}
                prior_games = safe_int(prior_stat.get("gamesPlayed"))
                if prior_games > 0:
                    ops_values.append(safe_float(prior_stat.get("ops")))
                    obp_values.append(safe_float(prior_stat.get("obp")))
                    slg_values.append(safe_float(prior_stat.get("slg")))
                    sample_games.append(prior_games)
                    continue

            ops_values.append(safe_float(pre.get("ops")))
            obp_values.append(safe_float(pre.get("obp")))
            slg_values.append(safe_float(pre.get("slg")))
            sample_games.append(games_played)

        return {
            "ops": float(np.mean(ops_values)) if ops_values else 0.700,
            "obp": float(np.mean(obp_values)) if obp_values else 0.320,
            "slg": float(np.mean(slg_values)) if slg_values else 0.380,
            "sample_games": float(np.mean(sample_games)) if sample_games else 0.0,
        }

    def _extract_bullpen_game_stats(self, team_box: dict[str, Any]) -> dict[str, float]:
        players = team_box.get("players") or {}
        pitcher_ids = (team_box.get("pitchers") or [])[1:]
        total_pitches = 0
        total_innings = 0.0
        total_er = 0
        for pid in pitcher_ids:
            player = players.get(f"ID{pid}") or {}
            game_stat = (player.get("stats") or {}).get("pitching", {})
            total_pitches += safe_int(game_stat.get("pitchesThrown") or game_stat.get("numberOfPitches"))
            total_innings += innings_to_float(game_stat.get("inningsPitched"))
            total_er += safe_int(game_stat.get("earnedRuns"))
        return {"pitches": total_pitches, "innings": total_innings, "earned_runs": total_er}

def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = frame.sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)

    result_frames: list[pd.DataFrame] = []
    for season, season_frame in frame.groupby("season", sort=True):
        season_frame = season_frame.copy()
        team_history: dict[int, list[TeamHistoryRecord]] = defaultdict(list)
        enriched_rows: list[dict[str, Any]] = []
        for row in season_frame.to_dict("records"):
            current_date = pd.Timestamp(row["game_date"]).date()
            away_features = team_context_from_history(
                team_history[safe_int(row["away_team_id"])],
                current_date,
                safe_float(row["venue_lat"]),
                safe_float(row["venue_lon"]),
                safe_int(row["venue_timezone_offset"]),
            )
            home_features = team_context_from_history(
                team_history[safe_int(row["home_team_id"])],
                current_date,
                safe_float(row["venue_lat"]),
                safe_float(row["venue_lon"]),
                safe_int(row["venue_timezone_offset"]),
            )
            row.update(prefix_keys("away_", away_features))
            row.update(prefix_keys("home_", home_features))
            enriched_rows.append(row)

            game_record = {
                "game_date": current_date,
                "venue_lat": safe_float(row["venue_lat"]),
                "venue_lon": safe_float(row["venue_lon"]),
                "timezone_offset": safe_int(row["venue_timezone_offset"]),
            }
            team_history[safe_int(row["away_team_id"])].append(
                TeamHistoryRecord(
                    **game_record,
                    won=int(row["home_win"] == 0),
                    bullpen_pitches=safe_int(row["away_bullpen_pitches_game"]),
                    bullpen_innings=safe_float(row["away_bullpen_ip_game"]),
                    bullpen_earned_runs=safe_int(row["away_bullpen_er_game"]),
                    runs_scored=safe_int(row["away_score"]),
                    runs_allowed=safe_int(row["home_score"]),
                )
            )
            team_history[safe_int(row["home_team_id"])].append(
                TeamHistoryRecord(
                    **game_record,
                    won=safe_int(row["home_win"]),
                    bullpen_pitches=safe_int(row["home_bullpen_pitches_game"]),
                    bullpen_innings=safe_float(row["home_bullpen_ip_game"]),
                    bullpen_earned_runs=safe_int(row["home_bullpen_er_game"]),
                    runs_scored=safe_int(row["home_score"]),
                    runs_allowed=safe_int(row["away_score"]),
                )
            )
        result_frames.append(pd.DataFrame(enriched_rows))

    return pd.concat(result_frames, ignore_index=True)


def team_context_from_history(
    history: list[TeamHistoryRecord],
    current_date: date,
    current_lat: float,
    current_lon: float,
    current_timezone: int,
) -> dict[str, float]:
    windows = {}
    for days in (7, 14, 30):
        subset = [
            record
            for record in history
            if 0 < (current_date - record.game_date).days <= days
        ]
        wins = sum(record.won for record in subset)
        windows[f"form_{days}d_win_pct"] = rate_or_default(wins, len(subset), 0.5)
        windows[f"form_{days}d_games"] = float(len(subset))

    usage_1d = [
        record for record in history if 0 < (current_date - record.game_date).days <= 1
    ]
    usage_3d = [
        record for record in history if 0 < (current_date - record.game_date).days <= 3
    ]
    quality_30d = [
        record for record in history if 0 < (current_date - record.game_date).days <= 30
    ]

    bullpen_ip = sum(record.bullpen_innings for record in quality_30d)
    bullpen_er = sum(record.bullpen_earned_runs for record in quality_30d)
    bullpen_era = 9.0 * bullpen_er / bullpen_ip if bullpen_ip > 0 else 4.2

    if history:
        last_game = history[-1]
        rest_days = max(0, (current_date - last_game.game_date).days - 1)
        travel_distance = haversine_miles(
            last_game.venue_lat,
            last_game.venue_lon,
            current_lat,
            current_lon,
        )
        timezone_jump = abs(current_timezone - last_game.timezone_offset)
    else:
        rest_days = 2
        travel_distance = 0.0
        timezone_jump = 0

    return {
        **windows,
        "bullpen_pitches_1d": float(sum(record.bullpen_pitches for record in usage_1d)),
        "bullpen_pitches_3d": float(sum(record.bullpen_pitches for record in usage_3d)),
        "bullpen_era_30d": float(bullpen_era),
        "rest_days": float(rest_days),
        "travel_distance_miles": float(travel_distance),
        "travel_flag": float(int(rest_days == 0 and (travel_distance >= 500 or timezone_jump >= 2))),
    }


def prefix_keys(prefix: str, payload: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in payload.items()}


def build_historical_dataset(
    seasons: list[int],
    *,
    output_path: Path = DATASET_PATH,
    include_odds: bool = True,
) -> pd.DataFrame:
    builder = HistoricalDatasetBuilder(seasons, include_odds=include_odds)
    frame = builder.build()
    frame = add_context_features(frame)
    frame = frame.sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a historical MLB training dataset.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="Regular seasons to include in the dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(DATASET_PATH),
        help="CSV output path.",
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip optional historical odds enrichment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_historical_dataset(
        args.seasons,
        output_path=Path(args.output),
        include_odds=not args.skip_odds,
    )
    print(f"Saved {len(dataset):,} rows to {args.output}")
    print(f"Columns: {len(dataset.columns)}")
    print("Seasons:", ", ".join(str(season) for season in sorted(set(dataset['season']))))


if __name__ == "__main__":
    main()

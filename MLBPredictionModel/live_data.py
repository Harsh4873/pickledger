from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import statsapi

from date_utils import get_mlb_slate_date
from historical_data import (
    TeamHistoryRecord,
    compute_estimated_fip,
    compute_fip_constant,
    innings_to_float,
    parse_weather,
    rate_or_default,
    safe_float,
    safe_int,
    team_context_from_history,
    team_short_name,
)
from mlb_api import StatsAPIClient
from park_factors import get_park_factor


def _season_for_date(target_date: date) -> int:
    return target_date.year


def _schedule_date(target_date: date) -> str:
    return target_date.strftime("%Y-%m-%d")


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_team_history(
    client: StatsAPIClient,
    season: int,
    target_date: date,
) -> dict[int, list[TeamHistoryRecord]]:
    history: dict[int, list[TeamHistoryRecord]] = defaultdict(list)
    for game in client.get_schedule_for_season(season):
        status = str(game.get("status") or "")
        if "final" not in status.lower():
            continue
        game_date = datetime.strptime(str(game.get("game_date")), "%Y-%m-%d").date()
        if game_date >= target_date:
            continue
        game_pk = safe_int(game.get("game_id"))
        payload = client.get_game_feed(game_pk)
        venue = payload.get("gameData", {}).get("venue", {})
        location = venue.get("location", {})
        linescore = payload.get("liveData", {}).get("linescore", {}).get("teams", {})
        live_box = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
        venue_lat = safe_float((location.get("defaultCoordinates") or {}).get("latitude"))
        venue_lon = safe_float((location.get("defaultCoordinates") or {}).get("longitude"))
        timezone_offset = safe_int((venue.get("timeZone") or {}).get("offsetAtGameTime"))

        away_box = live_box.get("away") or {}
        home_box = live_box.get("home") or {}
        away_bullpen = _extract_bullpen_game_stats(away_box)
        home_bullpen = _extract_bullpen_game_stats(home_box)

        away_score = safe_int((linescore.get("away") or {}).get("runs"))
        home_score = safe_int((linescore.get("home") or {}).get("runs"))

        away_team_id = safe_int(((payload.get("gameData") or {}).get("teams") or {}).get("away", {}).get("id"))
        home_team_id = safe_int(((payload.get("gameData") or {}).get("teams") or {}).get("home", {}).get("id"))

        common = {
            "game_date": game_date,
            "venue_lat": venue_lat,
            "venue_lon": venue_lon,
            "timezone_offset": timezone_offset,
        }
        history[away_team_id].append(
            TeamHistoryRecord(
                **common,
                won=int(away_score > home_score),
                bullpen_pitches=away_bullpen["pitches"],
                bullpen_innings=away_bullpen["innings"],
                bullpen_earned_runs=away_bullpen["earned_runs"],
                runs_scored=away_score,
                runs_allowed=home_score,
            )
        )
        history[home_team_id].append(
            TeamHistoryRecord(
                **common,
                won=int(home_score > away_score),
                bullpen_pitches=home_bullpen["pitches"],
                bullpen_innings=home_bullpen["innings"],
                bullpen_earned_runs=home_bullpen["earned_runs"],
                runs_scored=home_score,
                runs_allowed=away_score,
            )
        )
    return history


def _extract_bullpen_game_stats(team_box: dict[str, Any]) -> dict[str, float]:
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


def _team_final_win_pct(client: StatsAPIClient, season: int) -> dict[int, float]:
    wins = defaultdict(int)
    losses = defaultdict(int)
    for game in client.get_schedule_for_season(season):
        status = str(game.get("status") or "")
        if "final" not in status.lower():
            continue
        away_id = safe_int(game.get("away_id"))
        home_id = safe_int(game.get("home_id"))
        away_score = safe_int(game.get("away_score"))
        home_score = safe_int(game.get("home_score"))
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


def _group_hitting_by_team(
    season_hitting_stats: dict[int, dict[str, Any]],
) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"ops": [], "obp": [], "slg": [], "games": []}
    )
    for split in season_hitting_stats.values():
        team = split.get("team") or {}
        team_id = team.get("id")
        stat = split.get("stat") or {}
        if team_id is None or safe_int(stat.get("gamesPlayed")) <= 0:
            continue
        grouped[int(team_id)]["ops"].append(safe_float(stat.get("ops")))
        grouped[int(team_id)]["obp"].append(safe_float(stat.get("obp")))
        grouped[int(team_id)]["slg"].append(safe_float(stat.get("slg")))
        grouped[int(team_id)]["games"].append(safe_int(stat.get("gamesPlayed")))

    out: dict[int, dict[str, float]] = {}
    for team_id, values in grouped.items():
        out[team_id] = {
            "ops": sum(values["ops"]) / len(values["ops"]) if values["ops"] else 0.710,
            "obp": sum(values["obp"]) / len(values["obp"]) if values["obp"] else 0.320,
            "slg": sum(values["slg"]) / len(values["slg"]) if values["slg"] else 0.390,
            "sample_games": sum(values["games"]) / len(values["games"]) if values["games"] else 0.0,
        }
    return out


def _lineup_proxy_for_live_game(
    team_box: dict[str, Any],
    team_id: int,
    team_hitting: dict[int, dict[str, float]],
) -> dict[str, float]:
    players = team_box.get("players") or {}
    batting_order = (team_box.get("battingOrder") or [])[:9]
    if not batting_order:
        return team_hitting.get(team_id, {"ops": 0.710, "obp": 0.320, "slg": 0.390, "sample_games": 0.0})

    ops: list[float] = []
    obp: list[float] = []
    slg: list[float] = []
    sample_games: list[float] = []
    for pid in batting_order:
        player = players.get(f"ID{pid}") or {}
        season_batting = (player.get("seasonStats") or {}).get("batting", {})
        ops.append(_float_or_default(season_batting.get("ops"), 0.710))
        obp.append(_float_or_default(season_batting.get("obp"), 0.320))
        slg.append(_float_or_default(season_batting.get("slg"), 0.390))
        sample_games.append(_float_or_default(season_batting.get("gamesPlayed"), 0.0))

    return {
        "ops": sum(ops) / len(ops) if ops else 0.710,
        "obp": sum(obp) / len(obp) if obp else 0.320,
        "slg": sum(slg) / len(slg) if slg else 0.390,
        "sample_games": sum(sample_games) / len(sample_games) if sample_games else 0.0,
    }


def _pitcher_stat_bundle(
    player_id: int,
    season_pitching_stats: dict[int, dict[str, Any]],
    prior_pitching_stats: dict[int, dict[str, Any]],
    current_fip_constant: float,
    prior_fip_constant: float,
    season: int,
    client: StatsAPIClient,
) -> dict[str, Any]:
    current_split = season_pitching_stats.get(player_id, {})
    current_stat = current_split.get("stat") or {}
    prior_split = prior_pitching_stats.get(player_id, {})
    prior_stat = prior_split.get("stat") or {}
    person = client.get_person(player_id) if player_id else {}
    recent_form = _pitcher_recent_form(player_id, season, client)
    return {
        "hand": ((person.get("pitchHand") or {}).get("code")) or "U",
        "era": _float_or_default(current_stat.get("era"), 4.2),
        "fip": compute_estimated_fip(current_stat, current_fip_constant),
        "whip": _float_or_default(current_stat.get("whip"), 1.3),
        "ip": innings_to_float(current_stat.get("inningsPitched")),
        "starts": safe_int(current_stat.get("gamesStarted")),
        # Raw counts so v2 features can compute K/9, BB/9, HR/9 and K-BB%.
        "strikeouts": safe_float(current_stat.get("strikeOuts")),
        "walks": safe_float(current_stat.get("baseOnBalls")),
        "home_runs": safe_float(current_stat.get("homeRuns")),
        "batters_faced": safe_float(current_stat.get("battersFaced")),
        "prior_era": _float_or_default(prior_stat.get("era"), 4.2),
        "prior_fip": compute_estimated_fip(prior_stat, prior_fip_constant),
        "prior_ip": innings_to_float(prior_stat.get("inningsPitched")),
        "last_5_starts_era": recent_form["last_5_starts_era"],
        "last_5_starts_whip": recent_form["last_5_starts_whip"],
        "last_5_starts_count": recent_form["last_5_starts_count"],
    }


def _pitcher_recent_form(player_id: int, season: int, client: StatsAPIClient) -> dict[str, Any]:
    if not player_id:
        # Keep the placeholder explicit when there is no pitcher id to resolve.
        return {"last_5_starts_era": None, "last_5_starts_whip": None, "last_5_starts_count": 0}

    game_log = client.get_player_game_log(player_id, season, group="pitching")
    starts = [
        split
        for split in game_log
        if safe_int((split.get("stat") or {}).get("gamesStarted")) > 0
    ]
    starts.sort(key=lambda split: str(split.get("date") or ""), reverse=True)
    recent_starts = starts[:5]
    if not recent_starts:
        # StatsAPI can return no game log for new call-ups or pitchers without starts yet.
        return {"last_5_starts_era": None, "last_5_starts_whip": None, "last_5_starts_count": 0}

    total_ip = 0.0
    total_hits = 0
    total_walks = 0
    total_earned_runs = 0
    for start in recent_starts:
        stat = start.get("stat") or {}
        total_ip += innings_to_float(stat.get("inningsPitched"))
        total_hits += safe_int(stat.get("hits"))
        total_walks += safe_int(stat.get("baseOnBalls"))
        total_earned_runs += safe_int(stat.get("earnedRuns"))

    if total_ip <= 0:
        return {"last_5_starts_era": None, "last_5_starts_whip": None, "last_5_starts_count": len(recent_starts)}

    return {
        "last_5_starts_era": round((total_earned_runs * 9.0) / total_ip, 3),
        "last_5_starts_whip": round((total_hits + total_walks) / total_ip, 3),
        "last_5_starts_count": len(recent_starts),
    }


def build_live_dataframe(
    target_date: date | None = None,
    market_odds_map: dict[tuple[str, str], dict] | None = None,
) -> pd.DataFrame:
    target_date = target_date or get_mlb_slate_date()
    season = _season_for_date(target_date)
    previous_season = season - 1

    client = StatsAPIClient()
    season_pitching = client.get_season_player_stats(season, "pitching")
    season_hitting = client.get_season_player_stats(season, "hitting")
    prior_pitching = client.get_season_player_stats(previous_season, "pitching")

    team_history = _load_team_history(client, season, target_date)
    prior_team_win_pct = _team_final_win_pct(client, previous_season)

    current_fip_constant = compute_fip_constant(season_pitching)
    prior_fip_constant = compute_fip_constant(prior_pitching)
    team_hitting = _group_hitting_by_team(season_hitting)

    games = statsapi.schedule(
        start_date=_schedule_date(target_date),
        end_date=_schedule_date(target_date),
        sportId=1,
    )
    rows: list[dict[str, Any]] = []
    for game in games:
        if str(game.get("game_type", "")).upper() != "R":
            continue
        game_pk = safe_int(game.get("game_id"))
        payload = client.get_game_feed(game_pk)
        game_data = payload.get("gameData") or {}
        live_box = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
        away_box = live_box.get("away") or {}
        home_box = live_box.get("home") or {}

        away_meta = (game_data.get("teams") or {}).get("away") or {}
        home_meta = (game_data.get("teams") or {}).get("home") or {}
        venue = game_data.get("venue") or {}
        location = venue.get("location") or {}
        field_info = venue.get("fieldInfo") or {}

        probable_pitchers = game_data.get("probablePitchers") or {}
        away_pitcher_id = safe_int((probable_pitchers.get("away") or {}).get("id"))
        home_pitcher_id = safe_int((probable_pitchers.get("home") or {}).get("id"))
        if not away_pitcher_id and away_box.get("pitchers"):
            away_pitcher_id = safe_int(away_box["pitchers"][0])
        if not home_pitcher_id and home_box.get("pitchers"):
            home_pitcher_id = safe_int(home_box["pitchers"][0])

        away_pitcher = _pitcher_stat_bundle(
            away_pitcher_id,
            season_pitching,
            prior_pitching,
            current_fip_constant,
            prior_fip_constant,
            season,
            client,
        )
        home_pitcher = _pitcher_stat_bundle(
            home_pitcher_id,
            season_pitching,
            prior_pitching,
            current_fip_constant,
            prior_fip_constant,
            season,
            client,
        )

        away_team_id = safe_int(away_meta.get("id"))
        home_team_id = safe_int(home_meta.get("id"))
        away_context = team_context_from_history(
            team_history[away_team_id],
            target_date,
            safe_float((location.get("defaultCoordinates") or {}).get("latitude")),
            safe_float((location.get("defaultCoordinates") or {}).get("longitude")),
            safe_int((venue.get("timeZone") or {}).get("offsetAtGameTime")),
        )
        home_context = team_context_from_history(
            team_history[home_team_id],
            target_date,
            safe_float((location.get("defaultCoordinates") or {}).get("latitude")),
            safe_float((location.get("defaultCoordinates") or {}).get("longitude")),
            safe_int((venue.get("timeZone") or {}).get("offsetAtGameTime")),
        )
        weather = parse_weather(game_data.get("weather") or {}, field_info.get("roofType", ""))

        away_lineup = _lineup_proxy_for_live_game(away_box, away_team_id, team_hitting)
        home_lineup = _lineup_proxy_for_live_game(home_box, home_team_id, team_hitting)

        away_record = away_meta.get("record") or {}
        home_record = home_meta.get("record") or {}

        rows.append(
            {
                "season": season,
                "game_pk": game_pk,
                "game_date": target_date.isoformat(),
                "away_team_id": away_team_id,
                "home_team_id": home_team_id,
                "away_team": away_meta.get("name"),
                "home_team": home_meta.get("name"),
                "away_abbrev": team_short_name(away_meta),
                "home_abbrev": team_short_name(home_meta),
                "venue_name": venue.get("name"),
                "venue_id": safe_int(venue.get("id")),
                "park_factor_runs": get_park_factor(str(venue.get("name") or "")),
                "venue_lat": safe_float((location.get("defaultCoordinates") or {}).get("latitude")),
                "venue_lon": safe_float((location.get("defaultCoordinates") or {}).get("longitude")),
                "venue_timezone_offset": safe_int((venue.get("timeZone") or {}).get("offsetAtGameTime")),
                "temperature_f": weather["temperature_f"],
                "wind_speed_mph": weather["wind_speed_mph"],
                "wind_direction": weather["wind_direction"],
                "is_dome": weather["is_dome"],
                "away_season_win_pct": _float_or_default((away_record.get("leagueRecord") or {}).get("pct"), 0.5),
                "home_season_win_pct": _float_or_default((home_record.get("leagueRecord") or {}).get("pct"), 0.5),
                "away_games_played": safe_int(away_record.get("gamesPlayed")),
                "home_games_played": safe_int(home_record.get("gamesPlayed")),
                "away_prior_season_win_pct": prior_team_win_pct.get(away_team_id, 0.5),
                "home_prior_season_win_pct": prior_team_win_pct.get(home_team_id, 0.5),
                "away_starter_id": away_pitcher_id,
                "home_starter_id": home_pitcher_id,
                "away_starter_name": ((probable_pitchers.get("away") or {}).get("fullName")) or game.get("away_probable_pitcher") or "TBD",
                "home_starter_name": ((probable_pitchers.get("home") or {}).get("fullName")) or game.get("home_probable_pitcher") or "TBD",
                "away_starter_hand": away_pitcher["hand"],
                "home_starter_hand": home_pitcher["hand"],
                "away_starter_era": away_pitcher["era"],
                "home_starter_era": home_pitcher["era"],
                "away_starter_fip": away_pitcher["fip"],
                "home_starter_fip": home_pitcher["fip"],
                "away_starter_whip": away_pitcher["whip"],
                "home_starter_whip": home_pitcher["whip"],
                "away_starter_ip": away_pitcher["ip"],
                "home_starter_ip": home_pitcher["ip"],
                "away_starter_starts": away_pitcher["starts"],
                "home_starter_starts": home_pitcher["starts"],
                # Raw starter counts so v2 features can compute K/9, BB/9, HR/9.
                "away_starter_strikeouts": away_pitcher["strikeouts"],
                "home_starter_strikeouts": home_pitcher["strikeouts"],
                "away_starter_walks": away_pitcher["walks"],
                "home_starter_walks": home_pitcher["walks"],
                "away_starter_home_runs": away_pitcher["home_runs"],
                "home_starter_home_runs": home_pitcher["home_runs"],
                "away_starter_batters_faced": away_pitcher["batters_faced"],
                "home_starter_batters_faced": home_pitcher["batters_faced"],
                "away_starter_recent_era": away_pitcher["last_5_starts_era"],
                "home_starter_recent_era": home_pitcher["last_5_starts_era"],
                "away_starter_last_5_starts_era": away_pitcher["last_5_starts_era"],
                "home_starter_last_5_starts_era": home_pitcher["last_5_starts_era"],
                "away_starter_last_5_starts_whip": away_pitcher["last_5_starts_whip"],
                "home_starter_last_5_starts_whip": home_pitcher["last_5_starts_whip"],
                "away_starter_last_5_starts_count": away_pitcher["last_5_starts_count"],
                "home_starter_last_5_starts_count": home_pitcher["last_5_starts_count"],
                "away_prior_starter_era": away_pitcher["prior_era"],
                "home_prior_starter_era": home_pitcher["prior_era"],
                "away_prior_starter_fip": away_pitcher["prior_fip"],
                "home_prior_starter_fip": home_pitcher["prior_fip"],
                "away_prior_starter_ip": away_pitcher["prior_ip"],
                "home_prior_starter_ip": home_pitcher["prior_ip"],
                "away_lineup_ops_proxy": away_lineup["ops"],
                "home_lineup_ops_proxy": home_lineup["ops"],
                "away_lineup_obp_proxy": away_lineup["obp"],
                "home_lineup_obp_proxy": home_lineup["obp"],
                "away_lineup_slg_proxy": away_lineup["slg"],
                "home_lineup_slg_proxy": home_lineup["slg"],
                "away_lineup_sample_games": away_lineup["sample_games"],
                "home_lineup_sample_games": home_lineup["sample_games"],
                "away_form_7d_win_pct": away_context["form_7d_win_pct"],
                "home_form_7d_win_pct": home_context["form_7d_win_pct"],
                "away_form_14d_win_pct": away_context["form_14d_win_pct"],
                "home_form_14d_win_pct": home_context["form_14d_win_pct"],
                "away_form_30d_win_pct": away_context["form_30d_win_pct"],
                "home_form_30d_win_pct": home_context["form_30d_win_pct"],
                "away_form_7d_games": away_context["form_7d_games"],
                "home_form_7d_games": home_context["form_7d_games"],
                "away_form_14d_games": away_context["form_14d_games"],
                "home_form_14d_games": home_context["form_14d_games"],
                "away_form_30d_games": away_context["form_30d_games"],
                "home_form_30d_games": home_context["form_30d_games"],
                # Run-differential-based form used by v2 (stronger than W/L).
                "away_form_7d_run_diff": away_context.get("form_7d_run_diff", 0.0),
                "home_form_7d_run_diff": home_context.get("form_7d_run_diff", 0.0),
                "away_form_14d_run_diff": away_context.get("form_14d_run_diff", 0.0),
                "home_form_14d_run_diff": home_context.get("form_14d_run_diff", 0.0),
                "away_form_30d_run_diff": away_context.get("form_30d_run_diff", 0.0),
                "home_form_30d_run_diff": home_context.get("form_30d_run_diff", 0.0),
                # Season-to-date totals powering v2's Pythagorean estimate.
                "away_runs_scored_season": away_context.get("runs_scored_season", 0.0),
                "home_runs_scored_season": home_context.get("runs_scored_season", 0.0),
                "away_runs_allowed_season": away_context.get("runs_allowed_season", 0.0),
                "home_runs_allowed_season": home_context.get("runs_allowed_season", 0.0),
                "away_bullpen_pitches_1d": away_context["bullpen_pitches_1d"],
                "home_bullpen_pitches_1d": home_context["bullpen_pitches_1d"],
                "away_bullpen_pitches_3d": away_context["bullpen_pitches_3d"],
                "home_bullpen_pitches_3d": home_context["bullpen_pitches_3d"],
                "away_bullpen_era_30d": away_context["bullpen_era_30d"],
                "home_bullpen_era_30d": home_context["bullpen_era_30d"],
                "away_rest_days": away_context["rest_days"],
                "home_rest_days": home_context["rest_days"],
                "away_travel_distance_miles": away_context["travel_distance_miles"],
                "home_travel_distance_miles": home_context["travel_distance_miles"],
                "away_travel_flag": away_context["travel_flag"],
                "home_travel_flag": home_context["travel_flag"],
            }
        )

    frame = pd.DataFrame(rows)

    # Attach market_total_line from the SportsLine map when available. The
    # totals model reads this as a feature; if no map is provided or a game is
    # not matched, feature_engineering.ensure_feature_frame will default-fill
    # it with the league average (8.7) so training and inference stay aligned.
    if market_odds_map and not frame.empty:
        totals: list[float | None] = []
        ml_home: list[float | None] = []
        ml_away: list[float | None] = []
        for _, row in frame.iterrows():
            away = str(row.get("away_team") or "").strip().split()
            home = str(row.get("home_team") or "").strip().split()
            away_key = away[-1].lower() if away else ""
            home_key = home[-1].lower() if home else ""
            mo = market_odds_map.get((away_key, home_key), {})
            line = mo.get("total_line")
            totals.append(float(line) if line is not None else None)
            ml_home.append(float(mo.get("ml_home")) if mo.get("ml_home") is not None else None)
            ml_away.append(float(mo.get("ml_away")) if mo.get("ml_away") is not None else None)
        frame["market_total_line"] = totals
        # v2 features treat these as first-class inputs (vig-free prob + line move).
        frame["home_moneyline"] = ml_home
        frame["away_moneyline"] = ml_away

    return frame

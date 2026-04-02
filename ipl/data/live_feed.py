from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests


TEAM_NAME_MAP = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bengaluru",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "PBKS": "Punjab Kings",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "MI-W": "Mumbai Indians",
    "RCBv2": "Royal Challengers Bengaluru",
}
IPL_TEAMS = set(TEAM_NAME_MAP.values())

SCHEDULE_URLS = (
    "https://scores.iplt20.com/ipl/feeds/102-matchschedule.js",
    "https://scores.iplt20.com/ipl/feeds/284-matchschedule.js",
    "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/102-matchschedule.js",
)
LIVE_SQUAD_URLS = (
    "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/{match_id}-squad.js",
    "https://scores.iplt20.com/ipl/feeds/{match_id}-squad.js",
    "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/{match_id}-Innings1.js",
)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%b %d, %Y")
PLAYER_NAME_KEYS = (
    "PlayerName",
    "PlayerName1",
    "BatsManName",
    "BatsmanName",
    "BowlerName",
    "Name",
    "name",
    "PlayerShortName",
)
TEAM_NAME_KEYS = (
    "TeamName",
    "Team",
    "team",
    "TeamCode",
    "BattingTeamName",
    "BowlingTeamName",
    "HomeTeamName",
    "AwayTeamName",
)
ROLE_KEYS = (
    "PlayerSkill",
    "Role",
    "PlayerRole",
    "PlayerType",
    "Designation",
)
ROLE_ALIASES = {
    "batter": "Batsman",
    "batsman": "Batsman",
    "batting": "Batsman",
    "all rounder": "All-Rounder",
    "all-rounder": "All-Rounder",
    "allrounder": "All-Rounder",
    "bowler": "Bowler",
    "wicket keeper": "Wicket-Keeper",
    "wicket-keeper": "Wicket-Keeper",
    "wicketkeeper": "Wicket-Keeper",
    "keeper": "Wicket-Keeper",
}
TARGET_LIST_KEYS = {
    "Squad",
    "Players",
    "players",
    "Batsmen",
    "Bowlers",
    "BattingCard",
    "BowlingCard",
}


def fetch_today_matches(db_path) -> list[dict]:
    del db_path
    today = datetime.now().astimezone().date()
    last_error: Exception | None = None

    for url in SCHEDULE_URLS:
        try:
            payload = _request_jsonp(url)
            entries = _extract_schedule_entries(payload)
            records = [
                _normalize_schedule_entry(entry)
                for entry in entries
                if isinstance(entry, dict)
            ]
            records = [
                record
                for record in records
                if (
                    record["match_id"]
                    and record["team1"] in IPL_TEAMS
                    and record["team2"] in IPL_TEAMS
                    and record["match_date"]
                )
            ]
            if not records:
                continue
            return _serialize_schedule_records(_select_schedule_records(records, today))
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc

    if last_error is not None:
        print(f"[WARN] IPL schedule feed unavailable: {last_error}")
    else:
        print("[WARN] IPL schedule feed returned no usable matches")
    return []


def fetch_live_squads_for_match(
    match_id,
    db_path,
    season: str = "2026",
) -> bool:
    match_id_text = _normalize_text(match_id)
    if not match_id_text:
        print("[WARN] live squad fetch skipped because match_id was empty")
        return False

    payloads: list[tuple[str, Any]] = []
    last_error: Exception | None = None
    for template in LIVE_SQUAD_URLS:
        url = template.format(match_id=match_id_text)
        try:
            payloads.append((url, _request_jsonp(url)))
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc

    if not payloads:
        print(f"[WARN] live squad feed unavailable for match {match_id_text}: {last_error}")
        return False

    unique_entries = _dedupe_live_players(
        entry
        for _, payload in payloads
        for entry in _extract_live_players(payload)
    )
    if len(unique_entries) < 10:
        print(
            f"[WARN] live squad feed for match {match_id_text} produced only "
            f"{len(unique_entries)} player(s)"
        )
        return False

    fetched_at = datetime.now(timezone.utc).isoformat()
    db_file = Path(db_path)
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_current_squads_table(conn)
        current_rows = list(
            conn.execute(
                """
                SELECT player_name, team, role, is_available, season
                FROM ipl_current_squads
                WHERE season=?
                """,
                (season,),
            ).fetchall()
        )
        prepared_entries: list[tuple[str, str, str, int, str | None]] = []
        live_names_by_team: dict[str, set[str]] = {}

        for entry in unique_entries:
            player_name = entry["player_name"]
            matched = _match_existing_player(current_rows, player_name, entry.get("team", ""))
            team_name = _normalize_team_name(entry.get("team", ""))
            if matched is not None and not team_name:
                team_name = _normalize_team_name(matched["team"])
            if team_name and team_name not in IPL_TEAMS:
                continue
            if not team_name and matched is None:
                continue
            role_name = entry.get("role") or (matched["role"] if matched is not None else "") or "Batsman"
            prepared_entries.append(
                (
                    player_name,
                    team_name,
                    role_name,
                    int(bool(entry.get("is_overseas"))),
                    matched["player_name"] if matched is not None else None,
                )
            )

        if len(prepared_entries) < 10:
            print(
                f"[WARN] live squad feed for match {match_id_text} did not produce enough "
                "IPL squad players"
            )
            return False

        for player_name, team_name, role_name, is_overseas, matched_name in prepared_entries:
            if matched_name is not None:
                canonical_name = matched_name
                matched_team = next(
                    (
                        row["team"]
                        for row in current_rows
                        if row["player_name"] == matched_name
                    ),
                    "",
                )
                conn.execute(
                    """
                    UPDATE ipl_current_squads
                    SET team=?, role=?, is_available=1, fetched_at=?
                    WHERE player_name=? AND season=?
                    """,
                    (
                        team_name or matched_team,
                        role_name,
                        fetched_at,
                        canonical_name,
                        season,
                    ),
                )
            else:
                canonical_name = player_name
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ipl_current_squads (
                        player_name, team, role, is_overseas, is_available, season, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical_name,
                        team_name,
                        role_name,
                        is_overseas,
                        1,
                        season,
                        fetched_at,
                    ),
                )

            if team_name:
                live_names_by_team.setdefault(team_name, set()).add(canonical_name.lower())

        for team_name, live_names in live_names_by_team.items():
            team_rows = conn.execute(
                """
                SELECT player_name
                FROM ipl_current_squads
                WHERE season=? AND team=?
                """,
                (season, team_name),
            ).fetchall()
            for row in team_rows:
                player_name = row["player_name"]
                if player_name.lower() in live_names:
                    continue
                conn.execute(
                    """
                    UPDATE ipl_current_squads
                    SET is_available=0, fetched_at=?
                    WHERE player_name=? AND season=?
                    """,
                    (fetched_at, player_name, season),
                )

        conn.commit()

    source_list = ", ".join(url for url, _ in payloads)
    print(
        f"[INFO] live squad update complete for match {match_id_text} "
        f"from {source_list} with {len(unique_entries)} player(s)"
    )
    return True


def refresh_availability_from_live(match_id, team1, team2, db_path) -> dict:
    team1_name = _normalize_team_name(team1)
    team2_name = _normalize_team_name(team2)
    live_feed_success = fetch_live_squads_for_match(match_id, db_path)

    if not live_feed_success:
        print(
            f"[WARN] live feed failed for match {match_id}; "
            "using current hardcoded availability snapshot"
        )

    with sqlite3.connect(Path(db_path)) as conn:
        _ensure_current_squads_table(conn)
        team1_available = conn.execute(
            """
            SELECT COUNT(*)
            FROM ipl_current_squads
            WHERE season=? AND team=? AND is_available=1
            """,
            ("2026", team1_name),
        ).fetchone()[0]
        team2_available = conn.execute(
            """
            SELECT COUNT(*)
            FROM ipl_current_squads
            WHERE season=? AND team=? AND is_available=1
            """,
            ("2026", team2_name),
        ).fetchone()[0]

    summary = {
        "match_id": str(match_id),
        "team1": team1_name,
        "team2": team2_name,
        "team1_available": int(team1_available),
        "team2_available": int(team2_available),
        "live_feed_success": bool(live_feed_success),
    }
    print(
        f"[INFO] availability for {team1_name} vs {team2_name}: "
        f"{summary['team1_available']} | {summary['team2_available']} "
        f"({'live' if live_feed_success else 'fallback'})"
    )
    return summary


def run_live_feed_update(db_path) -> list[dict]:
    matches = fetch_today_matches(db_path)
    if not matches:
        print("No live IPL matches found today or in the next upcoming slot.")
        return []

    today_iso = datetime.now().astimezone().date().isoformat()
    if any(match.get("match_date") == today_iso for match in matches):
        print(f"Today's IPL matches: {len(matches)}")
    else:
        first_match = matches[0]
        print(
            "Next upcoming match: "
            f"{first_match['team1']} vs {first_match['team2']} @ "
            f"{first_match['venue']} on {first_match['match_date']}"
        )

    enriched_matches: list[dict[str, Any]] = []
    for match in matches:
        summary = refresh_availability_from_live(
            match["match_id"],
            match["team1"],
            match["team2"],
            db_path,
        )
        print(
            f"  {match['team1']} ({summary['team1_available']}) | "
            f"{match['team2']} ({summary['team2_available']}) | "
            f"Live feed: {'OK' if summary['live_feed_success'] else 'FALLBACK'}"
        )
        enriched_matches.append(
            {
                "match_id": match["match_id"],
                "team1": match["team1"],
                "team2": match["team2"],
                "venue": match["venue"],
                "match_date": match["match_date"],
                "team1_available": summary["team1_available"],
                "team2_available": summary["team2_available"],
                "live_feed_success": summary["live_feed_success"],
            }
        )

    return enriched_matches


def _request_jsonp(url: str) -> Any:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
    response.raise_for_status()
    return json.loads(_strip_jsonp(response.text))


def _strip_jsonp(text: str) -> str:
    payload = re.sub(r"^[^(]+\(", "", text.strip()).rstrip()
    if payload.endswith(");"):
        payload = payload[:-2]
    elif payload.endswith(")"):
        payload = payload[:-1]
    return payload.strip()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_team_name(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    return TEAM_NAME_MAP.get(text, TEAM_NAME_MAP.get(text.upper(), text))


def _first_non_empty(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key not in entry:
            continue
        value = entry.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_match_date(value: Any) -> date | None:
    text = _normalize_text(value)
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text[:19]).date()
    except ValueError:
        return None


def _extract_schedule_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("Matchsummary", "matches", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _normalize_schedule_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_id": _normalize_text(_first_non_empty(entry, ("MatchID", "matchId", "id"))),
        "team1": _normalize_team_name(
            _first_non_empty(
                entry,
                (
                    "HomeTeamName",
                    "FirstBattingTeamName",
                    "Team1",
                    "team1",
                    "HomeTeam",
                    "Team1Abbr",
                    "FirstBattingTeamCode",
                ),
            )
        ),
        "team2": _normalize_team_name(
            _first_non_empty(
                entry,
                (
                    "AwayTeamName",
                    "SecondBattingTeamName",
                    "Team2",
                    "team2",
                    "AwayTeam",
                    "Team2Abbr",
                    "SecondBattingTeamCode",
                ),
            )
        ),
        "venue": _normalize_text(_first_non_empty(entry, ("Venue", "venue", "GroundName"))),
        "match_date": _parse_match_date(
            _first_non_empty(entry, ("MatchDate", "date", "StartDate", "MATCH_COMMENCE_START_DATE"))
        ),
        "status": _normalize_text(_first_non_empty(entry, ("MatchStatus", "status"))),
    }


def _select_schedule_records(
    records: list[dict[str, Any]],
    today: date,
) -> list[dict[str, Any]]:
    today_records = [record for record in records if record["match_date"] == today]
    if today_records:
        return today_records

    future_dates = sorted({record["match_date"] for record in records if record["match_date"] > today})
    if future_dates:
        target_date = future_dates[0]
        return [record for record in records if record["match_date"] == target_date]

    past_dates = sorted({record["match_date"] for record in records if record["match_date"] < today})
    if past_dates:
        target_date = past_dates[-1]
        return [record for record in records if record["match_date"] == target_date]

    return records


def _serialize_schedule_records(records: list[dict[str, Any]]) -> list[dict]:
    return [
        {
            "match_id": record["match_id"],
            "team1": record["team1"],
            "team2": record["team2"],
            "venue": record["venue"],
            "match_date": record["match_date"].isoformat(),
            "status": record["status"],
        }
        for record in records
    ]


def _extract_live_players(payload: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("squadA", "squadB", "Squad"):
            value = payload.get(key)
            if isinstance(value, list):
                entries.extend(_extract_player_list(value, _default_team_from_list(value)))

        teams_value = payload.get("teams")
        if isinstance(teams_value, list):
            for team_block in teams_value:
                if not isinstance(team_block, dict):
                    continue
                default_team = _pick_team_name(
                    team_block.get("TeamName"),
                    team_block.get("team"),
                    team_block.get("name"),
                    team_block.get("TeamCode"),
                )
                players = (
                    team_block.get("players")
                    or team_block.get("Players")
                    or team_block.get("Squad")
                )
                if isinstance(players, list):
                    entries.extend(_extract_player_list(players, default_team))

        innings_value = payload.get("Innings")
        if isinstance(innings_value, list):
            for innings_block in innings_value:
                entries.extend(_extract_players_from_innings_block(innings_block))

        for key, value in payload.items():
            if isinstance(value, dict) and key.lower().startswith("innings"):
                entries.extend(_extract_players_from_innings_block(value))

    entries.extend(_walk_player_lists(payload))
    return entries


def _default_team_from_list(rows: list[Any]) -> str:
    for row in rows:
        if not isinstance(row, dict):
            continue
        team_name = _pick_team_name(*(row.get(key) for key in TEAM_NAME_KEYS))
        if team_name:
            return team_name
    return ""


def _extract_player_list(
    rows: list[Any],
    default_team: str = "",
    default_role: str = "",
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry = _player_entry_from_row(row, default_team, default_role)
        if entry:
            entries.append(entry)
    return entries


def _extract_players_from_innings_block(block: Any) -> list[dict[str, Any]]:
    if not isinstance(block, dict):
        return []
    entries: list[dict[str, Any]] = []
    entries.extend(
        _extract_player_list(block.get("Batsmen", []), _pick_team_name(block.get("BattingTeamName")), "Batsman")
    )
    entries.extend(
        _extract_player_list(block.get("Bowlers", []), _pick_team_name(block.get("BowlingTeamName")), "Bowler")
    )
    entries.extend(_extract_player_list(block.get("BattingCard", []), "", "Batsman"))
    entries.extend(_extract_player_list(block.get("BowlingCard", []), "", "Bowler"))
    return entries


def _walk_player_lists(obj: Any, default_team: str = "") -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        team_hint = _pick_team_name(*(obj.get(key) for key in TEAM_NAME_KEYS), default_team)
        for key in TARGET_LIST_KEYS:
            value = obj.get(key)
            if isinstance(value, list):
                role_hint = "Bowler" if "Bowl" in key else ""
                if "Bat" in key:
                    role_hint = "Batsman"
                entries.extend(_extract_player_list(value, team_hint, role_hint))
        for value in obj.values():
            entries.extend(_walk_player_lists(value, team_hint))
    elif isinstance(obj, list):
        for item in obj:
            entries.extend(_walk_player_lists(item, default_team))
    return entries


def _player_entry_from_row(
    row: dict[str, Any],
    default_team: str = "",
    default_role: str = "",
) -> dict[str, Any]:
    raw_name = ""
    for key in PLAYER_NAME_KEYS:
        raw_name = _normalize_text(row.get(key))
        if raw_name:
            break
    if not raw_name:
        return {}

    player_name, suffix_role = _normalize_player_name(raw_name)
    if not player_name:
        return {}

    role_name = _normalize_role(
        _first_non_empty(row, ROLE_KEYS),
        raw_name,
        default_role,
        row.get("IsWK"),
    )
    team_name = _pick_team_name(*(row.get(key) for key in TEAM_NAME_KEYS), default_team)

    return {
        "player_name": player_name,
        "team": team_name,
        "role": suffix_role or role_name,
        "is_overseas": int(str(row.get("IsNonDomestic") or "0").strip() == "1"),
    }


def _normalize_player_name(value: Any) -> tuple[str, str]:
    raw_name = _normalize_text(value)
    if not raw_name:
        return "", ""
    suffix_role = "Wicket-Keeper" if re.search(r"\((?:wk|wkt)\)", raw_name, flags=re.I) else ""
    clean_name = re.sub(
        r"\s*\((?:c|wk|vc|wkt|ip|rp|captain|vice[- ]captain)\)",
        "",
        raw_name,
        flags=re.I,
    )
    return _normalize_text(clean_name), suffix_role


def _normalize_role(
    value: Any,
    raw_name: str = "",
    default_role: str = "",
    is_wk: Any = None,
) -> str:
    if str(is_wk or "").strip() == "1" or re.search(r"\((?:wk|wkt)\)", raw_name, flags=re.I):
        return "Wicket-Keeper"

    for candidate in (value, default_role):
        text = _normalize_text(candidate).lower().replace("_", " ")
        if not text:
            continue
        if "wicket" in text or text in {"wk", "keeper"}:
            return "Wicket-Keeper"
        if "all round" in text:
            return "All-Rounder"
        if "bowler" in text or "fast" in text or "spin" in text:
            return "Bowler"
        if "bat" in text:
            return "Batsman"
        mapped = ROLE_ALIASES.get(text)
        if mapped:
            return mapped
    return "Batsman"


def _pick_team_name(*values: Any) -> str:
    for value in values:
        team_name = _normalize_team_name(value)
        if team_name:
            return team_name
    return ""


def _dedupe_live_players(entries: Any) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        player_name = _normalize_text(entry.get("player_name"))
        if not player_name:
            continue
        key = player_name.lower()
        if key not in deduped:
            deduped[key] = {
                "player_name": player_name,
                "team": _normalize_team_name(entry.get("team")),
                "role": _normalize_text(entry.get("role")) or "Batsman",
                "is_overseas": int(bool(entry.get("is_overseas"))),
            }
            continue

        existing = deduped[key]
        if not existing["team"] and entry.get("team"):
            existing["team"] = _normalize_team_name(entry.get("team"))
        if existing["role"] == "Batsman" and entry.get("role") not in ("", "Batsman", None):
            existing["role"] = _normalize_text(entry.get("role"))
        if not existing["is_overseas"] and entry.get("is_overseas"):
            existing["is_overseas"] = 1
    return list(deduped.values())


def _match_existing_player(
    current_rows: list[sqlite3.Row],
    live_name: str,
    live_team: str = "",
) -> sqlite3.Row | None:
    live_text = _normalize_text(live_name)
    live_lower = live_text.lower()
    live_team_name = _normalize_team_name(live_team)

    candidates = current_rows
    if live_team_name:
        team_matches = [
            row
            for row in current_rows
            if _normalize_team_name(row["team"]) == live_team_name
        ]
        if team_matches:
            candidates = team_matches

    for row in candidates:
        if row["player_name"] == live_text:
            return row
    for row in candidates:
        if row["player_name"].lower() == live_lower:
            return row
    for row in candidates:
        row_lower = row["player_name"].lower()
        if live_lower in row_lower or row_lower in live_lower:
            return row
    for row in candidates:
        if _token_prefix_match(row["player_name"], live_text):
            return row
    return None


def _token_prefix_match(left: str, right: str) -> bool:
    left_tokens = re.findall(r"[a-z0-9]+", left.lower())
    right_tokens = re.findall(r"[a-z0-9]+", right.lower())
    if len(left_tokens) != len(right_tokens) or not left_tokens:
        return False
    for left_token, right_token in zip(left_tokens, right_tokens):
        if left_token == right_token:
            continue
        if len(left_token) >= 4 and left_token.startswith(right_token):
            continue
        if len(right_token) >= 4 and right_token.startswith(left_token):
            continue
        return False
    return True


def _ensure_current_squads_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ipl_current_squads (
            player_name   TEXT,
            team          TEXT,
            role          TEXT,
            is_overseas   INTEGER,
            is_available  INTEGER DEFAULT 1,
            season        TEXT,
            fetched_at    TEXT,
            PRIMARY KEY (player_name, season)
        )
        """
    )


if __name__ == "__main__":
    import sys, os

    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    from ipl.data_loader import _default_db_path

    db = _default_db_path()
    matches = run_live_feed_update(db)
    print(f"\nReady for prediction: {len(matches)} match(es)")
    for m in matches:
        print(f"  {m['team1']} vs {m['team2']} @ {m['venue']} on {m['match_date']}")
        print(
            f"  Available: {m['team1']} ({m['team1_available']}) | "
            f"{m['team2']} ({m['team2_available']})"
        )
        print(f"  Live feed: {'OK' if m['live_feed_success'] else 'FALLBACK'}")

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipl.data_loader import _default_db_path


SCHEDULE_URLS = (
    "https://scores.iplt20.com/ipl/feeds/102-matchschedule.js",
    "https://scores.iplt20.com/ipl/feeds/284-matchschedule.js",
)
SQUAD_URL_TEMPLATE = (
    "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/"
    "{match_id}-squad.js"
)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
PLAYER_NAME_ALIASES = {
    "K L Rahul": "KL Rahul",
}
FORCED_PLAYER_TEAMS = {
    "Sanju Samson": "Rajasthan Royals",
    "KL Rahul": "Delhi Capitals",
    "Noor Ahmad": "Chennai Super Kings",
}
REMOVED_PLAYERS = {
    "AB de Villiers",
}


FALLBACK_SQUADS: dict[str, list[tuple[str, str, int]]] = {
    "Mumbai Indians": [
        ("Rohit Sharma", "Batsman", 0),
        ("Suryakumar Yadav", "Batsman", 0),
        ("Hardik Pandya", "All-Rounder", 0),
        ("Jasprit Bumrah", "Bowler", 0),
        ("Tilak Varma", "Batsman", 0),
        ("Naman Dhir", "All-Rounder", 0),
        ("Trent Boult", "Bowler", 1),
        ("Ryan Rickelton", "Batsman", 1),
        ("Will Jacks", "All-Rounder", 1),
        ("Deepak Chahar", "Bowler", 0),
        ("Karn Sharma", "Bowler", 0),
    ],
    "Chennai Super Kings": [
        ("Ruturaj Gaikwad", "Batsman", 0),
        ("MS Dhoni", "Wicket-Keeper", 0),
        ("Ravindra Jadeja", "All-Rounder", 0),
        ("Matheesha Pathirana", "Bowler", 1),
        ("Devon Conway", "Batsman", 1),
        ("Shivam Dube", "All-Rounder", 0),
        ("Rachin Ravindra", "All-Rounder", 1),
        ("Noor Ahmad", "Bowler", 1),
        ("Khaleel Ahmed", "Bowler", 0),
        ("Vijay Shankar", "All-Rounder", 0),
        ("MS Wade", "Wicket-Keeper", 1),
    ],
    "Royal Challengers Bengaluru": [
        ("Virat Kohli", "Batsman", 0),
        ("Rajat Patidar", "Batsman", 0),
        ("Phil Salt", "Batsman", 1),
        ("Josh Hazlewood", "Bowler", 1),
        ("Liam Livingstone", "All-Rounder", 1),
        ("Krunal Pandya", "All-Rounder", 0),
        ("Yash Dayal", "Bowler", 0),
        ("Bhuvneshwar Kumar", "Bowler", 0),
        ("Tim David", "Batsman", 1),
        ("Swapnil Singh", "All-Rounder", 0),
        ("Suyash Sharma", "Bowler", 0),
    ],
    "Kolkata Knight Riders": [
        ("Ajinkya Rahane", "Batsman", 0),
        ("Sunil Narine", "All-Rounder", 1),
        ("Andre Russell", "All-Rounder", 1),
        ("Varun Chakravarthy", "Bowler", 0),
        ("Venkatesh Iyer", "All-Rounder", 0),
        ("Spencer Johnson", "Bowler", 1),
        ("Quinton de Kock", "Wicket-Keeper", 1),
        ("Manish Pandey", "Batsman", 0),
        ("Moeen Ali", "All-Rounder", 1),
        ("Harshit Rana", "Bowler", 0),
        ("Rinku Singh", "Batsman", 0),
    ],
    "Delhi Capitals": [
        ("KL Rahul", "Wicket-Keeper", 0),
        ("Jake Fraser-McGurk", "Batsman", 1),
        ("Axar Patel", "All-Rounder", 0),
        ("Mitchell Starc", "Bowler", 1),
        ("Tristan Stubbs", "Batsman", 1),
        ("Faf du Plessis", "Batsman", 1),
        ("Kuldeep Yadav", "Bowler", 0),
        ("Mukesh Kumar", "Bowler", 0),
        ("Ashutosh Sharma", "Batsman", 0),
        ("Harry Brook", "Batsman", 1),
        ("Karun Nair", "Batsman", 0),
    ],
    "Rajasthan Royals": [
        ("Sanju Samson", "Wicket-Keeper", 0),
        ("Yashasvi Jaiswal", "Batsman", 0),
        ("Riyan Parag", "All-Rounder", 0),
        ("Shimron Hetmyer", "Batsman", 1),
        ("Jos Buttler", "Batsman", 1),
        ("Wanindu Hasaranga", "All-Rounder", 1),
        ("Trent Boult", "Bowler", 1),
        ("Sandeep Sharma", "Bowler", 0),
        ("Dhruv Jurel", "Wicket-Keeper", 0),
        ("Maheesh Theekshana", "Bowler", 1),
        ("Kumar Kartikeya", "Bowler", 0),
    ],
    "Sunrisers Hyderabad": [
        ("Travis Head", "Batsman", 1),
        ("Heinrich Klaasen", "Wicket-Keeper", 1),
        ("Pat Cummins", "All-Rounder", 1),
        ("Abhishek Sharma", "Batsman", 0),
        ("Ishan Kishan", "Wicket-Keeper", 0),
        ("Adam Markram", "Batsman", 1),
        ("Nitish Kumar Reddy", "All-Rounder", 0),
        ("Harshal Patel", "Bowler", 0),
        ("Mohammed Shami", "Bowler", 0),
        ("Rahul Tripathi", "Batsman", 0),
        ("T Natarajan", "Bowler", 0),
    ],
    "Punjab Kings": [
        ("Shreyas Iyer", "Batsman", 0),
        ("Shashank Singh", "All-Rounder", 0),
        ("Glenn Maxwell", "All-Rounder", 1),
        ("Marco Jansen", "Bowler", 1),
        ("Arshdeep Singh", "Bowler", 0),
        ("Josh Inglis", "Wicket-Keeper", 1),
        ("Yuzvendra Chahal", "Bowler", 0),
        ("Nehal Wadhera", "Batsman", 0),
        ("Harpreet Brar", "All-Rounder", 0),
        ("Azmatullah Omarzai", "All-Rounder", 1),
        ("Liam Livingstone", "All-Rounder", 1),
    ],
    "Gujarat Titans": [
        ("Shubman Gill", "Batsman", 0),
        ("Jos Buttler", "Batsman", 1),
        ("Rashid Khan", "All-Rounder", 1),
        ("Mohammed Siraj", "Bowler", 0),
        ("Washington Sundar", "All-Rounder", 0),
        ("Shahrukh Khan", "Batsman", 0),
        ("Gerald Coetzee", "Bowler", 1),
        ("Rahul Tewatia", "All-Rounder", 0),
        ("Wriddhiman Saha", "Wicket-Keeper", 0),
        ("Sai Sudharsan", "Batsman", 0),
        ("Noor Ahmad", "Bowler", 1),
    ],
    "Lucknow Super Giants": [
        ("KL Rahul", "Wicket-Keeper", 0),
        ("Nicholas Pooran", "Wicket-Keeper", 1),
        ("Ravi Bishnoi", "Bowler", 0),
        ("Mitchell Marsh", "All-Rounder", 1),
        ("Ayush Badoni", "Batsman", 0),
        ("David Miller", "Batsman", 1),
        ("Avesh Khan", "Bowler", 0),
        ("Mohsin Khan", "Bowler", 0),
        ("Deepak Hooda", "All-Rounder", 0),
        ("Krunal Pandya", "All-Rounder", 0),
        ("Shamar Joseph", "Bowler", 1),
    ],
}


def _strip_jsonp(text: str) -> str:
    payload = text.strip()
    if payload.endswith(";"):
        payload = payload[:-1].rstrip()
    start = payload.find("(")
    end = payload.rfind(")")
    if start != -1 and end != -1 and end > start:
        return payload[start + 1 : end].strip()
    return payload


def _request_text(url: str, retries: int = 2) -> str:
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def _parse_jsonp(text: str) -> Any:
    return json.loads(_strip_jsonp(text))


def _normalize_team(team: str | None) -> str:
    return " ".join(str(team or "").split())


def _clean_player_name(name: str | None) -> str:
    text = " ".join(str(name or "").split()).strip()
    if not text:
        return text
    text = re.sub(r"\s*\((?:c|wk|vc|ip|rp)\)", "", text, flags=re.I)
    text = re.sub(r"\s*\((?:captain|vice[- ]captain)\)", "", text, flags=re.I)
    text = " ".join(text.split()).strip()
    return PLAYER_NAME_ALIASES.get(text, text)


def _map_role(row: dict[str, Any]) -> str:
    skill = str(row.get("PlayerSkill") or "").strip().lower()
    is_wk = str(row.get("IsWK") or "").strip() == "1" or "wicket keeper" in skill
    if is_wk:
        return "Wicket-Keeper"
    if "all round" in skill:
        return "All-Rounder"
    if "bowler" in skill:
        return "Bowler"
    return "Batsman"


def _match_ids_for_season(season: str) -> list[int]:
    season_text = str(season)
    match_ids: list[int] = []
    seen: set[int] = set()
    for url in SCHEDULE_URLS:
        try:
            data = _parse_jsonp(_request_text(url))
        except Exception:
            continue
        match_summary = data.get("Matchsummary") if isinstance(data, dict) else None
        if not isinstance(match_summary, list):
            continue
        for item in match_summary:
            if not isinstance(item, dict):
                continue
            match_date = str(item.get("MatchDate") or "")
            if not match_date.startswith(season_text):
                continue
            try:
                match_id = int(item.get("MatchID"))
            except (TypeError, ValueError):
                continue
            if match_id in seen:
                continue
            seen.add(match_id)
            match_ids.append(match_id)
        if match_ids:
            break
    return match_ids


def _extract_team_rosters(squad_payload: dict[str, Any]) -> dict[str, list[tuple[str, str, int]]]:
    rosters: dict[str, list[tuple[str, str, int]]] = {}
    for side in ("squadA", "squadB"):
        players = squad_payload.get(side)
        if not isinstance(players, list) or not players:
            continue
        team_name = _normalize_team(players[0].get("TeamName"))
        roster: list[tuple[str, str, int]] = []
        for row in players:
            if not isinstance(row, dict):
                continue
            player_name = _clean_player_name(row.get("PlayerName"))
            if not player_name:
                continue
            role = _map_role(row)
            is_overseas = 1 if str(row.get("IsNonDomestic") or "").strip() == "1" else 0
            roster.append((player_name, role, is_overseas))
        if team_name and roster:
            rosters[team_name] = roster
    return rosters


def _persist_squads(
    db_path: str | os.PathLike[str],
    rows: list[tuple[str, str, str, int, int, str, str]],
) -> None:
    seasons = sorted({row[5] for row in rows})
    con = sqlite3.connect(db_path)
    try:
        con.execute(
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
        for season in seasons:
            con.execute("DELETE FROM ipl_current_squads WHERE season=?", (season,))
        con.executemany(
            """
            INSERT OR REPLACE INTO ipl_current_squads (
                player_name, team, role, is_overseas, is_available, season, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
    finally:
        con.close()


def _clear_season_rows(
    db_path: str | os.PathLike[str],
    season: str,
) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
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
        con.execute("DELETE FROM ipl_current_squads WHERE season=?", (season,))
        con.commit()
    finally:
        con.close()


def _sanitize_roster_map(
    roster_by_team: dict[str, list[tuple[str, str, int]]],
    supplement_forced_players: bool = True,
) -> dict[str, list[tuple[str, str, int]]]:
    sanitized: dict[str, list[tuple[str, str, int]]] = OrderedDict()
    seen_players: set[str] = set()

    for team, roster in roster_by_team.items():
        for player_name, role, is_overseas in roster:
            clean_name = _clean_player_name(player_name)
            if not clean_name or clean_name in REMOVED_PLAYERS:
                continue
            forced_team = FORCED_PLAYER_TEAMS.get(clean_name, team)
            sanitized.setdefault(forced_team, [])
            if clean_name in seen_players:
                continue
            sanitized[forced_team].append((clean_name, role, is_overseas))
            seen_players.add(clean_name)

    if supplement_forced_players:
        for team, players in FALLBACK_SQUADS.items():
            for player_name, role, is_overseas in players:
                clean_name = _clean_player_name(player_name)
                if not clean_name or clean_name in REMOVED_PLAYERS:
                    continue
                forced_team = FORCED_PLAYER_TEAMS.get(clean_name, team)
                if clean_name in seen_players:
                    continue
                if clean_name in FORCED_PLAYER_TEAMS:
                    sanitized.setdefault(forced_team, []).append((clean_name, role, is_overseas))
                    seen_players.add(clean_name)

    return sanitized


def _fallback_rows(season: str) -> list[tuple[str, str, str, int, int, str, str]]:
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[tuple[str, str, str, int, int, str, str]] = []
    fallback_map = _sanitize_roster_map(FALLBACK_SQUADS, supplement_forced_players=False)
    for team, players in fallback_map.items():
        for player_name, role, is_overseas in players:
            rows.append(
                (
                    player_name,
                    team,
                    role,
                    is_overseas,
                    1,
                    season,
                    fetched_at,
                )
            )
    return rows


def validate_squad_integrity(
    db_path: str | os.PathLike[str],
    season: str = "2026",
) -> dict[str, Any]:
    con = sqlite3.connect(db_path)
    try:
        total_players = con.execute(
            """
            SELECT COUNT(*)
            FROM ipl_current_squads
            WHERE season=? AND is_available=1
            """,
            (season,),
        ).fetchone()[0]
        per_team_counts = dict(
            con.execute(
                """
                SELECT team, COUNT(*)
                FROM ipl_current_squads
                WHERE season=? AND is_available=1
                GROUP BY team
                ORDER BY team
                """,
                (season,),
            ).fetchall()
        )
        duplicate_rows = con.execute(
            """
            SELECT player_name, GROUP_CONCAT(DISTINCT team), COUNT(DISTINCT team)
            FROM ipl_current_squads
            WHERE season=?
            GROUP BY player_name
            HAVING COUNT(DISTINCT team) > 1
            ORDER BY player_name
            """,
            (season,),
        ).fetchall()
    finally:
        con.close()

    duplicate_players = [
        {"player_name": player_name, "teams": teams, "team_count": team_count}
        for player_name, teams, team_count in duplicate_rows
    ]
    if duplicate_players:
        print(f"FAIL: squad integrity for {season}")
        print(f"per-team counts: {per_team_counts}")
        print(f"duplicate players: {duplicate_players}")
    else:
        print(f"PASS: squad integrity for {season}")
        print(f"per-team counts: {per_team_counts}")

    return {
        "total_players": int(total_players),
        "per_team_counts": per_team_counts,
        "duplicate_players": duplicate_players,
    }


def fetch_current_squads(season: str = "2026") -> None:
    db_path = _default_db_path()
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[tuple[str, str, str, int, int, str, str]] = []
    _clear_season_rows(db_path, season)

    try:
        match_ids = _match_ids_for_season(season)
        if not match_ids:
            raise RuntimeError("No match IDs found for requested season")

        seen_teams: set[str] = set()
        roster_by_team: OrderedDict[str, list[tuple[str, str, int]]] = OrderedDict()

        for match_id in match_ids:
            if len(seen_teams) >= 10:
                break
            try:
                squad_payload = _parse_jsonp(_request_text(SQUAD_URL_TEMPLATE.format(match_id=match_id)))
            except Exception:
                continue
            if not isinstance(squad_payload, dict):
                continue
            rosters = _extract_team_rosters(squad_payload)
            for team, roster in rosters.items():
                if team in seen_teams:
                    continue
                seen_teams.add(team)
                roster_by_team[team] = roster
                if len(seen_teams) >= 10:
                    break

        if len(seen_teams) < 10:
            raise RuntimeError("Live squad feed did not produce all 10 teams")

        sanitized_map = _sanitize_roster_map(roster_by_team)
        for team, roster in sanitized_map.items():
            for player_name, role, is_overseas in roster:
                rows.append((player_name, team, role, is_overseas, 1, season, fetched_at))
    except Exception:
        rows = _fallback_rows(season)

    _persist_squads(db_path, rows)
    report = validate_squad_integrity(db_path, season)
    if report["duplicate_players"]:
        raise RuntimeError(f"Duplicate players across teams detected: {report['duplicate_players']}")


def get_available_players(team, db_path):
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT player_name
            FROM ipl_current_squads
            WHERE team=? AND season=? AND is_available=1
            ORDER BY player_name
            """,
            (team, "2026"),
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        con.close()


if __name__ == "__main__":
    fetch_current_squads()
    con = sqlite3.connect(_default_db_path())
    try:
        for team in con.execute(
            "SELECT DISTINCT team FROM ipl_current_squads WHERE season='2026' ORDER BY team"
        ).fetchall():
            players = con.execute(
                "SELECT player_name, role, is_overseas FROM ipl_current_squads "
                "WHERE team=? AND season='2026' AND is_available=1",
                (team[0],),
            ).fetchall()
            print(f"\n{team[0]} ({len(players)} players):")
            for p in players:
                print(f"  {p[0]} [{p[1]}]{'  *overseas*' if p[2] else ''}")

        duplicates = con.execute(
            """
            SELECT player_name, GROUP_CONCAT(DISTINCT team), COUNT(DISTINCT team)
            FROM ipl_current_squads
            WHERE season='2026'
            GROUP BY player_name
            HAVING COUNT(DISTINCT team) > 1
            ORDER BY player_name
            """
        ).fetchall()
        print("\nMulti-team duplicates:")
        if duplicates:
            for row in duplicates:
                print(row)
        else:
            print("none")
    finally:
        con.close()

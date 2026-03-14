#!/usr/bin/env python3
"""
Local auto-grading service for PickLedger.

Runs a small HTTP server that accepts picks from the dashboard,
fetches completed game results from ESPN's public scoreboard endpoints,
and returns graded outcomes.

Usage:
  python3 pickgrader_server.py
Then click REFRESH in pickledger.html.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

HOST = "127.0.0.1"
PORT = 8765

SPORT_TO_ESPNSLUG = {
    "NBA": ("basketball", "nba"),
    "NHL": ("hockey", "nhl"),
    "MLB": ("baseball", "mlb"),
    "EPL": ("soccer", "eng.1"),
    "WBC": ("baseball", "world-baseball-classic"),
}

USER_AGENT = "PickLedgerAutoGrader/1.0"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_pick_date(date_text: str, year: int) -> str | None:
    try:
        dt = datetime.strptime(f"{date_text} {year}", "%b %d %Y")
        return dt.strftime("%Y%m%d")
    except ValueError:
        return None


def fetch_scoreboard(sport: str, league: str, yyyymmdd: str) -> dict[str, Any] | None:
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
        f"?dates={yyyymmdd}"
    )
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def competitor_fields(comp: dict[str, Any]) -> list[str]:
    team = comp.get("team", {})
    out = [
        str(team.get("displayName", "")),
        str(team.get("shortDisplayName", "")),
        str(team.get("name", "")),
        str(team.get("abbreviation", "")),
    ]
    return [f for f in out if f]


def team_matches_competitor(team_text: str, comp: dict[str, Any]) -> bool:
    t = normalize(team_text)
    if not t:
        return False

    for field in competitor_fields(comp):
        nf = normalize(field)
        if t == nf:
            return True
        if t in nf:
            return True
        if nf in t and len(nf) > 2:
            return True

    # Last-token fallback for names like "Knicks", "Blues", "Senators".
    t_tokens = t.split()
    if t_tokens:
        last = t_tokens[-1]
        if len(last) >= 3:
            for field in competitor_fields(comp):
                f_tokens = normalize(field).split()
                if last in f_tokens:
                    return True

    return False


def get_games(scoreboard: dict[str, Any], completed_only: bool) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for event in scoreboard.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp0 = comps[0]
        status = comp0.get("status", {}).get("type", {})
        if completed_only and not status.get("completed", False):
            continue

        competitors = comp0.get("competitors", [])
        if len(competitors) != 2:
            continue

        parsed = []
        valid = True
        for c in competitors:
            try:
                parsed.append({
                    "raw": c,
                    "score": int(c.get("score", "0")),
                    "homeAway": c.get("homeAway", ""),
                })
            except (ValueError, TypeError):
                valid = False
                break
        if not valid:
            continue

        start_time = str(comp0.get("date") or event.get("date") or "")
        games.append({"competitors": parsed, "startTime": start_time})
    return games


def parse_matchup(pick_text: str) -> tuple[str, str] | None:
    m = re.search(r"\(([^)]+)\)", pick_text)
    if not m:
        return None
    inside = m.group(1)
    parts = re.split(r"\s+(?:vs|@)\s+", inside, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def find_game_for_pick(games: list[dict[str, Any]], pick_text: str) -> dict[str, Any] | None:
    matchup = parse_matchup(pick_text)
    if not matchup:
        return None

    team_a, team_b = matchup
    for game in games:
        c1 = game["competitors"][0]["raw"]
        c2 = game["competitors"][1]["raw"]

        direct = team_matches_competitor(team_a, c1) and team_matches_competitor(team_b, c2)
        reverse = team_matches_competitor(team_a, c2) and team_matches_competitor(team_b, c1)
        if direct or reverse:
            return game

    return None


def resolve_team_score(game: dict[str, Any], team_text: str) -> tuple[int, int] | None:
    comps = game["competitors"]
    for idx, c in enumerate(comps):
        if team_matches_competitor(team_text, c["raw"]):
            opp = comps[1 - idx]
            return c["score"], opp["score"]
    return None


def parse_line(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, TypeError):
        return None


def grade_pick(pick: dict[str, Any], game: dict[str, Any]) -> str:
    pick_text = str(pick.get("pick", ""))
    head = pick_text.split("(", 1)[0].strip()
    lower = head.lower()

    total_points = game["competitors"][0]["score"] + game["competitors"][1]["score"]

    # Full-game totals (Over/Under X)
    m_total = re.search(r"\b(over|under)\s+(\d+(?:\.\d+)?)\b", lower)
    if m_total and "team total" not in lower and not lower.endswith(" tg"):
        side = m_total.group(1)
        line = float(m_total.group(2))
        if total_points == line:
            return "push"
        if side == "over":
            return "win" if total_points > line else "loss"
        return "win" if total_points < line else "loss"

    # Team total over/under, e.g. "Korea Team Total Over 9.5"
    m_team_total = re.search(r"^(.*?)\s+team total\s+(over|under)\s+(\d+(?:\.\d+)?)", lower)
    if m_team_total:
        team_label = m_team_total.group(1).strip()
        side = m_team_total.group(2)
        line = float(m_team_total.group(3))
        resolved = resolve_team_score(game, team_label)
        if resolved is None:
            return "pending"
        team_score = resolved[0]
        if team_score == line:
            return "push"
        if side == "over":
            return "win" if team_score > line else "loss"
        return "win" if team_score < line else "loss"

    # Team goals shorthand, e.g. "Senators Over 3 TG"
    m_tg = re.search(r"^(.*?)\s+(over|under)\s+(\d+(?:\.\d+)?)\s*tg\b", lower)
    if m_tg:
        team_label = m_tg.group(1).strip()
        side = m_tg.group(2)
        line = float(m_tg.group(3))
        resolved = resolve_team_score(game, team_label)
        if resolved is None:
            return "pending"
        team_score = resolved[0]
        if team_score == line:
            return "push"
        if side == "over":
            return "win" if team_score > line else "loss"
        return "win" if team_score < line else "loss"

    # Skip 1H / partial-game markets for now.
    if re.search(r"\b1h\b|first half|period", lower):
        return "pending"

    # Draw pick (soccer): "Draw (Team A vs Team B)"
    if re.match(r"^draw$", lower):
        c0 = game["competitors"][0]["score"]
        c1 = game["competitors"][1]["score"]
        return "win" if c0 == c1 else "loss"

    # Both Teams to Score: "BTTS Yes" or "BTTS No"
    m_btts = re.match(r"^btts\s+(yes|no)$", lower)
    if m_btts:
        c0 = game["competitors"][0]["score"]
        c1 = game["competitors"][1]["score"]
        both_scored = c0 > 0 and c1 > 0
        side = m_btts.group(1)
        if side == "yes":
            return "win" if both_scored else "loss"
        return "win" if not both_scored else "loss"

    # Spread / run line / puck line, e.g. "Knicks -11.5"
    m_spread = re.search(r"^(.*?)\s*([+-]\d+(?:\.\d+)?)\b", head)
    if m_spread:
        team_label = m_spread.group(1).strip()
        try:
            spread = float(m_spread.group(2))
        except ValueError:
            return "pending"
        resolved = resolve_team_score(game, team_label)
        if resolved is None:
            return "pending"
        team_score, opp_score = resolved
        adj = team_score + spread
        if abs(adj - opp_score) < 1e-9:
            return "push"
        return "win" if adj > opp_score else "loss"

    # Moneyline explicit: "Team ML"
    m_ml = re.search(r"^(.*?)\s+ml\b", lower)
    if m_ml:
        team_label = m_ml.group(1).strip()
        resolved = resolve_team_score(game, team_label)
        if resolved is None:
            return "pending"
        team_score, opp_score = resolved
        if team_score == opp_score:
            return "push"
        return "win" if team_score > opp_score else "loss"

    # Fallback: treat leading team label as winner pick.
    fallback_team = re.sub(r"\s*[+-]\d+(?:\.\d+)?\s*$", "", head, flags=re.IGNORECASE).strip()
    if fallback_team:
        resolved = resolve_team_score(game, fallback_team)
        if resolved is None:
            return "pending"
        team_score, opp_score = resolved
        if team_score == opp_score:
            return "push"
        return "win" if team_score > opp_score else "loss"

    return "pending"


def auto_grade(picks: list[dict[str, Any]], existing: dict[str, str], year: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pick in picks:
        pid = str(pick.get("id"))
        if not pid:
            continue
        sport_key = str(pick.get("sport", "")).upper()
        if sport_key not in SPORT_TO_ESPNSLUG:
            continue

        d = parse_pick_date(str(pick.get("date", "")), year)
        if not d:
            continue

        all_grouped.setdefault((sport_key, d), []).append(pick)

        current = existing.get(pid, pick.get("result", "pending"))
        if current != "pending":
            continue

        grouped.setdefault((sport_key, d), []).append(pick)

    graded: dict[str, str] = {}
    start_times: dict[str, str] = {}
    attempted = 0
    board_cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    for (sport_key, d), batch in all_grouped.items():
        sport, league = SPORT_TO_ESPNSLUG[sport_key]
        key = (sport_key, d)
        if key not in board_cache:
            board_cache[key] = fetch_scoreboard(sport, league, d)
        board = board_cache[key]
        if not board:
            continue
        all_games = get_games(board, completed_only=False)

        for pick in batch:
            game = find_game_for_pick(all_games, str(pick.get("pick", "")))
            if game and game.get("startTime"):
                start_times[str(pick["id"])] = str(game["startTime"])

    for (sport_key, d), batch in grouped.items():
        sport, league = SPORT_TO_ESPNSLUG[sport_key]
        key = (sport_key, d)
        if key not in board_cache:
            board_cache[key] = fetch_scoreboard(sport, league, d)
        board = board_cache[key]
        if not board:
            continue
        games = get_games(board, completed_only=True)

        for pick in batch:
            attempted += 1
            game = find_game_for_pick(games, str(pick.get("pick", "")))
            if not game:
                continue
            result = grade_pick(pick, game)
            if result in {"win", "loss", "push"}:
                graded[str(pick["id"])] = result

    return {
        "graded": graded,
        "startTimes": start_times,
        "summary": {
            "attempted": attempted,
            "updated": len(graded),
            "remaining": max(0, attempted - len(graded)),
        },
    }


# ─── Model Runner Helpers ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NBA_MODEL_DIR = os.path.join(BASE_DIR, "NBAPredictionModel")
MLB_MODEL_DIR = os.path.join(BASE_DIR, "MLBPredictionModel")
SCORES24_VENV = os.path.join(BASE_DIR, ".venv", "bin", "python")

# ─── Async Job Store ──────────────────────────────────────────────────────────
# Tracks running/completed model jobs so the frontend can poll for results.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _run_script(python_bin: str, script: str, cwd: str, timeout: int = 300, extra_args: list[str] | None = None) -> str:
    """Run a Python script and return its stdout."""
    cmd = [python_bin, script] + (extra_args or [])
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def _parse_nba_output(output: str) -> list[dict[str, Any]]:
    """Parse NBA model stdout into pick dicts."""
    picks: list[dict[str, Any]] = []

    # The output has GAME: headers and ### prediction blocks separated by ===
    # Strategy: scan line-by-line, tracking current game context
    lines = output.split("\n")
    current_away = ""
    current_home = ""

    for i, line in enumerate(lines):
        # Pick up game header: "GAME: Grizzlies @ Pistons (7:30 pm ET)"
        game_m = re.match(r"^GAME:\s*(.+?)\s*@\s*(.+?)(?:\s*\(|$)", line)
        if game_m:
            current_away = game_m.group(1).strip()
            current_home = game_m.group(2).strip()
            continue

        if not current_away or not current_home:
            continue

        # Extract winner
        winner_m = re.search(r"\*\*Winner:\*\*\s*(.+?)\s*\(Model Prob:\s*([\d.]+)%\)", line)
        if winner_m:
            winner = winner_m.group(1).strip()
            prob = float(winner_m.group(2)) / 100
            # Look ahead for spread, edge, and decision
            spread_val = 0.0
            edge_val = None
            decision = "PASS"
            for j in range(i + 1, min(i + 20, len(lines))):
                sp_m = re.search(r"\*\*Spread:\*\*\s*\S+\s*by\s*([\d.]+)\s*points", lines[j])
                if sp_m:
                    spread_val = float(sp_m.group(1))
                edge_m = re.search(r"\*\*Edge:\*\*\s*\S+\s*([+-]?[\d.]+)%", lines[j])
                if edge_m:
                    edge_val = float(edge_m.group(1))
                dec_m = re.search(r"\*\*Decision:\s*(BET|PASS)", lines[j])
                if dec_m:
                    decision = dec_m.group(1)
                    break

            matchup = f"{current_away} @ {current_home}"
            if spread_val > 0 and decision == "BET":
                pick_text = f"{winner} -{spread_val:.1f} ({matchup})"
            else:
                pick_text = f"{winner} ML ({matchup})"

            # Edge is from the home team perspective; flip sign if bet is on away team
            display_edge = edge_val
            if display_edge is not None and winner != current_home:
                display_edge = -display_edge

            picks.append({
                "source": "NBA Model",
                "pick": pick_text,
                "sport": "NBA",
                "odds": -110,
                "units": 1,
                "probability": prob,
                "edge": display_edge,
                "decision": decision,
            })

        # Over/Under decision: "**O/U Decision: BET OVER**"
        ou_m = re.search(r"\*\*O/U Decision:\s*(BET OVER|BET UNDER|PASS)\*\*", line)
        if ou_m:
            ou_decision_raw = ou_m.group(1)
            # Look back for total line
            model_total = None
            line_val = 225.0
            for j in range(max(0, i - 5), i):
                total_m = re.search(r"\*\*Over/Under:\*\*\s*Model Total\s*([\d.]+)\s*vs\s*Line\s*([\d.]+)", lines[j])
                if total_m:
                    model_total = float(total_m.group(1))
                    line_val = float(total_m.group(2))

            matchup = f"{current_away} @ {current_home}"
            if ou_decision_raw.startswith("BET"):
                ou_side = "Over" if "OVER" in ou_decision_raw else "Under"
                picks.append({
                    "source": "NBA Model",
                    "pick": f"{ou_side} {line_val:.0f} ({matchup})",
                    "sport": "NBA",
                    "odds": -110,
                    "units": 1,
                    "probability": None,
                    "edge": abs(model_total - line_val) if model_total else None,
                    "decision": "BET",
                })
            else:
                picks.append({
                    "source": "NBA Model",
                    "pick": f"O/U {line_val:.0f} ({matchup})",
                    "sport": "NBA",
                    "odds": -110,
                    "units": 1,
                    "probability": None,
                    "edge": None,
                    "decision": "PASS",
                })

    return picks


def _shorten_mlb_name(full_name: str) -> str:
    """Shorten full MLB team names to match ledger style (e.g. 'Tampa Bay Rays' -> 'Rays')."""
    # Multi-word team names that shouldn't be split
    multi_word = {"Red Sox", "White Sox", "Blue Jays"}
    for mw in multi_word:
        if full_name.endswith(mw):
            return mw
    # Default: last word
    parts = full_name.strip().split()
    return parts[-1] if parts else full_name


def _parse_mlb_output(output: str) -> list[dict[str, Any]]:
    """Parse MLB model stdout into pick dicts."""
    picks: list[dict[str, Any]] = []

    # The MLB model uses pipe-delimited output lines:
    # TeamA|TeamB|OddsA|OddsB|ProbA|ProbB
    # Also check for standard format output blocks

    # Try pipe-delimited format first (from the run_predictions.py style)
    pipe_lines = [l.strip() for l in output.split("\n") if l.count("|") >= 2 and not l.startswith("---")]

    if pipe_lines:
        # Track current game context for O/U lines
        current_team_a = ""
        current_team_b = ""

        for line in pipe_lines:
            parts = [p.strip() for p in line.split("|")]

            # O/U line: OU|OVER/UNDER/PASS|line|predicted_total
            if len(parts) >= 4 and parts[0] == "OU":
                ou_side = parts[1]  # OVER, UNDER, or PASS
                try:
                    ou_line = float(parts[2])
                    predicted_total = float(parts[3])
                except (ValueError, IndexError):
                    continue

                if not current_team_a or not current_team_b:
                    continue

                matchup = f"{_shorten_mlb_name(current_team_a)} vs {_shorten_mlb_name(current_team_b)}"
                if ou_side in ("OVER", "UNDER"):
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"{'Over' if ou_side == 'OVER' else 'Under'} {ou_line:.1f} ({matchup})",
                        "sport": "MLB",
                        "odds": -110,
                        "units": 1,
                        "probability": None,
                        "edge": abs(predicted_total - ou_line),
                        "decision": "BET",
                    })
                else:
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"O/U {ou_line:.1f} ({matchup})",
                        "sport": "MLB",
                        "odds": -110,
                        "units": 1,
                        "probability": None,
                        "edge": None,
                        "decision": "PASS",
                    })
                continue

            # Moneyline line: TeamA|TeamB|OddsA|OddsB|ProbA|ProbB
            if len(parts) < 6:
                continue
            try:
                team_a, team_b = parts[0], parts[1]
                odds_a, odds_b = int(float(parts[2])), int(float(parts[3]))
                prob_a, prob_b = float(parts[4]), float(parts[5])
            except (ValueError, IndexError):
                continue

            current_team_a = team_a
            current_team_b = team_b

            # Shorten names to match ledger style
            short_a = _shorten_mlb_name(team_a)
            short_b = _shorten_mlb_name(team_b)

            # Pipe format odds are model-derived (not market), so edge vs those
            # odds is always ~0.  Instead, compare model prob vs a generic 50%
            # market (flat -110 each side) — BET when model gives 55%+ to one side.
            if prob_a >= prob_b:
                bet_team = short_a
                bet_prob = prob_a
                bet_odds = odds_a
                edge = (prob_a - 0.50) * 100  # edge vs 50/50 market
            else:
                bet_team = short_b
                bet_prob = prob_b
                bet_odds = odds_b
                edge = (prob_b - 0.50) * 100

            matchup = f"{short_a} vs {short_b}"
            decision = "BET" if bet_prob >= 0.55 else "PASS"

            picks.append({
                "source": "MLB Model",
                "pick": f"{bet_team} ML ({matchup})",
                "sport": "MLB",
                "odds": bet_odds if bet_odds != 0 else -110,
                "units": 1,
                "probability": bet_prob,
                "edge": edge,
                "decision": decision,
            })
        return picks

    # Fallback: parse structured markdown output (from test_live.py/main.py format)
    blocks = re.split(r"={40,}", output)
    current_away = ""
    current_home = ""

    for block in blocks:
        game_m = re.search(r"###\s*\[(.+?)\]\s*vs\s*\[(.+?)\]", block)
        if game_m:
            current_away = game_m.group(1).strip()
            current_home = game_m.group(2).strip()

        if not current_away or not current_home:
            continue

        winner_m = re.search(r"\*\*Winner:\*\*\s*(.+?)\s*\(Model Prob:\s*([\d.]+)%\)", block)
        decision_m = re.search(r"\*\*Decision:\s*(BET|PASS)(?:\s+on\s+(.+?))?\*\*", block)
        edge_m = re.search(r"\*\*Edge:\*\*\s*\S+\s*([+-]?[\d.]+)%", block)
        total_m = re.search(r"\*\*Total Runs:\*\*\s*([\d.]+)", block)

        if winner_m and decision_m:
            winner = _shorten_mlb_name(winner_m.group(1).strip())
            prob = float(winner_m.group(2)) / 100
            decision = decision_m.group(1)
            edge_val = float(edge_m.group(1)) if edge_m else None
            matchup = f"{_shorten_mlb_name(current_away)} vs {_shorten_mlb_name(current_home)}"

            picks.append({
                "source": "MLB Model",
                "pick": f"{winner} ML ({matchup})",
                "sport": "MLB",
                "odds": -110,
                "units": 1,
                "probability": prob,
                "edge": edge_val,
                "decision": decision,
            })

            # Also emit total runs pick if available
            if total_m:
                total_val = float(total_m.group(1))
                # Default line of 8.5 for MLB
                line = 8.5
                if total_val > line + 0.5:
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"Over {line} ({matchup})",
                        "sport": "MLB",
                        "odds": -110,
                        "units": 1,
                        "probability": None,
                        "edge": total_val - line,
                        "decision": "BET",
                    })
                elif total_val < line - 0.5:
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"Under {line} ({matchup})",
                        "sport": "MLB",
                        "odds": -110,
                        "units": 1,
                        "probability": None,
                        "edge": line - total_val,
                        "decision": "BET",
                    })

        if game_m:
            current_away = ""
            current_home = ""

    return picks


def _parse_scores24_output(output: str) -> list[dict[str, Any]]:
    """Parse Scores24 scraper stdout into pick dicts."""
    picks: list[dict[str, Any]] = []

    # Split by the ━━━ separators
    blocks = re.split(r"━{10,}", output)

    for block in blocks:
        match_m = re.search(r"Match:\s*(.+)", block)
        tip_m = re.search(r"Tip:\s*(.+)", block)
        odds_m = re.search(r"Odds:\s*(.+)", block)
        conf_m = re.search(r"Confidence:\s*(.+)", block)
        league_m = re.search(r"League:\s*(.+)", block)

        if not match_m or not tip_m:
            continue

        matchup = match_m.group(1).strip()
        tip = tip_m.group(1).strip()

        if not tip or tip == "[not found on page]":
            continue

        odds_str = odds_m.group(1).strip() if odds_m else ""
        confidence = conf_m.group(1).strip() if conf_m else ""
        league = league_m.group(1).strip().upper() if league_m else ""

        # Map league to sport code
        sport = "Other"
        if "NBA" in league or "BASKETBALL" in league:
            sport = "NBA"
        elif "NHL" in league or "ICE-HOCKEY" in league or "HOCKEY" in league:
            sport = "NHL"
        elif "MLB" in league or "BASEBALL" in league:
            sport = "MLB"
        elif "PREMIER" in league or "EPL" in league or "SOCCER" in league:
            sport = "EPL"

        # Parse odds to numeric
        odds_val = -110
        if odds_str and odds_str != "[not found on page]":
            try:
                odds_val = int(float(odds_str.replace("+", "")))
            except ValueError:
                odds_val = -110

        # Parse confidence to float
        conf_val = None
        if confidence and confidence != "[not found on page]":
            conf_num = re.search(r"(\d+)", confidence)
            if conf_num:
                conf_val = int(conf_num.group(1))

        # Build pick text: "Tip (Matchup)"
        pick_text = f"{tip} ({matchup})"

        picks.append({
            "source": "Scores24",
            "pick": pick_text,
            "sport": sport,
            "odds": odds_val,
            "units": 3 if conf_val and conf_val >= 80 else 1,
            "probability": conf_val / 100 if conf_val else None,
            "edge": None,
            "decision": "BET",  # All Scores24 tips are presented as BET
        })

    return picks


def run_nba_model(date_str: str | None = None) -> dict[str, Any]:
    """Execute the NBA model and return parsed picks."""
    python_bin = os.path.join(NBA_MODEL_DIR, "venv", "bin", "python")
    if not os.path.exists(python_bin):
        return {"ok": False, "error": f"NBA venv not found at {python_bin}"}

    try:
        output = _run_script(python_bin, "run_live.py", NBA_MODEL_DIR, timeout=300)
        picks = _parse_nba_output(output)
        return {"ok": True, "picks": picks, "raw_lines": len(output.split("\n"))}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "NBA model timed out (5 min limit)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_mlb_model(date_str: str | None = None) -> dict[str, Any]:
    """Execute the MLB model and return parsed picks."""
    python_bin = os.path.join(MLB_MODEL_DIR, "venv", "bin", "python")
    if not os.path.exists(python_bin):
        return {"ok": False, "error": f"MLB venv not found at {python_bin}"}

    extra = []
    if date_str:
        extra = [date_str]

    try:
        output = _run_script(python_bin, "run_today.py", MLB_MODEL_DIR, timeout=300, extra_args=extra)
        picks = _parse_mlb_output(output)
        return {"ok": True, "picks": picks, "raw_lines": len(output.split("\n"))}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "MLB model timed out (5 min limit)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_scores24_scraper(sports: list[str]) -> dict[str, Any]:
    """Execute the Scores24 scraper for selected sports."""
    if not os.path.exists(SCORES24_VENV):
        return {"ok": False, "error": f"Scores24 venv not found at {SCORES24_VENV}"}

    sport_map = {
        "nba": "basketball",
        "nhl": "ice-hockey",
        "mlb": "baseball",
    }

    all_picks: list[dict[str, Any]] = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    scraper_path = os.path.join(BASE_DIR, "scores24_scraper.py")

    for sport_code in sports:
        sport_slug = sport_map.get(sport_code)
        if not sport_slug:
            continue

        try:
            result = subprocess.run(
                [SCORES24_VENV, scraper_path, "--sport", sport_slug, "--date", today_str],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=180,
            )
            output = result.stdout + result.stderr
            picks = _parse_scores24_output(output)
            # Override sport code from scraper to match our system
            for p in picks:
                if sport_code == "nba":
                    p["sport"] = "NBA"
                elif sport_code == "nhl":
                    p["sport"] = "NHL"
                elif sport_code == "mlb":
                    p["sport"] = "MLB"
            all_picks.extend(picks)
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    return {"ok": True, "picks": all_picks}


def _launch_job(target_fn, *args) -> str:
    """Launch a model run in a background thread and return a job_id."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "result": None}

    def _worker():
        result = target_fn(*args)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": result}

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return job_id


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        # Poll job status: GET /job-status?id=<job_id>
        if self.path.startswith("/job-status"):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            job_id = (qs.get("id") or [""])[0]
            with _jobs_lock:
                job = _jobs.get(job_id)
            if not job:
                self._send_json(404, {"ok": False, "error": "Job not found"})
            elif job["status"] == "running":
                self._send_json(200, {"ok": True, "status": "running"})
            else:
                self._send_json(200, {"ok": True, "status": "done", **job["result"]})
                # Clean up finished job
                with _jobs_lock:
                    _jobs.pop(job_id, None)
        else:
            self._send_json(404, {"ok": False, "error": "Route not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length"})
            return

        try:
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
            return

        async_mode = body.get("async", False)
        date_str = body.get("date")  # optional MM/DD/YYYY date for the model

        if self.path == "/grade":
            picks = body.get("picks", [])
            existing = body.get("existing", {})
            year = int(body.get("year") or datetime.now().year)

            if not isinstance(picks, list) or not isinstance(existing, dict):
                self._send_json(400, {"ok": False, "error": "Invalid payload shape"})
                return

            result = auto_grade(picks, existing, year)
            self._send_json(200, {"ok": True, **result})

        elif self.path == "/run-nba-model":
            if async_mode:
                job_id = _launch_job(run_nba_model, date_str)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_nba_model(date_str)
                self._send_json(200, result)

        elif self.path == "/run-mlb-model":
            if async_mode:
                job_id = _launch_job(run_mlb_model, date_str)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_mlb_model(date_str)
                self._send_json(200, result)

        elif self.path == "/run-scores24":
            sports = body.get("sports", ["nba", "nhl", "mlb"])
            if async_mode:
                job_id = _launch_job(run_scores24_scraper, sports)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_scores24_scraper(sports)
                self._send_json(200, result)

        else:
            self._send_json(404, {"ok": False, "error": "Route not found"})


def main() -> None:
    # Allow concurrent requests via threading
    import socketserver
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = ThreadedHTTPServer((HOST, PORT), Handler)
    print(f"Pickgrader running on http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down pickgrader...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

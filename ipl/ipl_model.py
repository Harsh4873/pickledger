from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipl.data.live_feed import run_live_feed_update
from ipl.data_loader import _default_db_path
from ipl.models.fantasy_selector import run_match_fantasy_model
from ipl.models.win_predictor import predict_winner


def _resolve_db_path(db_path: str | Path | None) -> Path:
    return Path(db_path) if db_path is not None else _default_db_path()


def _load_fallback_match(db_path: str | Path) -> dict[str, str]:
    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT team1, team2, venue
            FROM ipl_matches
            WHERE team1 IS NOT NULL
              AND team2 IS NOT NULL
              AND venue IS NOT NULL
            ORDER BY date(date) DESC, match_id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()

    if row is None:
        raise RuntimeError("Unable to resolve an IPL match from live feed or fallback database")

    return {
        "team1": str(row["team1"]).strip(),
        "team2": str(row["team2"]).strip(),
        "venue": str(row["venue"]).strip(),
    }


def run_ipl_model(
    team1: str | None = None,
    team2: str | None = None,
    venue: str | None = None,
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    db_file = _resolve_db_path(db_path)

    if team1 is None or team2 is None or venue is None:
        matches = run_live_feed_update(db_file)
        if matches:
            match = matches[0]
            team1 = match["team1"]
            team2 = match["team2"]
            venue = match["venue"]
            if toss_winner is None:
                toss_winner = team1
            if toss_decision is None:
                toss_decision = "bat"
            print(f"Auto-detected match: {team1} vs {team2} @ {venue}")
        else:
            fallback_match = _load_fallback_match(db_file)
            team1 = fallback_match["team1"]
            team2 = fallback_match["team2"]
            venue = fallback_match["venue"]
            if toss_winner is None:
                toss_winner = team1
            if toss_decision is None:
                toss_decision = "bat"
            print(f"Fallback match from DB: {team1} vs {team2} @ {venue}")

    if toss_winner is None:
        toss_winner = team1
    if toss_decision is None:
        toss_decision = "bat"

    result = predict_winner(team1, team2, venue, toss_winner, toss_decision, db_file)
    predicted_winner = result["predicted_winner"]
    team1_win_prob = result["team1_win_prob"]
    team2_win_prob = result["team2_win_prob"]
    confidence = result["confidence"]

    fantasy = run_match_fantasy_model(team1, team2, venue, toss_winner, toss_decision, db_file)
    players = fantasy["selected_players"]

    return {
        "match": f"{team1} vs {team2}",
        "venue": venue,
        "toss": f"{toss_winner} elected to {toss_decision}",
        "predicted_winner": predicted_winner,
        "team1": team1,
        "team2": team2,
        "team1_win_prob": round(team1_win_prob, 4),
        "team2_win_prob": round(team2_win_prob, 4),
        "confidence": confidence,
        "selected_players": [
            {
                "player_name": p["player_name"],
                "team": p["team"],
                "role": p["role"],
                "fantasy_probability_pct": round(p["fantasy_probability_pct"], 1),
                "decision": p["decision"],
                "is_captain": p["captain"],
                "is_vice_captain": p["vice_captain"],
            }
            for p in players
        ],
    }


def format_ipl_output(result: dict[str, Any]) -> str:
    lines = [
        "=== IPL MATCH PREDICTION ===",
        f"{result['team1']} vs {result['team2']} @ {result['venue']}",
        f"Toss: {result['toss']}",
        "",
        (
            f"WINNER: {result['predicted_winner']} "
            f"({result['team1_win_prob']:.1%} / {result['team2_win_prob']:.1%}) "
            f"[{result['confidence']}]"
        ),
        "",
        "FANTASY XI:",
    ]

    for player in result["selected_players"]:
        if player["is_captain"]:
            emoji = "👑"
        elif player["is_vice_captain"]:
            emoji = "⭐"
        elif player["decision"] == "BET":
            emoji = "🏏"
        else:
            emoji = "⬜"
        lines.append(
            f"{emoji} {player['player_name']} | {player['team']} | {player['role']} | "
            f"{player['fantasy_probability_pct']:.1f}% | {player['decision']}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db = _default_db_path()
    result = run_ipl_model(db_path=db)
    print(format_ipl_output(result))
    print("\nJSON output:")
    print(json.dumps(result, indent=2))

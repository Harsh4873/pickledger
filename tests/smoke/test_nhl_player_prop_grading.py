"""NHL player props settle from the player's box-score line, never the game total."""
from __future__ import annotations

import copy

import pytest

SKATER_KEYS = ["blockedShots", "hits", "takeaways", "plusMinus", "timeOnIce", "goals", "assists",
               "shotsTotal", "shotsMissed", "shootoutGoals"]
SKATER_LABELS = ["BS", "HT", "TK", "+/-", "TOI", "G", "A", "S", "SM", "SOG"]
GOALIE_KEYS = ["goalsAgainst", "shotsAgainst", "saves", "savePct", "timeOnIce"]
GOALIE_LABELS = ["GA", "SA", "SV", "SV%", "TOI"]


def _skater(name, aid, goals, assists, shots, blocked=0, shootout_goals=0):
    # ESPN's "SOG" column is shootout goals; shots on goal live under "S".
    return {"athlete": {"id": aid, "displayName": name},
            "stats": [str(blocked), "0", "0", "0", "18:00", str(goals), str(assists), str(shots), "1", str(shootout_goals)]}


def _goalie(name, aid, saves, toi="60:00"):
    return {"athlete": {"id": aid, "displayName": name}, "stats": ["1", str(saves + 1), str(saves), ".950", toi]}


SUMMARY = {
    "header": {"competitions": [{"status": {"type": {"completed": True, "name": "STATUS_FINAL"}}}]},
    "boxscore": {"players": [
        {"team": {"abbreviation": "FLA"}, "statistics": [
            {"name": "forwards", "keys": SKATER_KEYS, "labels": SKATER_LABELS, "athletes": [
                _skater("Carter Verhaeghe", "1", 1, 0, 3, shootout_goals=1),
                _skater("Sam Reinhart", "2", 0, 2, 4),
                _skater("Sam Bennett", "3", 0, 0, 2),
            ]},
            {"name": "defenses", "keys": SKATER_KEYS, "labels": SKATER_LABELS, "athletes": [
                _skater("Gustav Forsling", "4", 0, 1, 0, blocked=3),
            ]},
            {"name": "goalies", "keys": GOALIE_KEYS, "labels": GOALIE_LABELS, "athletes": [
                _goalie("Sergei Bobrovsky", "5", 25),
                _goalie("Spencer Knight", "6", 0, toi="0:00"),
            ]},
        ]},
    ]},
}


def _pick(player, stat, direction, line):
    return {
        "id": f"{player}-{stat}-{direction}-{line}",
        "sport": "NHL",
        "league": "NHL",
        "market": "player_props",
        "market_type": "player_props",
        "player": player,
        "player_name": player,
        "stat": f"{player} {stat} O/U",
        "stat_label": f"{player} {stat} O/U",
        "direction": direction,
        "line": line,
        "pick": f"{player} {player} {stat} O/U {direction} {line} (Florida Panthers @ Carolina Hurricanes)",
        "away_abbrev": "FLA",
        "home_abbrev": "CAR",
        "game_start_time": "2026-09-29T21:00Z",
        "date": "2026-09-29",
    }


@pytest.mark.parametrize(
    "player,stat,direction,line,expected",
    [
        # 0.5 lines, both sides, every supported stat.
        ("Carter Verhaeghe", "Points", "over", 0.5, "win"),
        ("Carter Verhaeghe", "Points", "under", 0.5, "loss"),
        ("Sam Bennett", "Points", "over", 0.5, "loss"),
        ("Sam Bennett", "Points", "under", 0.5, "win"),
        ("Sam Reinhart", "Assists", "over", 1.5, "win"),
        ("Sam Reinhart", "Assists", "under", 1.5, "loss"),
        ("Carter Verhaeghe", "Assists", "under", 0.5, "win"),
        ("Carter Verhaeghe", "Goals", "over", 0.5, "win"),
        ("Sam Reinhart", "Goals", "under", 0.5, "win"),
        ("Sam Reinhart", "Shots on Goal", "over", 3.5, "win"),
        ("Sam Bennett", "Shots on Goal", "over", 2.5, "loss"),
        ("Sam Bennett", "Shots on Goal", "under", 2.5, "win"),
        ("Sergei Bobrovsky", "Saves", "over", 24.5, "win"),
        ("Sergei Bobrovsky", "Saves", "under", 24.5, "loss"),
        ("Gustav Forsling", "Blocked Shots", "over", 2.5, "win"),
        # Whole-number lines push on an exact hit.
        ("Sam Reinhart", "Shots on Goal", "over", 4, "push"),
        ("Sam Reinhart", "Shots on Goal", "under", 4, "push"),
        ("Sam Reinhart", "Points", "under", 2, "push"),
        ("Sergei Bobrovsky", "Saves", "over", 25, "push"),
    ],
)
def test_nhl_prop_settles_from_player_line(player, stat, direction, line, expected):
    import pickgrader_server as g

    pick = _pick(player, stat, direction, line)
    parsed = g.parse_player_prop_pick(pick)
    assert parsed is not None and parsed["sport"] == "NHL"
    assert parsed["selection"] == direction.upper()
    assert g.grade_player_prop_pick(pick, {}, copy.deepcopy(SUMMARY)) == expected


def test_shots_on_goal_reads_espn_s_column_not_shootout_goals():
    import pickgrader_server as g

    value, status = g._extract_nhl_player_stat(SUMMARY, "Carter Verhaeghe", "nhl_shots_on_goal")
    assert (value, status) == (3.0, "played")


def test_text_only_pick_parses_repeated_player_name():
    import pickgrader_server as g

    parsed = g.parse_player_prop_pick({
        "sport": "NHL",
        "pick": "Sean Walker Sean Walker Points O/U under 0.5 (Florida Panthers @ Carolina Hurricanes)",
    })
    assert parsed == {"player_name": "Sean Walker", "stat_key": "nhl_points", "selection": "UNDER",
                      "line": 0.5, "opponent": "", "sport": "NHL"}


def test_nhl_prop_never_falls_back_to_game_total():
    """Regression: 'under 0.5' was read as a full-game total (every Under lost)."""
    import pickgrader_server as g

    game = {"competitors": [{"score": 1, "team": "CAR"}, {"score": 0, "team": "FLA"}]}
    pick = _pick("Sam Bennett", "Power Play Points", "under", 0.5)  # unsupported stat
    assert g.parse_player_prop_pick(pick) is None
    assert g.grade_pick(pick, game) == "pending"
    assert pick["grade_unsupported_reason"] == "nhl_player_prop_unparsed"


def test_backup_goalie_who_did_not_play_pushes_saves():
    import pickgrader_server as g

    assert g.grade_player_prop_pick(_pick("Spencer Knight", "Saves", "over", 20.5), {}, SUMMARY) == "push"


def test_scratched_skater_pushes_only_when_nhl_boxscore_confirms(monkeypatch):
    import pickgrader_server as g

    calls = []

    def dressed(pick, name):
        calls.append(name)
        return False

    monkeypatch.setattr(g, "nhl_api_player_dressed", dressed)
    pick = _pick("Aaron Ekblad", "Points", "under", 0.5)
    assert g.grade_player_prop_pick(pick, {}, SUMMARY) == "push"
    assert calls == ["Aaron Ekblad"]

    monkeypatch.setattr(g, "nhl_api_player_dressed", lambda pick, name: None)
    unknown = _pick("Aaron Ekblad", "Points", "under", 0.5)
    assert g.grade_player_prop_pick(unknown, {}, SUMMARY) == "pending"
    assert unknown["grade_anomaly"] == "player_not_in_boxscore"

    monkeypatch.setattr(g, "nhl_api_player_dressed", lambda pick, name: True)
    mismatch = _pick("Aaron Ekblad", "Points", "under", 0.5)
    assert g.grade_player_prop_pick(mismatch, {}, SUMMARY) == "pending"


def test_nhl_api_dressed_lookup(monkeypatch):
    import pickgrader_server as g

    pages = {
        "https://api-web.nhle.com/v1/score/2026-09-28": {"games": []},
        "https://api-web.nhle.com/v1/score/2026-09-29": {"games": [
            {"id": 2026020001, "awayTeam": {"abbrev": "FLA"}, "homeTeam": {"abbrev": "CAR"}}]},
        "https://api-web.nhle.com/v1/gamecenter/2026020001/boxscore": {
            "gameState": "OFF",
            "playerByGameStats": {
                "awayTeam": {"forwards": [{"name": {"default": "S. Bennett"}}], "defense": [], "goalies": []},
                "homeTeam": {"forwards": [], "defense": [], "goalies": []},
            },
        },
    }
    monkeypatch.setattr(g, "_NHL_API_BOX_CACHE", {})
    monkeypatch.setattr(g, "_fetch_json_url", lambda url: pages.get(url))
    pick = _pick("Sam Bennett", "Points", "over", 0.5)
    assert g.nhl_api_player_dressed(pick, "Sam Bennett") is True
    assert g.nhl_api_player_dressed(pick, "Aaron Ekblad") is False


def test_auto_grade_routes_nhl_prop_to_player_grader(monkeypatch):
    import pickgrader_server as g

    scoreboard = {"events": [{
        "id": "401891773",
        "competitions": [{
            "date": "2026-09-29T21:00Z",
            "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
            "competitors": [
                {"score": "1", "homeAway": "home", "team": {"displayName": "Carolina Hurricanes",
                 "shortDisplayName": "Hurricanes", "name": "Hurricanes", "abbreviation": "CAR"}},
                {"score": "0", "homeAway": "away", "team": {"displayName": "Florida Panthers",
                 "shortDisplayName": "Panthers", "name": "Panthers", "abbreviation": "FLA"}},
            ],
        }],
    }]}
    monkeypatch.setattr(g, "fetch_scoreboard", lambda *_a, **_k: scoreboard)
    monkeypatch.setattr(g, "fetch_event_summary", lambda *_a, **_k: copy.deepcopy(SUMMARY))
    picks = [_pick("Sam Bennett", "Points", "under", 0.5), _pick("Sam Bennett", "Points", "over", 0.5)]
    result = g.auto_grade(picks, {}, 2026)
    # The game had 1 total goal; the old fallback graded these loss/win.
    assert result["graded"] == {picks[0]["id"]: "win", picks[1]["id"]: "loss"}


def test_regrade_only_touches_nhl_prop_results_and_is_idempotent():
    from scripts import regrade_nhl_player_props as r

    nhl = {"model_key": "nhl", "market": "player_props", "slate_date": "2026-09-29", "game_id": "401891773",
           "result": "loss", "pregame_snapshot": _pick("Sam Bennett", "Points", "under", 0.5)}
    team = {"model_key": "nhl", "market": "totals", "slate_date": "2026-09-29", "game_id": "401891773",
            "result": "loss", "pick": "Under 0.5"}
    other = {"model_key": "mlb_new", "market": "player_props", "slate_date": "2026-09-29", "result": "loss",
             "sport": "MLB", "pick": "X Hits Under 0.5"}
    before_team, before_other = copy.deepcopy(team), copy.deepcopy(other)
    report = {"transitions": {}, "anomalies": {}}
    summaries: dict = {}
    fetch = lambda _eid: copy.deepcopy(SUMMARY)  # noqa: E731
    for row in (nhl, team, other):
        r._apply(row, summaries, fetch, "2026-09-29", report)
    assert nhl["result"] == "win"
    assert team == before_team and other == before_other
    snapshot = copy.deepcopy(nhl)
    assert r._apply(nhl, summaries, fetch, "2026-09-29", report) is False
    assert nhl == snapshot

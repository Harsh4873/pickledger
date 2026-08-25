from __future__ import annotations

import copy
import sys
from types import ModuleType


def test_nba_summer_has_an_official_grading_scoreboard_mapping():
    import pickgrader_server

    assert pickgrader_server.SPORT_TO_ESPNSLUG["NBA SUMMER"] == (
        "basketball",
        "nba-summer",
    )


def test_certified_snapshot_grader_never_grades_existing_pass_rows(monkeypatch, tmp_path):
    import scripts.auto_grade_picks as auto_grade_picks

    snapshot = {
        "date": "2026-07-09",
        "sport": "NBA SUMMER",
        "pick": "Utah Jazz ML (Oklahoma City Thunder @ Utah Jazz)",
        "matchup": "Oklahoma City Thunder @ Utah Jazz",
        "decision": "BET",
    }
    ledger = {
        "records": [
            {
                "id": "certified-pass",
                "result": "pending",
                "decision": "PASS",
                "certification": {"status": "certified"},
                "pregame_snapshot": {**snapshot, "decision": "PASS"},
            },
            {
                "id": "calibrated-pass",
                "result": "pending",
                "decision": "PASS",
                "raw_decision": "BET",
                "certification": {"status": "certified"},
                "pregame_snapshot": {**snapshot, "pick": "Raw BET downgraded to PASS"},
            },
            {
                "id": "certified-bet",
                "result": "pending",
                "decision": "BET",
                "raw_decision": "BET",
                "certification": {"status": "certified"},
                "pregame_snapshot": copy.deepcopy(snapshot),
            },
        ]
    }
    writes: list[dict] = []
    ledger_module = ModuleType("scripts.team_prop_pregame_ledger")

    def load_team_prop_pregame_ledger(*, repo_root):
        assert repo_root == tmp_path
        return ledger

    def write_team_prop_pregame_ledger(payload, *, repo_root):
        assert repo_root == tmp_path
        writes.append(copy.deepcopy(payload))
        return True

    ledger_module.load_team_prop_pregame_ledger = load_team_prop_pregame_ledger
    ledger_module.write_team_prop_pregame_ledger = write_team_prop_pregame_ledger
    monkeypatch.setitem(sys.modules, "scripts.team_prop_pregame_ledger", ledger_module)

    captured: list[dict] = []

    def auto_grade(candidates, _existing, _year):
        captured.extend(copy.deepcopy(candidates))
        return {
            "graded": {"certified-bet": "win"},
            "startTimes": {"certified-bet": "2026-07-09T23:00:00Z"},
        }

    monkeypatch.setattr(auto_grade_picks.pickgrader_server, "auto_grade", auto_grade)

    summary = auto_grade_picks.grade_certified_team_prop_snapshots(tmp_path)

    assert summary == {
        "available": True,
        "candidates": 1,
        "graded": 1,
        "start_times": 1,
        "changed": True,
    }
    assert captured == [{**snapshot, "id": "certified-bet", "result": "pending"}]
    assert ledger["records"][0]["result"] == "pending"
    assert ledger["records"][1]["result"] == "pending"
    assert ledger["records"][2]["result"] == "win"
    assert ledger["records"][2]["start_time"] == "2026-07-09T23:00:00Z"
    assert ledger["records"][2]["game_start_time"] == "2026-07-09T23:00:00Z"
    assert ledger["records"][2]["pregame_snapshot"] == snapshot
    assert len(writes) == 1

    legacy_payload = {
        "models": {"nba_summer": {"picks": [{**snapshot, "decision": "PASS", "result": "pending"}]}}
    }
    assert auto_grade_picks.grade_payload(legacy_payload) == 0
    assert legacy_payload["models"]["nba_summer"]["picks"][0]["result"] == "pending"


def test_certified_mlb_new_candidate_copies_slate_date_onto_date():
    from scripts.auto_grade_picks import _pending_certified_team_prop_candidate

    record = {
        "id": "mlb-new-pending",
        "model_key": "mlb_new",
        "result": "pending",
        "decision": "LEAN",
        "raw_decision": "LEAN",
        "slate_date": "2026-08-22",
        "sport": "MLB",
        "certification": {"status": "certified"},
        "pregame_snapshot": {
            "sport": "MLB",
            "pick": "Yankees ML (Blue Jays vs Yankees)",
            "decision": "LEAN",
            "away_team": "Toronto Blue Jays",
            "home_team": "New York Yankees",
            "game_start_time": "2026-08-22T17:35:00Z",
        },
    }

    candidate_row = _pending_certified_team_prop_candidate(record)

    assert candidate_row is not None
    record_id, candidate = candidate_row
    assert record_id == "mlb-new-pending"
    assert candidate["date"] == "2026-08-22"
    assert candidate["slate_date"] == "2026-08-22"
    assert candidate["sport"] == "MLB"
    assert candidate["pick"] == "Yankees ML (Blue Jays vs Yankees)"

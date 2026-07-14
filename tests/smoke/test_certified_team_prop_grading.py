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


def test_certified_snapshot_grader_includes_pass_without_changing_legacy_cache(monkeypatch, tmp_path):
    import scripts.auto_grade_picks as auto_grade_picks

    snapshot = {
        "date": "2026-07-09",
        "sport": "NBA SUMMER",
        "pick": "Utah Jazz ML (Oklahoma City Thunder @ Utah Jazz)",
        "matchup": "Oklahoma City Thunder @ Utah Jazz",
        "decision": "PASS",
    }
    ledger = {
        "records": [
            {
                "id": "certified-pass",
                "result": "pending",
                "certification": {"status": "certified"},
                "pregame_snapshot": copy.deepcopy(snapshot),
            },
            {
                "id": "uncertified-pass",
                "result": "pending",
                "certification": {"status": "draft"},
                "pregame_snapshot": {**snapshot, "pick": "Draft PASS"},
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
            "graded": {"certified-pass": "win"},
            "startTimes": {"certified-pass": "2026-07-09T23:00:00Z"},
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
    assert captured == [{**snapshot, "id": "certified-pass", "result": "pending"}]
    assert ledger["records"][0]["result"] == "win"
    assert ledger["records"][0]["start_time"] == "2026-07-09T23:00:00Z"
    assert ledger["records"][0]["game_start_time"] == "2026-07-09T23:00:00Z"
    assert ledger["records"][0]["pregame_snapshot"] == snapshot
    assert len(writes) == 1

    legacy_payload = {"models": {"nba_summer": {"picks": [{**snapshot, "result": "pending"}]}}}
    assert auto_grade_picks.grade_payload(legacy_payload) == 0
    assert legacy_payload["models"]["nba_summer"]["picks"][0]["result"] == "pending"

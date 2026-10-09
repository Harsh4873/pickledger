"""NBA player-prop consensus: trainer wiring stays fail-closed and gate-identical."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from player_props import consensus
from scripts import build_player_prop_outcome_history as outcome_history
from scripts import train_player_prop_consensus_ml as trainer


def _shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _market(**overrides) -> dict:
    row = {
        "sport": "NBA", "season": 2026, "season_type": "regular", "date": "2026-01-15",
        "start_time": "2026-01-16T00:30Z", "event_id": "e1", "athlete_id": "7", "stat_key": "points",
        "market_format": "total", "line": 20.5, "over_odds": -110, "under_odds": -110,
        "over_implied": 0.5238, "under_implied": 0.5238, "actual": 25.0, "over_outcome": 1,
    }
    row.update(overrides)
    return row


def test_nba_season_labels_cross_the_calendar_year():
    assert consensus.season_for_date("NBA", "2025-11-03") == 2026
    assert consensus.season_for_date("NBA", "2026-06-10") == 2026
    assert consensus.season_for_date("NBA", "2026-10-21") == 2027
    # Single-year sports keep their calendar-year season.
    assert consensus.season_for_date("WNBA", "2026-08-30") == 2026
    assert consensus.season_for_date("MLB", "2026-09-30") == 2026
    assert consensus.season_for_date(None, "2026-09-30") == 2026


def test_outcome_features_treat_an_nba_season_as_one_season():
    rows = [
        {"sport": "NBA", "season": 2026, "date": day, "event_id": day, "actual": value, "usage": 30.0}
        for day, value in (("2025-11-20", 20.0), ("2025-12-01", 22.0), ("2025-12-10", 24.0))
    ]
    features = consensus.outcome_features(rows, target_date="2025-12-20", sport="NBA")
    assert features is not None and features["season_count"] == 3.0
    # Without the sport, a December date maps to "season 2025" and abstains.
    assert consensus.outcome_features(rows, target_date="2025-12-20") is None
    # A new season resets season features (opening night abstains).
    assert consensus.outcome_features(rows, target_date="2026-10-22", sport="NBA") is None


def test_nba_artifacts_are_configured_but_absent_so_props_fail_closed(monkeypatch, tmp_path):
    assert consensus.MODEL_PATHS[("NBA", "season")].name == "nba_player_props_season.joblib"
    assert consensus.MODEL_PATHS[("NBA", "history")].name == "nba_player_props_history.joblib"
    paths = dict(consensus.MODEL_PATHS)
    paths[("NBA", "season")] = tmp_path / "missing-season.joblib"
    paths[("NBA", "history")] = tmp_path / "missing-history.joblib"
    monkeypatch.setattr(consensus, "MODEL_PATHS", paths)
    result = consensus.evaluate_consensus_pick({"sport": "NBA", "stat_key": "points", "line": 20.5})
    assert result["qualified"] is False
    if result["required"]:
        assert result["reason"] == "no NBA consensus model configured"


def test_nba_policies_use_the_same_bar_and_never_lower_sample_floors():
    assert trainer.TARGET_ACCURACY == 0.70
    nba = trainer.POLICIES["NBA"]
    assert set(nba) == consensus.TARGET_STATS["NBA"]
    strictest_validation = max(
        int(policy["minimum_validation_samples"])
        for sport, policies in trainer.POLICIES.items() if sport != "NBA" for policy in policies.values()
    )
    strictest_holdout = max(
        int(policy["minimum_holdout_samples"])
        for sport, policies in trainer.POLICIES.items() if sport != "NBA" for policy in policies.values()
    )
    for stat_key, policy in nba.items():
        assert policy["minimum_validation_samples"] >= strictest_validation
        assert policy["minimum_holdout_samples"] >= strictest_holdout
        assert ("NBA", stat_key) in trainer.SEARCHED_MARKETS
    # The search grid tunes filters only; it never touches floors or accuracy.
    assert not any("samples" in name or "accuracy" in name for name in trainer.CLASSIFIER_GRID)
    assert "NBA" in trainer.SOFT_SKIP_SPORTS


def test_trainer_reads_only_regular_or_postseason_pregame_totals(tmp_path):
    _shard(tmp_path / "2026-01-15.jsonl.gz", [
        _market(),
        _market(athlete_id="8", season_type="postseason"),
        _market(athlete_id="9", season_type="preseason"),
        _market(athlete_id="10", season_type=None),
        _market(athlete_id="11", market_format="milestone"),
        _market(athlete_id="12", stat_key="steals"),
    ])
    rows = trainer._read_nba_market_rows(tmp_path)
    assert sorted(row["athlete_id"] for row in rows) == ["7", "8"]
    assert trainer._read_nba_market_rows(tmp_path / "missing") == []
    assert trainer._nba_history_seasons(rows) == {2024, 2025, 2026}
    assert trainer._nba_history_seasons([]) == set()


def test_nba_market_features_reset_each_season():
    rows = [
        _market(event_id=f"a{i}", date=f"2026-0{i}-10", start_time=f"2026-0{i}-10T23:00Z", actual=float(10 * i))
        for i in range(1, 6)
    ] + [
        _market(event_id=f"b{i}", season=2027, date=f"2026-11-0{i}", start_time=f"2026-11-0{i}T23:00Z", actual=30.0)
        for i in range(1, 5)
    ]
    features, profiles = trainer._season_reset_training_features(rows)
    later = [row for row in features if row["season"] == 2027]
    # Only the 4th 2027 game has three prior same-season results.
    assert [row["event_id"] for row in later] == ["b4"]
    assert later[0]["history_count"] == 3 and later[0]["season_mean"] == 30.0
    # Serving profiles come from the latest season only.
    assert profiles["7|points"] == [30.0, 30.0, 30.0, 30.0]


def test_outcome_history_counts_nba_minutes_and_skips_dnp():
    payload = {
        "names": ["minutes", "totalRebounds", "assists", "points",
                  "threePointFieldGoalsMade-threePointFieldGoalsAttempted"],
        "events": {
            "g1": {"gameDate": "2026-01-16T00:30Z", "opponent": {"id": "2"}, "team": {"id": "1"}, "atVs": "vs"},
            "g2": {"gameDate": "2026-01-18T00:30Z", "opponent": {"id": "3"}, "team": {"id": "1"}, "atVs": "@"},
        },
        "seasonTypes": [{"displayName": "2025-26 Regular Season", "categories": [{"type": "event", "events": [
            {"eventId": "g1", "stats": ["32", "8", "6", "21", "3-7"]},
            {"eventId": "g2", "stats": ["0", "0", "0", "0", "0-0"]},
        ]}]}],
    }
    rows = outcome_history._event_rows("NBA", "7", 2026, payload)
    by_stat = {row["stat_key"]: row for row in rows}
    assert {row["event_id"] for row in rows} == {"g1"}
    assert by_stat["points"]["usage"] == 32.0 and by_stat["points"]["date"] == "2026-01-15"
    assert by_stat["rebounds_assists"]["actual"] == 14.0
    assert by_stat["three_pointers_made"]["actual"] == 3.0


def test_outcome_history_keeps_nba_in_its_own_corpus(monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--sports", "NBA,WNBA"])
    try:
        outcome_history.main()
    except SystemExit as exc:
        assert "separate corpus" in str(exc)
    else:  # pragma: no cover - must refuse mixing corpora
        raise AssertionError("mixing NBA with other sports must be refused")


def test_nba_activation_is_evaluated_on_regular_season_rows_only():
    assert trainer.NBA_EVALUATION_SEASON_TYPES == {"regular"}
    assert "preseason" not in trainer.NBA_TRAINING_SEASON_TYPES
    source = Path(trainer.__file__).read_text()
    assert 'outcome_market["season_type"].isin(NBA_EVALUATION_SEASON_TYPES)' in source
    assert '_windows(sport, nba_evaluation_rows if sport == "NBA" else market_rows)' in source


def test_daily_refresh_keeps_the_nba_outcome_corpus_current():
    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/player-props-refresh.yml").read_text()
    step = workflow.split("Refresh NBA player outcome history", 1)[1].split("- name:", 1)[0]
    assert "continue-on-error: true" in step
    assert "--sports NBA" in step and "--refresh-current-nba-season" in step
    commit_step = workflow.split("Commit player-props cache if changed", 1)[1]
    assert 'cp "$GENERATED_NBA_OUTCOMES" data/player_props_training/nba_outcome_history.jsonl.gz' in commit_step

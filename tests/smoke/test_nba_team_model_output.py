"""NBA team-model probability side, moneyline labels, and persisted log."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NBA_DIR = ROOT / "NBAPredictionModel"
if str(NBA_DIR) not in sys.path:
    sys.path.insert(0, str(NBA_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _new_format_output(
    *,
    away: str,
    home: str,
    winner: str,
    confidence: float,
    margin: float,
) -> str:
    return "\n".join(
        [
            f"GAME: {away} @ {home} (7:30 pm ET)",
            f"- **Pick:** {winner}",
            f"- **Projected Margin:** {winner} by {margin:.2f} points",
            f"- **Model Confidence:** {confidence:.1f}%",
        ]
    )


def _parse_nba_new(output: str, monkeypatch, *, spread=None, ml=None):
    import pickgrader_server as ps

    spread = (None, None, None) if spread is None else spread
    ml = (None, None) if ml is None else ml
    monkeypatch.setattr(ps, "_sl_get_spread", lambda h, a, league: spread)
    monkeypatch.setattr(ps, "_sl_get_ml", lambda h, a, league: ml)
    monkeypatch.setattr(ps, "_sl_get_total", lambda h, a, league: (None, None))
    monkeypatch.setattr(ps, "_nba_fatigue_multiplier", lambda *args: None)
    return [pick for pick in ps._parse_nba_output(output, source_label="NBA New") if pick.get("market_type") != "totals"]


def test_nba_new_away_confidence_is_not_flipped_on_unpriced_row(monkeypatch):
    output = _new_format_output(
        away="Celtics", home="Knicks", winner="Celtics", confidence=75.0, margin=19.9,
    )
    picks = _parse_nba_new(output, monkeypatch)
    assert len(picks) == 1
    pick = picks[0]
    assert pick["team"] == "Celtics"
    assert pick["probability"] == pytest.approx(0.75)
    assert pick["prob"] == pytest.approx(0.75)
    assert pick["pick"] == "Celtics ML (Celtics @ Knicks)"
    assert " -19.9" not in pick["pick"]
    assert pick["pricing_type"] == "unpriced"


def test_nba_new_home_confidence_stays_on_the_picked_side(monkeypatch):
    output = _new_format_output(
        away="Wizards", home="Hornets", winner="Hornets", confidence=75.0, margin=8.4,
    )
    picks = _parse_nba_new(output, monkeypatch)
    pick = picks[0]
    assert pick["team"] == "Hornets"
    assert pick["probability"] == pytest.approx(0.75)
    assert pick["pick"] == "Hornets ML (Wizards @ Hornets)"


def test_nba_new_ml_market_uses_picked_side_probability(monkeypatch):
    output = _new_format_output(
        away="Timberwolves", home="Cavaliers", winner="Timberwolves", confidence=75.0, margin=13.1,
    )
    picks = _parse_nba_new(output, monkeypatch, ml=(-180, 150))
    pick = picks[0]
    assert pick["team"] == "Timberwolves"
    assert pick["pick"] == "Timberwolves ML (Timberwolves @ Cavaliers)"
    assert pick["probability"] == pytest.approx(0.75)
    assert pick["odds"] == 150
    assert pick["edge"] is not None
    assert pick["edge"] > 0


def test_nba_new_spread_market_still_labels_the_vegas_line(monkeypatch):
    output = _new_format_output(
        away="Grizzlies", home="Pistons", winner="Pistons", confidence=66.0, margin=5.0,
    )
    picks = _parse_nba_new(output, monkeypatch, spread=(-4.5, 4.5, -110))
    pick = picks[0]
    assert pick["pick"] == "Pistons -4.5 (Grizzlies @ Pistons)"
    assert pick["odds"] == -110
    assert pick["market_line"] == -4.5
    assert pick["probability"] != pytest.approx(0.66)


def test_prediction_log_upserts_and_survives_on_committed_data_path(tmp_path):
    import prediction_logging as pl
    from calibration import fit_platt_scaler_from_log
    from data_models import GameContext, Team, TeamStats, Venue

    log_path = tmp_path / "data" / "nba" / "predictions.csv"
    stats = TeamStats(
        net_rating=1.0, off_rating_10=110.0, def_rating_10=109.0, ts_pct=0.55,
        reb_pct=0.5, pace=100.0, last_10_win_pct=0.5, is_b2b_second_leg=False,
        is_3_in_4_nights=False, season_win_pct=0.5, rest_days=2.0, back_to_back_flag=False,
    )
    away = Team(1, "Celtics", False, stats, [])
    home = Team(2, "Knicks", True, stats, [])
    ctx = GameContext("2026-10-22", Venue("Garden"), home, away, 0.5, game_id="g1")
    pl.append_prediction_log(ctx, -8.5, 0.22, 0.18, 0.25, log_path=log_path, log_source="live")
    pl.append_prediction_log(ctx, -8.4, 0.21, 0.17, 0.25, log_path=log_path, log_source="live")

    _, rows = pl.load_prediction_rows(log_path)
    assert len(rows) == 1
    assert rows[0]["raw_probability"] == "0.210000"
    assert rows[0]["log_source"] == "live"
    assert log_path == tmp_path / "data" / "nba" / "predictions.csv"

    pl.apply_prediction_outcome(rows[0], actual_home_win=1, home_score=110, away_score=101, outcome_source="espn_scoreboard")
    pl.write_prediction_rows(rows, log_path=log_path)
    changed = pl.apply_prediction_outcome(rows[0], actual_home_win=0, outcome_source="invented")
    assert changed is False
    assert rows[0]["actual_home_win"] == "1"

    scaler, diag = fit_platt_scaler_from_log(log_path=str(log_path), min_samples=50)
    assert diag.sample_count == 1
    assert scaler.fitted is False
    assert scaler.confidence_cap == 0.75


def test_sync_grades_from_espn_and_graded_picks_without_inventing(tmp_path, monkeypatch):
    from scripts.sync_nba_prediction_log import (
        outcome_from_graded_pick,
        published_home_win_probability,
        sync_nba_prediction_log,
    )

    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    payload = {
        "date": "2026-10-22",
        "generatedAt": "2026-10-22T16:00:00Z",
        "models": {
            "nba": {
                "picks": [
                    {
                        "sport": "NBA",
                        "team": "Celtics",
                        "away_team": "Celtics",
                        "home_team": "Knicks",
                        "probability": 0.25,
                        "predicted_spread": 19.9,
                        "result": "win",
                    },
                    {
                        "sport": "NBA",
                        "team": "Hornets",
                        "away_team": "Wizards",
                        "home_team": "Hornets",
                        "probability": 0.75,
                        "predicted_spread": 8.4,
                        "result": "pending",
                    },
                    {
                        "sport": "NBA",
                        "market_type": "totals",
                        "team": "Over",
                        "away_team": "Heat",
                        "home_team": "Magic",
                        "probability": 0.55,
                        "result": "win",
                    },
                ]
            }
        },
    }
    (cache_dir / "2026-10-22.json").write_text(json.dumps(payload), encoding="utf-8")

    def fake_scoreboard(sport, league, yyyymmdd):
        assert (sport, league, yyyymmdd) == ("basketball", "nba", "20261022")
        return {
            "events": [
                {
                    "id": "1",
                    "competitions": [
                        {
                            "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
                            "competitors": [
                                {"homeAway": "away", "score": "118", "team": {"name": "Celtics"}},
                                {"homeAway": "home", "score": "110", "team": {"name": "Knicks"}},
                            ],
                        }
                    ],
                },
                {
                    "id": "2",
                    "competitions": [
                        {
                            "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
                            "competitors": [
                                {"homeAway": "away", "score": "99", "team": {"name": "Wizards"}},
                                {"homeAway": "home", "score": "112", "team": {"name": "Hornets"}},
                            ],
                        }
                    ],
                },
            ]
        }

    log_path = tmp_path / "data" / "nba" / "predictions.csv"
    summary = sync_nba_prediction_log(
        repo_root=tmp_path,
        log_path=log_path,
        fetch_scoreboard=fake_scoreboard,
    )
    assert summary["total"] == 2
    assert summary["graded"] == 2
    assert summary["platt_eligible"] == 0
    with log_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_home = {row["home_team"]: row for row in rows}
    assert by_home["Knicks"]["actual_home_win"] == "0"
    assert by_home["Knicks"]["away_score"] == "118"
    assert by_home["Knicks"]["raw_probability"] == ""
    assert by_home["Knicks"]["calibrated_probability"] == "0.250000"
    assert by_home["Hornets"]["actual_home_win"] == "1"
    assert by_home["Hornets"]["home_score"] == "112"

    away_pick = payload["models"]["nba"]["picks"][0]
    home_pick = payload["models"]["nba"]["picks"][1]
    assert published_home_win_probability(away_pick) == pytest.approx(0.25)
    assert published_home_win_probability({**away_pick, "probability": 0.75}) == pytest.approx(0.25)
    assert published_home_win_probability(home_pick) == pytest.approx(0.75)
    assert outcome_from_graded_pick(away_pick) == 0
    assert outcome_from_graded_pick(home_pick) is None


def test_sync_does_not_grade_unfinished_or_tied_games(tmp_path):
    from prediction_logging import write_prediction_rows
    from scripts.sync_nba_prediction_log import grade_rows_from_espn

    rows = [
        {
            "game_date": "2026-10-22",
            "away_team": "Heat",
            "home_team": "Magic",
            "raw_probability": "0.60",
            "actual_home_win": "",
        }
    ]
    write_prediction_rows(rows, log_path=tmp_path / "predictions.csv")

    def unfinished_scoreboard(_sport, _league, _date):
        return {
            "events": [
                {
                    "id": "live",
                    "competitions": [
                        {
                            "status": {"type": {"completed": False, "name": "STATUS_IN_PROGRESS"}},
                            "competitors": [
                                {"homeAway": "away", "score": "20", "team": {"name": "Heat"}},
                                {"homeAway": "home", "score": "22", "team": {"name": "Magic"}},
                            ],
                        }
                    ],
                }
            ]
        }

    updated = grade_rows_from_espn(rows, fetch_scoreboard=unfinished_scoreboard)
    assert updated == 0
    assert rows[0]["actual_home_win"] == ""

    def tied_scoreboard(_sport, _league, _date):
        return {
            "events": [
                {
                    "id": "tie",
                    "competitions": [
                        {
                            "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
                            "competitors": [
                                {"homeAway": "away", "score": "100", "team": {"name": "Heat"}},
                                {"homeAway": "home", "score": "100", "team": {"name": "Magic"}},
                            ],
                        }
                    ],
                }
            ]
        }

    updated = grade_rows_from_espn(rows, fetch_scoreboard=tied_scoreboard)
    assert updated == 0
    assert rows[0]["actual_home_win"] == ""


def test_refresh_workflow_commits_the_nba_prediction_log():
    workflow = (ROOT / ".github/workflows/model-cache-refresh.yml").read_text(encoding="utf-8")
    refresh = (ROOT / "scripts/refresh_model_cache.py").read_text(encoding="utf-8")
    auto_grade = (ROOT / "scripts/auto_grade_picks.py").read_text(encoding="utf-8")
    commit_step = workflow.split("Commit cache JSON if changed", 1)[1]
    assert "data/nba" in commit_step
    assert "scripts/sync_nba_prediction_log.py" in commit_step
    assert "GENERATED_NBA_LOG" in commit_step
    assert "sync_nba_prediction_log" in refresh
    assert "sync_nba_prediction_log" in auto_grade
    import prediction_logging as pl
    from calibration import _default_prediction_log_path

    assert pl.DEFAULT_PREDICTION_LOG_PATH == ROOT / "data" / "nba" / "predictions.csv"
    assert _default_prediction_log_path() == ROOT / "data" / "nba" / "predictions.csv"


def test_cache_backfill_skips_preseason_slates(tmp_path):
    from scripts.sync_nba_prediction_log import backfill_rows_from_model_cache

    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "2026-10-09.json").write_text(json.dumps({
        "date": "2026-10-09",
        "models": {"nba": {"picks": [{
            "sport": "NBA", "team": "Lakers", "away_team": "Lakers",
            "home_team": "Suns", "probability": 0.75, "predicted_spread": 6.0,
        }]}},
    }), encoding="utf-8")
    (cache_dir / "2026-10-22.json").write_text(json.dumps({
        "date": "2026-10-22",
        "models": {"nba": {"picks": [{
            "sport": "NBA", "team": "Celtics", "away_team": "Celtics",
            "home_team": "Knicks", "probability": 0.25, "predicted_spread": 19.9,
        }]}},
    }), encoding="utf-8")
    rows = backfill_rows_from_model_cache(tmp_path)
    assert [row["game_date"] for row in rows] == ["2026-10-22"]


def test_live_log_source_is_not_replaced_by_cache_backfill(tmp_path):
    import prediction_logging as pl
    from scripts.sync_nba_prediction_log import merge_prediction_rows

    live = {
        "game_date": "2026-10-26",
        "away_team": "Wizards",
        "home_team": "Hornets",
        "raw_probability": "0.184000",
        "calibrated_probability": "0.250000",
        "log_source": "live",
        "timestamp": "2026-10-26T12:00:00Z",
        "actual_home_win": "",
    }
    cache = {
        "game_date": "2026-10-26",
        "away_team": "Wizards",
        "home_team": "Hornets",
        "raw_probability": "0.250000",
        "calibrated_probability": "0.250000",
        "log_source": "model_cache_backfill",
        "timestamp": "2026-10-26T22:00:00Z",
        "actual_home_win": "0",
        "outcome_source": "graded_pick",
    }
    merged = merge_prediction_rows([live], [cache])
    assert len(merged) == 1
    assert merged[0]["raw_probability"] == "0.184000"
    assert merged[0]["log_source"] == "live"
    assert merged[0]["actual_home_win"] == "0"
    assert merged[0]["outcome_source"] == "graded_pick"
    assert pl.count_graded_predictions(merged) == {"total": 1, "graded": 1, "platt_eligible": 1}

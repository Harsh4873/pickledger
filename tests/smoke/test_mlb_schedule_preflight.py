from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SLATES = json.loads((ROOT / "tests/fixtures/mlb_schedule_20261008.json").read_text())


@pytest.fixture
def mlb_modules(monkeypatch):
    model_dir = ROOT / "MLBPredictionModel"
    # Other model packages share flat module names (calibration, live_data, ...).
    # Drop cached copies so this import resolves against the MLB directory.
    for name in [path.stem for path in model_dir.glob("*.py")] + [
        "MLBPredictionModel.live_data",
        "MLBPredictionModel.run_today",
    ]:
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(model_dir))
    from MLBPredictionModel import live_data, run_today
    monkeypatch.setattr(run_today, "build_live_dataframe", live_data.build_live_dataframe)
    return live_data, run_today


@pytest.mark.parametrize("games", [[], SLATES["postseason"]])
def test_empty_eligible_mlb_slate_skips_all_season_history(monkeypatch, mlb_modules, games):
    live, _ = mlb_modules
    calls = []

    def schedule(**kwargs):
        calls.append(kwargs)
        return games

    def forbidden_history():
        pytest.fail("an ineligible slate must not load season stats or game history")

    monkeypatch.setattr(live.statsapi, "schedule", schedule)
    monkeypatch.setattr(live, "StatsAPIClient", forbidden_history)
    assert live.build_live_dataframe(date(2026, 10, 8), market_odds_map={}).empty
    assert calls == [{"start_date": "2026-10-08", "end_date": "2026-10-08", "sportId": 1}]


def test_regular_season_slate_still_enters_feature_loading(monkeypatch, mlb_modules):
    live, _ = mlb_modules
    calls = []

    def schedule(**kwargs):
        calls.append("schedule")
        return SLATES["regular"]

    def history():
        calls.append("history")
        raise RuntimeError("feature loading reached")

    monkeypatch.setattr(live.statsapi, "schedule", schedule)
    monkeypatch.setattr(live, "StatsAPIClient", history)
    with pytest.raises(RuntimeError, match="feature loading reached"):
        live.build_live_dataframe(date(2026, 9, 27), market_odds_map={})
    assert calls == ["schedule", "history"]


def test_failed_schedule_cannot_become_an_empty_success(monkeypatch, mlb_modules, capsys):
    live, runner = mlb_modules

    def failed_schedule(**kwargs):
        raise TimeoutError("schedule unavailable")

    monkeypatch.setattr(live.statsapi, "schedule", failed_schedule)
    monkeypatch.setattr(runner, "fetch_mlb_market_odds_for_date", lambda _date: {})
    assert runner.main(["run_today.py", "--date", "2026-10-08", "--variant", "new", "--no-log"]) == 1
    output = capsys.readouterr()
    assert "MLB live inference failed: schedule unavailable" in output.err
    assert "No eligible" not in output.out


def test_postseason_cli_explains_model_coverage(monkeypatch, mlb_modules, capsys):
    live, runner = mlb_modules
    monkeypatch.setattr(live.statsapi, "schedule", lambda **_kwargs: SLATES["postseason"])
    monkeypatch.setattr(runner, "fetch_mlb_market_odds_for_date", lambda _date: {})
    monkeypatch.setattr(live, "StatsAPIClient", lambda: pytest.fail("unexpected history fetch"))
    assert runner.main(["run_today.py", "--date", "2026-10-08", "--variant", "new", "--no-log"]) == 0
    output = capsys.readouterr().out
    assert "No eligible regular-season MLB games found for 2026-10-08." in output
    assert "No MLB games found" not in output


@pytest.mark.parametrize("variant", ["old", "new"])
@pytest.mark.parametrize("output,expected", [
    ("No eligible regular-season MLB games found for 2026-10-08.\n", "No eligible regular-season MLB games"),
    ("No MLB games found for 2026-10-08.\n", "No MLB games found"),
])
def test_mlb_server_keeps_coverage_note(monkeypatch, variant, output, expected):
    import pickgrader_server as server
    monkeypatch.setattr(server, "_run_script", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(server, "_mlb_new_artifact_status", lambda: {"ready": True, "stack": "v2"})
    monkeypatch.setattr(server, "_save_admin_picks_doc", lambda *_args: None)
    monkeypatch.setattr(server, "_stamp_mlb_game_start_times", lambda *_args: None)
    result = server.run_mlb_model("2026-10-08", variant)
    assert result["ok"] is True
    assert result["picks"] == []
    assert expected in result["note"]

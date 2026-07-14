from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
MODEL_KEYS = {
    "mlb_new",
    "mlb_inning",
    "mlb_first_five",
    "wnba",
    "nba",
    "nba_playoffs",
    "nba_summer",
    "fifa_world_cup",
}
PLAYER_PROP_KEYS = {
    "nba_player_props",
    "mlb_player_props",
    "wnba_player_props",
}
SCORES24_KEYS = {"scores24_fifa_world_cup", "scores24_mlb", "scores24_wnba"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _upcheck_repo(tmp_path: Path, date: str) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts" / "site_upcheck.py", scripts / "site_upcheck.py")

    model_payload = {
        "date": date,
        "models": {key: {"ok": True, "picks": []} for key in MODEL_KEYS},
        "external_feeds": {
            key: {
                "ok": True,
                "date": date,
                "picks": [],
                "meta": {"expectedMatchups": 0, "matchedPicks": 0, "missingMatchups": []},
            }
            for key in SCORES24_KEYS
        },
    }
    props_payload = {"date": date, "models": {key: {"ok": True, "picks": []} for key in PLAYER_PROP_KEYS}}
    parlay_payload = {
        "date": date,
        "engineVersion": "parlay_cards_v5_market_excess",
        "summary": {"displayedCards": 0, "threeLegCards": 0},
        "cards": [],
    }
    profit_payload = {
        "schemaVersion": "2",
        "date": date,
        "engineVersion": "profit_desk_v2_live",
        "phase": "live",
        "policy": {"mode": "live", "status": "LIVE", "liveStaking": True},
        "summary": {
            "candidateCount": 0,
            "candidatesEvaluated": 0,
            "shadowQualified": 0,
            "researchQualified": 0,
            "edgeQualified": 0,
            "valueQualified": 0,
            "liveQualified": 0,
        },
        "candidates": [],
        "portfolio": {"team": [], "player": [], "all": [], "live": []},
    }
    for cache_name, payload in (("model_cache", model_payload), ("player_props_cache", props_payload)):
        cache_dir = tmp_path / "data" / cache_name
        _write_json(cache_dir / "latest.json", payload)
        _write_json(cache_dir / f"{date}.json", payload)
        _write_json(cache_dir / "index.json", {"files": [f"{date}.json"]})
    parlay_dir = tmp_path / "data" / "parlay_cards"
    _write_json(parlay_dir / "latest.json", parlay_payload)
    _write_json(parlay_dir / f"{date}.json", parlay_payload)
    _write_json(parlay_dir / "index.json", {"files": [f"{date}.json"]})
    profit_dir = tmp_path / "data" / "profit_desk"
    _write_json(profit_dir / "latest.json", profit_payload)
    _write_json(profit_dir / f"{date}.json", profit_payload)
    _write_json(
        profit_dir / "index.json",
        {
            "engineVersion": "profit_desk_v2_live",
            "files": [f"{date}.json"],
        },
    )
    return scripts / "site_upcheck.py"


def test_data_only_readiness_passes_without_build_or_cannon(tmp_path: Path):
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, today)

    result = subprocess.run(
        [sys.executable, str(script), "--data-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "daily data is ready" in result.stdout
    assert not (tmp_path / "dist").exists()
    assert "Cannon" not in result.stdout


def test_data_only_readiness_defers_stale_daily_data(tmp_path: Path):
    yesterday = (datetime.now(ZoneInfo("America/Chicago")) - timedelta(days=1)).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, yesterday)

    result = subprocess.run(
        [sys.executable, str(script), "--data-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "[readiness] waiting:" in result.stdout
    assert "expected" in result.stdout


def test_data_only_readiness_rejects_incomplete_scores24_bucket(tmp_path: Path):
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, today)
    cache_path = tmp_path / "data" / "model_cache" / "latest.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["external_feeds"]["scores24_fifa_world_cup"] = {
        "ok": False,
        "date": today,
        "picks": [{"matchup": "Qatar @ Canada"}],
        "error": "blocked before official slate completed",
        "meta": {
            "expectedMatchups": 2,
            "matchedPicks": 1,
            "missingMatchups": ["South Africa @ Czechia"],
        },
    }
    _write_json(cache_path, payload)

    result = subprocess.run(
        [sys.executable, str(script), "--data-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "scores24_fifa_world_cup failed" in result.stdout


def test_data_only_readiness_allows_stale_but_valid_scores24_feed(tmp_path: Path):
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    yesterday = (datetime.now(ZoneInfo("America/Chicago")) - timedelta(days=1)).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, today)
    # Scores24 refreshes only from a residential IP, so its published date can lag a day
    # behind the rest of today's data. A stale-but-valid feed (still ok and slate-complete)
    # must warn without freezing the deploy — the core model/props/parlay data is today's.
    cache_path = tmp_path / "data" / "model_cache" / "latest.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    for key in SCORES24_KEYS:
        payload["external_feeds"][key]["date"] = yesterday
    _write_json(cache_path, payload)
    _write_json(tmp_path / "data" / "model_cache" / f"{today}.json", payload)

    result = subprocess.run(
        [sys.executable, str(script), "--data-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "daily data is ready" in result.stdout
    assert "[readiness] warning:" in result.stdout
    assert f"scores24_mlb is {yesterday}" in result.stdout


def test_data_only_readiness_allows_weak_parlay_slate_without_team_cards(tmp_path: Path):
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, today)
    model_payload_path = tmp_path / "data" / "model_cache" / "latest.json"
    model_payload = json.loads(model_payload_path.read_text(encoding="utf-8"))
    model_payload["models"]["mlb_new"]["picks"] = [
        {"date": today, "sport": "MLB", "pick": "Visible bet A", "decision": "BET", "grade_supported": True},
        {"date": today, "sport": "MLB", "pick": "Visible bet B", "decision": "BET", "grade_supported": True},
        {"date": today, "sport": "MLB", "pick": "Visible lean C", "decision": "LEAN", "grade_supported": True},
    ]
    _write_json(model_payload_path, model_payload)
    _write_json(tmp_path / "data" / "model_cache" / f"{today}.json", model_payload)

    parlay_payload = {
        "date": today,
        "engineVersion": "parlay_cards_v5_market_excess",
        "summary": {
            "eligibleLegs": 3,
            "generatedThreeLegCandidates": 0,
            "displayedCards": 0,
            "threeLegCards": 0,
            "modes": {
                "team": {"displayedCards": 0},
                "player": {"displayedCards": 0},
            },
        },
        "cards": [],
    }
    _write_json(tmp_path / "data" / "parlay_cards" / "latest.json", parlay_payload)
    _write_json(tmp_path / "data" / "parlay_cards" / f"{today}.json", parlay_payload)

    result = subprocess.run(
        [sys.executable, str(script), "--data-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "daily data is ready" in result.stdout


def test_data_only_readiness_rejects_stake_without_live_qualification(tmp_path: Path):
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, today)
    profit_path = tmp_path / "data" / "profit_desk" / "latest.json"
    payload = json.loads(profit_path.read_text(encoding="utf-8"))
    payload["summary"]["candidateCount"] = 1
    payload["summary"]["candidatesEvaluated"] = 1
    payload["candidates"] = [{
        "id": "unsafe",
        "tier": "value",
        "stakeUnits": 0.5,
        "liveQualified": False,
        "blockers": [],
    }]
    _write_json(profit_path, payload)

    result = subprocess.run(
        [sys.executable, str(script), "--data-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "stake without live qualification" in result.stdout


def test_data_only_readiness_requires_profit_desk_manifest(tmp_path: Path):
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, today)
    (tmp_path / "data" / "profit_desk" / "index.json").unlink()

    result = subprocess.run(
        [sys.executable, str(script), "--data-only"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "data/profit_desk/index.json is missing or invalid" in result.stdout


def test_upcheck_reports_raw_and_visible_pick_counts(tmp_path: Path):
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    script = _upcheck_repo(tmp_path, today)
    (tmp_path / "index.html").write_text(
        '<link rel="stylesheet" href="./src/styles/pickledger.css">'
        '<script type="module" src="./src/main.ts"></script>',
        encoding="utf-8",
    )
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.html").write_text(
        '<link rel="stylesheet" href="/assets/index.css">'
        '<script type="module" src="/assets/index.js"></script>',
        encoding="utf-8",
    )

    model_payload_path = tmp_path / "data" / "model_cache" / "latest.json"
    model_payload = json.loads(model_payload_path.read_text(encoding="utf-8"))
    model_payload["models"]["mlb_new"]["picks"] = [
        {"date": today, "sport": "MLB", "pick": "Raw pass", "decision": "PASS"},
        {"date": today, "sport": "MLB", "pick": "Visible lean", "decision": "LEAN"},
    ]
    _write_json(model_payload_path, model_payload)
    _write_json(tmp_path / "data" / "model_cache" / f"{today}.json", model_payload)

    props_payload_path = tmp_path / "data" / "player_props_cache" / "latest.json"
    props_payload = json.loads(props_payload_path.read_text(encoding="utf-8"))
    props_payload["models"]["mlb_player_props"]["picks"] = [
        {"date": today, "scope": "player", "sport": "MLB", "pick": "Visible pass", "decision": "PASS"},
        {"date": today, "scope": "team", "sport": "MLB", "pick": "Hidden wrong scope", "decision": "BET"},
    ]
    _write_json(props_payload_path, props_payload)
    _write_json(tmp_path / "data" / "player_props_cache" / f"{today}.json", props_payload)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "teams_raw=" in result.stdout
    assert "teams_visible=" in result.stdout
    assert "player_props_raw=" in result.stdout
    assert "player_props_visible=" in result.stdout
    assert "'mlb_new': 2" in result.stdout
    assert "'mlb_new': 1" in result.stdout
    assert "'mlb_player_props': 2" in result.stdout
    assert "'mlb_player_props': 1" in result.stdout

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import merge_player_props_cache_payload as publisher
from scripts.merge_player_props_cache_payload import merge_payload
from scripts.site_upcheck import _published_player_prop_keys


def test_cache_write_failure_keeps_existing_json_and_removes_partial_temp(monkeypatch, tmp_path):
    path = tmp_path / "latest.json"
    path.write_text('{"date": "2026-09-30"}\n', encoding="utf-8")

    def partial_dump(_payload, handle, **_kwargs):
        handle.write('{"date":')
        raise OSError("disk full")

    monkeypatch.setattr(publisher.json, "dump", partial_dump)
    with pytest.raises(OSError, match="disk full"):
        publisher._write_json(path, {"date": "2026-10-01"})

    assert json.loads(path.read_text(encoding="utf-8"))["date"] == "2026-09-30"
    assert list(tmp_path.glob("*.tmp")) == []


def test_historical_merge_updates_dated_cache_without_downgrading_latest(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    latest = {"date": "2026-09-30", "models": {}}
    (cache_dir / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"date": "2026-09-29", "models": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge", str(generated), "--cache-dir", str(cache_dir),
                                  "--snapshot-dir", str(tmp_path / "snapshots")])

    assert publisher.main() == 0
    assert json.loads((cache_dir / "latest.json").read_text(encoding="utf-8")) == latest
    assert json.loads((cache_dir / "2026-09-29.json").read_text(encoding="utf-8"))["date"] == "2026-09-29"
    assert json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))["files"] == ["2026-09-29.json"]


def _pick(
    pick_id: str,
    market_id: str,
    pick_text: str,
    *,
    game_id: str | None = None,
    player_id: str | None = None,
    selection: str = "Under",
    line: float = 0.5,
    decision: str = "BET",
    ml_expected_value: float = 0.1,
    consensus_qualified: bool = True,
    mode: str = "four_model_consensus_gate",
) -> dict:
    return {
        "id": pick_id,
        "scope": "player",
        "source": "MLBPlayerProps",
        "model_key": "mlb_player_props",
        "sport": "MLB",
        "date": "2026-06-20",
        "game_id": game_id or f"game-{market_id}",
        "player_id": player_id or f"player-{market_id}",
        "stat_key": "hits",
        "selection": selection,
        "line": line,
        "pick": pick_text,
        "matchup": "A @ B",
        "market_priced": True,
        "probability_source": "player_props_ml_v1",
        "decision": decision,
        "ml_model_version": "player_props_consensus_v2.0.0",
        "ml_probability_mode": mode,
        "consensus_qualified": consensus_qualified,
        "ml_rank": 1,
        "ml_edge": 0.1,
        "ml_expected_value": ml_expected_value,
        "ml_probability": 0.6,
        "result": "pending",
    }


def test_merge_caps_each_game_and_player_by_ml_expected_value(tmp_path: Path):
    cache_dir = tmp_path / "data" / "player_props_cache"
    snapshot_dir = tmp_path / "data" / "player_props_snapshots"
    cache_dir.mkdir(parents=True)

    def candidate(game: str, index: int, expected_value: float, **overrides) -> dict:
        fields = {
            "game_id": game,
            "player_id": f"{game}-player-{index}",
            "ml_expected_value": expected_value,
            **overrides,
        }
        return _pick(
            f"{game}-{index}",
            f"{game}-{index}",
            f"{game} Player {index} Over 0.5 Hits",
            **fields,
        )

    current_game_a = [candidate("game-a", index, 0.70 - index * 0.02) for index in range(5)]
    current_game_a.append(
        candidate(
            "game-a",
            99,
            0.10,
            player_id="shared-player",
            selection="Under",
            line=1.5,
        )
    )
    generated_game_a = [candidate("game-a", index + 10, 0.90 - index * 0.03) for index in range(5)]
    generated_game_a.append(
        candidate(
            "game-a",
            98,
            0.95,
            player_id="shared-player",
            decision="LEAN",
            selection="Over",
            line=2.5,
        )
    )
    generated_game_b = [
        candidate("game-b", index, 0.55 if index < 2 else 0.55 - index * 0.02)
        for index in range(10)
    ]

    current = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "ranking_epoch": "MLB:player_props_consensus_v2.0.0:published:test",
                "picks": current_game_a,
            }
        },
    }
    generated = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "model_key": "mlb_player_props",
                "ranking_epoch": "MLB:player_props_consensus_v2.0.0:published:test",
                "picks": [*generated_game_a, *generated_game_b],
            }
        },
    }
    (cache_dir / "2026-06-20.json").write_text(json.dumps(current), encoding="utf-8")

    merged = merge_payload(generated, cache_dir, snapshot_dir)
    picks = merged["models"]["mlb_player_props"]["picks"]

    assert sum(pick["game_id"] == "game-a" for pick in picks) == 6
    assert sum(pick["game_id"] == "game-b" for pick in picks) == 8
    assert len({pick["player_id"] for pick in picks}) == len(picks)
    assert "game-a-98" in {pick["id"] for pick in picks}
    assert "game-a-99" not in {pick["id"] for pick in picks}
    assert picks[0]["id"] == "game-a-98"
    assert picks[0]["decision"] == "LEAN"
    assert [pick["ml_expected_value"] for pick in picks] == sorted(
        (pick["ml_expected_value"] for pick in picks),
        reverse=True,
    )
    assert [pick["ml_rank"] for pick in picks] == list(range(1, len(picks) + 1))

    reversed_generated = {
        **generated,
        "models": {
            "mlb_player_props": {
                **generated["models"]["mlb_player_props"],
                "picks": list(reversed(generated["models"]["mlb_player_props"]["picks"])),
            }
        },
    }
    reversed_picks = merge_payload(reversed_generated, cache_dir, snapshot_dir)["models"]["mlb_player_props"][
        "picks"
    ]
    assert [pick["id"] for pick in reversed_picks] == [pick["id"] for pick in picks]


def _wnba_3pm_pick(pick_id: str, market_id: str, *, source: str = "WNBA3PM", model_key: str = "wnba_3pm") -> dict:
    pick = _pick(pick_id, market_id, "Shooter Over 1.5 3-Point Field Goals")
    pick.update(
        {
            "source": source,
            "model_key": model_key,
            "sport": "WNBA",
            "stat_key": "three_pointers_made",
            "line": 1.5,
            "ml_rank_epoch": "WNBA3PM:player_props_consensus_v2.0.0:published:test",
        }
    )
    return pick


def test_merge_uses_only_fresh_generated_markets_for_latest_board(tmp_path: Path):
    cache_dir = tmp_path / "data" / "player_props_cache"
    snapshot_dir = tmp_path / "data" / "player_props_snapshots"
    cache_dir.mkdir(parents=True)
    (snapshot_dir / "2026-06-20").mkdir(parents=True)

    current = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "ranking_epoch": "MLB:player_props_consensus_v2.0.0:published:test",
                "picks": [_pick(f"current-{index}", f"current-{index}", f"Current {index}") for index in range(8)],
            }
        },
    }
    snapshot = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "picks": [_pick("snapshot-only", "snapshot-only", "Snapshot Only")],
            }
        },
    }
    generated = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "ranking_epoch": "MLB:player_props_consensus_v2.0.0:published:test",
                "picks": [_pick(f"generated-{index}", f"generated-{index}", f"Generated {index}") for index in range(8)],
            }
        },
    }

    (cache_dir / "2026-06-20.json").write_text(json.dumps(current), encoding="utf-8")
    (snapshot_dir / "2026-06-20" / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    merged = merge_payload(generated, cache_dir, snapshot_dir)
    merged_picks = merged["models"]["mlb_player_props"]["picks"]
    merged_keys = _published_player_prop_keys(merged, "2026-06-20")
    expected_keys = _published_player_prop_keys(generated, "2026-06-20")

    assert merged_keys == expected_keys
    assert len(merged_picks) == len(expected_keys)
    assert [pick["ml_rank"] for pick in merged_picks] == list(range(1, len(merged_picks) + 1))


def test_merge_preserves_grading_metadata_only_for_a_fresh_matching_market(tmp_path: Path):
    cache_dir = tmp_path / "data" / "player_props_cache"
    snapshot_dir = tmp_path / "data" / "player_props_snapshots"
    cache_dir.mkdir(parents=True)
    current_pick = _pick("current", "same-market", "Current")
    current_pick.update({"result": "win", "start_time": "2026-06-21T00:00:00Z"})
    generated_pick = _pick("generated", "same-market", "Generated")
    generated = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "model_key": "mlb_player_props",
                "ranking_epoch": "MLB:player_props_consensus_v2.0.0:published:test",
                "picks": [generated_pick],
            }
        },
    }
    current = {
        "date": "2026-06-20",
        "models": {"mlb_player_props": {"ok": True, "picks": [current_pick]}},
    }
    archived_pending = dict(current_pick)
    archived_pending["result"] = "pending"
    (cache_dir / "2026-06-20.json").write_text(json.dumps(current), encoding="utf-8")
    (snapshot_dir / "2026-06-20").mkdir(parents=True)
    (snapshot_dir / "2026-06-20" / "older.json").write_text(
        json.dumps({"date": "2026-06-20", "models": {"mlb_player_props": {"picks": [archived_pending]}}}),
        encoding="utf-8",
    )

    merged_pick = merge_payload(generated, cache_dir, snapshot_dir)["models"]["mlb_player_props"]["picks"][0]

    assert merged_pick["id"] == "generated"
    assert merged_pick["result"] == "win"
    assert merged_pick["start_time"] == "2026-06-21T00:00:00Z"
    assert "preserved_from_prior_refresh" not in merged_pick


def test_merge_does_not_force_rejected_variant_snapshots_into_latest_board(tmp_path: Path):
    cache_dir = tmp_path / "data" / "player_props_cache"
    snapshot_dir = tmp_path / "data" / "player_props_snapshots"
    cache_dir.mkdir(parents=True)
    (snapshot_dir / "2026-06-20").mkdir(parents=True)

    snapshot = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "picks": [
                    _pick(
                        "fallback-snapshot",
                        "fallback-snapshot",
                        "Rejected Variant",
                        consensus_qualified=False,
                        mode="all_time_variant",
                    )
                ],
            }
        },
    }
    generated = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "ranking_epoch": "MLB:player_props_consensus_v2.0.0:published:test",
                "picks": [_pick("generated", "generated", "Generated")],
            }
        },
    }

    (cache_dir / "2026-06-20.json").write_text(json.dumps({"date": "2026-06-20", "models": {}}), encoding="utf-8")
    (snapshot_dir / "2026-06-20" / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    merged = merge_payload(generated, cache_dir, snapshot_dir)
    picks = merged["models"]["mlb_player_props"]["picks"]

    assert [pick["id"] for pick in picks] == ["generated"]


def test_merge_keeps_wnba_3pm_research_bucket_out_of_public_cache(tmp_path: Path):
    cache_dir = tmp_path / "data" / "player_props_cache"
    snapshot_dir = tmp_path / "data" / "player_props_snapshots"
    cache_dir.mkdir(parents=True)
    (snapshot_dir / "2026-06-20").mkdir(parents=True)

    current = {
        "date": "2026-06-20",
        "models": {
            "wnba_player_props": {
                "ok": True,
                "picks": [
                    {
                        **_wnba_3pm_pick(
                            "generic-wnba",
                            "generic",
                            source="WNBAPlayerProps",
                            model_key="wnba_player_props",
                        ),
                        "stat_key": "points",
                    }
                ],
            },
            "wnba_3pm": {
                "ok": True,
                "picks": [_wnba_3pm_pick("current-3pm", "current")],
            },
        },
    }
    snapshot = {
        "date": "2026-06-20",
        "models": {
            "wnba_player_props": {"ok": True, "picks": [current["models"]["wnba_player_props"]["picks"][0]]},
            "wnba_3pm": {"ok": True, "picks": [_wnba_3pm_pick("snapshot-3pm", "snapshot")]},
        },
    }
    generated = {
        "date": "2026-06-20",
        "models": {
            "wnba_player_props": {
                "ok": True,
                "model_key": "wnba_player_props",
                "ranking_epoch": "WNBA:player_props_consensus_v2.0.0:published:test",
                "picks": [],
            },
            "wnba_3pm": {
                "ok": True,
                "model_key": "wnba_3pm",
                "ranking_epoch": "WNBA3PM:player_props_consensus_v2.0.0:published:test",
                "picks": [_wnba_3pm_pick("generated-3pm", "generated")],
            },
        },
    }

    (cache_dir / "2026-06-20.json").write_text(json.dumps(current), encoding="utf-8")
    (snapshot_dir / "2026-06-20" / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    merged = merge_payload(generated, cache_dir, snapshot_dir)
    assert set(merged["models"]) == {"wnba_player_props"}
    generic_picks = merged["models"]["wnba_player_props"]["picks"]

    assert generic_picks == []


def test_soft_football_preserves_prior_pass_research_on_failed_empty_refresh(tmp_path: Path):
    from scripts.merge_player_props_cache_payload import merge_payload

    date = "2026-09-30"
    prior_pick = {
        "id": "nfl-pass-1",
        "sport": "NFL",
        "date": date,
        "decision": "PASS",
        "units": 0,
        "full_kelly": 0,
        "quarter_kelly": 0,
        "player_name": "Example",
        "stat_key": "passing_yards",
        "market_retrieved_at": "2026-09-30T16:00:00Z",
    }
    current = {
        "date": date,
        "models": {
            "nfl_player_props": {
                "ok": True,
                "date": date,
                "picks": [prior_pick],
                "updatedAt": "2026-09-30T16:05:00Z",
            }
        },
    }
    (tmp_path / f"{date}.json").write_text(__import__("json").dumps(current), encoding="utf-8")
    generated = {
        "date": date,
        "models": {
            "nfl_player_props": {
                "ok": False,
                "date": date,
                "picks": [],
                "errors": ["synthetic crash"],
            }
        },
    }
    merged = merge_payload(generated, tmp_path)
    bucket = merged["models"]["nfl_player_props"]
    assert bucket["publication_status"] == "preserved_research"
    assert bucket["note"] == (
        "Refresh failed; retaining earlier same-day PASS research with its original quote timestamps."
    )
    assert len(bucket["picks"]) == 1
    assert bucket["picks"][0]["market_retrieved_at"] == "2026-09-30T16:00:00Z"
    assert bucket["ok"] is False
    assert bucket["errors"] == ["synthetic crash"]


def _priced_football_pick(
    sport: str,
    date: str,
    pick_id: str,
    player_id: str,
    *,
    line: float,
    units: float = 1.0,
) -> dict:
    return {
        "id": pick_id,
        "source": f"{sport}PlayerProps",
        "sport": sport,
        "date": date,
        "game_id": "game-1",
        "player_id": player_id,
        "player_name": player_id,
        "stat_key": "receiving_yards",
        "selection": "Over",
        "line": line,
        "pick": f"{player_id} Over {line} Receiving Yards",
        "matchup": "A @ B",
        "market_priced": True,
        "odds": -110,
        "decision": "BET",
        "units": units,
        "full_kelly": 0.04,
        "quarter_kelly": 0.01,
        "ml_expected_value": 0.08,
        "ml_probability": 0.58,
        "probability_source": "player_props_ml_v1",
        "market_retrieved_at": f"{date}T16:00:00Z",
        "result": "pending",
    }


@pytest.mark.parametrize(
    ("model_key", "sport", "via_snapshot"),
    [
        ("nfl_player_props", "NFL", False),
        ("cfb_player_props", "CFB", True),
    ],
)
def test_soft_football_failed_refresh_keeps_priced_same_day_picks(
    tmp_path: Path,
    model_key: str,
    sport: str,
    via_snapshot: bool,
):
    date = "2026-10-01"
    priced = [
        _priced_football_pick(sport, date, f"{sport.lower()}-a", "Alpha", line=64.5),
        _priced_football_pick(sport, date, f"{sport.lower()}-a2", "Alpha", line=72.5, units=0.25),
        _priced_football_pick(sport, date, f"{sport.lower()}-b", "Bravo", line=45.5, units=0.5),
    ]
    wrong_day = _priced_football_pick(sport, "2026-09-30", f"{sport.lower()}-old", "Old", line=10.5)
    prior_bucket = {
        "ok": True,
        "date": date,
        "updatedAt": f"{date}T16:05:00Z",
        "picks": [*priced, wrong_day],
    }
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snapshots"
    cache_dir.mkdir()
    board = {"date": date, "models": {model_key: prior_bucket}}
    if via_snapshot:
        day_dir = snapshot_dir / date
        day_dir.mkdir(parents=True)
        (day_dir / "board.json").write_text(json.dumps(board), encoding="utf-8")
    else:
        (cache_dir / f"{date}.json").write_text(json.dumps(board), encoding="utf-8")

    failed = {
        "date": date,
        "models": {
            model_key: {
                "ok": False,
                "date": date,
                "picks": [],
                "errors": ["consensus refresh failed"],
                "football_baseline": True,
            }
        },
    }
    bucket = merge_payload(failed, cache_dir, snapshot_dir)["models"][model_key]
    assert bucket["ok"] is False
    assert bucket["errors"] == ["consensus refresh failed"]
    assert bucket.get("football_baseline") is not True
    assert bucket["publication_status"] == "preserved_research"
    assert bucket["preserved_research_from"] == f"{date}T16:05:00Z"
    assert bucket["note"] == (
        "Refresh failed; retaining earlier same-day picks with their original quote timestamps."
    )
    assert {pick["id"] for pick in bucket["picks"]} == {pick["id"] for pick in priced}
    assert {pick["id"] for pick in bucket["picks"]}.isdisjoint({wrong_day["id"]})
    kept = {pick["id"]: pick for pick in bucket["picks"]}
    assert kept[priced[0]["id"]]["odds"] == -110
    assert kept[priced[0]["id"]]["units"] == 1.0
    assert kept[priced[0]["id"]]["market_retrieved_at"] == f"{date}T16:00:00Z"
    assert kept[priced[1]["id"]]["decision"] == "BET"

    nonempty = {
        "date": date,
        "models": {
            model_key: {
                "ok": True,
                "date": date,
                "picks": [_priced_football_pick(sport, date, f"{sport.lower()}-new", "Newer", line=70.5)],
                "errors": [],
            }
        },
    }
    replaced = merge_payload(nonempty, cache_dir, snapshot_dir)["models"][model_key]
    assert [pick["id"] for pick in replaced["picks"]] == [f"{sport.lower()}-new"]
    assert replaced.get("publication_status") != "preserved_research"

    mismatched = {
        "date": date,
        "models": {
            model_key: {
                "ok": False,
                "date": "2026-09-30",
                "picks": [],
                "errors": ["consensus refresh failed"],
            }
        },
    }
    blank = merge_payload(mismatched, cache_dir, snapshot_dir)["models"][model_key]
    assert blank["picks"] == []
    assert blank.get("publication_status") != "preserved_research"

    other_day = merge_payload({**failed, "date": "2026-09-30"}, cache_dir, snapshot_dir)
    assert other_day["models"][model_key]["picks"] == []

    warned = {
        "date": date,
        "models": {
            model_key: {
                "ok": True,
                "date": date,
                "picks": [],
                "errors": ["partial feed"],
            }
        },
    }
    warned_bucket = merge_payload(warned, cache_dir, snapshot_dir)["models"][model_key]
    assert {pick["id"] for pick in warned_bucket["picks"]} == {pick["id"] for pick in priced}
    assert warned_bucket["errors"] == ["partial feed"]
    assert warned_bucket["publication_status"] == "preserved_research"

    healthy_empty = {
        "date": date,
        "models": {
            model_key: {
                "ok": True,
                "date": date,
                "picks": [],
                "errors": [],
            }
        },
    }
    assert merge_payload(healthy_empty, cache_dir, snapshot_dir)["models"][model_key]["picks"] == []

    partial = {
        "date": date,
        "models": {
            model_key: {
                "ok": False,
                "date": date,
                "picks": [_priced_football_pick(sport, date, f"{sport.lower()}-partial", "Partial", line=12.5)],
                "errors": ["partial crash"],
            }
        },
    }
    partial_bucket = merge_payload(partial, cache_dir, snapshot_dir)["models"][model_key]
    assert partial_bucket["ok"] is False
    assert partial_bucket["errors"] == ["partial crash"]
    assert {pick["id"] for pick in partial_bucket["picks"]} == {pick["id"] for pick in priced}


def test_hard_player_prop_failure_does_not_preserve_prior_picks(tmp_path: Path):
    date = "2026-10-01"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prior = _priced_football_pick("NBA", date, "nba-priced", "Center", line=18.5)
    prior["sport"] = "NBA"
    prior["source"] = "NBAPlayerProps"
    (cache_dir / f"{date}.json").write_text(
        json.dumps({
            "date": date,
            "models": {
                "nba_player_props": {"ok": True, "date": date, "picks": [prior], "updatedAt": f"{date}T12:00:00Z"},
                "mlb_player_props": {
                    "ok": True,
                    "date": date,
                    "picks": [_pick("mlb-priced", "mlb-market", "Batter Over 0.5 Hits")],
                    "updatedAt": f"{date}T12:00:00Z",
                },
            },
        }),
        encoding="utf-8",
    )
    generated = {
        "date": date,
        "models": {
            "nba_player_props": {"ok": False, "date": date, "picks": [], "errors": ["nba down"]},
            "mlb_player_props": {"ok": False, "date": date, "picks": [], "errors": ["mlb down"]},
        },
    }
    merged = merge_payload(generated, cache_dir, tmp_path / "snapshots")
    assert merged["models"]["nba_player_props"]["picks"] == []
    assert merged["models"]["mlb_player_props"]["picks"] == []
    assert merged["models"]["nba_player_props"].get("publication_status") != "preserved_research"
    assert merged["models"]["nba_player_props"]["ok"] is False
    assert merged["models"]["nba_player_props"]["errors"] == ["nba down"]

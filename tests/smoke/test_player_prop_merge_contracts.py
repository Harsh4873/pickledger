from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_player_props_cache_payload import merge_payload
from scripts.site_upcheck import _published_player_prop_keys


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

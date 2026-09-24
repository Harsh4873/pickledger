"""Loss feedback: grade settled desk picks from rows already in the repo.

A new pick must keep the id that points at that settled record, and it must
carry the graded report. Missing source rows stay ungraded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_profit_desk as desk
from scripts.desk_loss_feedback import (
    LossFeedbackLinkError,
    attach_loss_feedback_link,
    grade_desk_loss_feedback,
    index_settled_sources,
    require_loss_feedback_link,
)
from scripts.merge_external_feed_cache_payload import (
    _preserve_pick_metadata as preserve_feed_pick,
)
from scripts.merge_model_cache_payload import _preserve_pick_metadata as preserve_model_pick
from tests.smoke.test_profit_desk import LIVE_DATE, make_payload, make_pick


ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-07-22"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _identity(pick: dict, *, mode: str = "player") -> str:
    payload = make_payload([pick], slate_date=pick["date"])
    context = next(desk._iter_records(payload, mode))
    return desk.canonical_market_identity(
        context.record,
        mode=context.mode,
        sport=str(pick.get("sport") or ""),
        date_iso=pick["date"],
    )


def _desk_row(pick: dict, *, result: str, identity: str, stake: float = 0.5, decimal_odds: float = 1.591716) -> dict:
    return {
        "id": "profit-row",
        "date": pick["date"],
        "sourceKey": "test",
        "sourcePickId": pick.get("id") or None,
        "pick": pick["pick"],
        "sport": pick.get("sport"),
        "market": pick.get("market_type"),
        "lane": "value",
        "stakeUnits": stake,
        "decimalOdds": decimal_odds,
        "oddsAmerican": pick.get("odds"),
        "marketIdentity": identity,
        "result": result,
    }


def _layout(tmp_path: Path, pick: dict, *, desk_result: str, cache_result: str | None, snapshot_result: str | None = None):
    identity = _identity(pick)
    row = _desk_row(pick, result=desk_result, identity=identity)
    profit = tmp_path / "profit"
    model = tmp_path / "model"
    props = tmp_path / "props"
    snaps = tmp_path / "snaps"
    _write(profit / f"{DATE}.json", {"date": DATE, "summary": {}, "portfolio": {"live": [row]}})
    if cache_result is not None:
        cached = dict(pick, result=cache_result)
        _write(props / f"{DATE}.json", make_payload([cached], slate_date=DATE))
    if snapshot_result is not None:
        snapped = dict(pick, result=snapshot_result)
        _write(snaps / DATE / "2026-07-22T21-00-00Z_test.json", make_payload([snapped], slate_date=DATE))
    return profit, model, props, snaps


def _grade(tmp_path: Path, pick: dict, **kwargs) -> dict:
    profit, model, props, snaps = _layout(tmp_path, pick, **kwargs)
    return grade_desk_loss_feedback(
        tmp_path,
        profit_dir=profit,
        model_dir=model,
        player_dir=props,
        snapshot_dir=snaps,
    )


def _prop(**overrides) -> dict:
    payload = make_pick(
        pick="Wyatt Langford Under 0.5 Walks",
        game="Texas Rangers @ Boston Red Sox",
        slate_date=DATE,
        player="Wyatt Langford",
        direction="Under",
        line=0.5,
        market="batter_walks",
        odds=-169,
        pick_id="pp_langford",
        result="pending",
    )
    payload.update(overrides)
    return payload


def test_source_loss_overrides_a_desk_win_and_a_missing_row_stays_ungraded(tmp_path: Path):
    report = _grade(tmp_path, _prop(), desk_result="win", cache_result="loss")
    graded = report["picks"][0]

    assert graded["graded"] is True
    assert graded["result"] == "loss"
    assert graded["deskResult"] == "win"
    assert graded["disagreesWithDesk"] is True
    assert graded["resultSource"] == "player_props_cache"
    assert graded["profitUnits"] == pytest.approx(-0.5)
    assert graded["sourcePickId"] == "pp_langford"
    assert report["summary"]["disagreements"] == 1
    assert report["summary"]["sourceGradedRecord"]["losses"] == 1
    assert report["summary"]["sourceGradedRecord"]["wins"] == 0

    # The desk's own "win" is not a result. With no source row, nothing is scored.
    bare = tmp_path / "bare"
    report = _grade(bare, _prop(pick="Nobody Under 0.5 Hits", pick_id="pp_missing"), desk_result="win", cache_result=None)
    missing = report["picks"][0]
    assert missing["graded"] is False
    assert missing["result"] is None
    assert missing["profitUnits"] is None
    assert missing["ungradedReason"] == "source_row_missing"
    assert report["summary"]["sourceGradedRecord"]["settled"] == 0


def test_agreeing_snapshot_grades_a_row_the_dated_cache_dropped(tmp_path: Path):
    report = _grade(
        tmp_path,
        _prop(pick="Zebby Matthews Under 4.5 Strikeouts", player="Zebby Matthews", line=4.5, market="strikeouts", pick_id="pp_zebby"),
        desk_result="pending",
        cache_result=None,
        snapshot_result="loss",
    )
    row = report["picks"][0]
    assert row["graded"] is True
    assert row["result"] == "loss"
    assert row["resultSource"] == "player_props_snapshot"
    assert row["disagreesWithDesk"] is False
    assert row["profitUnits"] == pytest.approx(-0.5)


def test_conflicting_snapshots_are_not_graded(tmp_path: Path):
    pick = _prop()
    profit, model, props, snaps = _layout(tmp_path, pick, desk_result="pending", cache_result=None, snapshot_result="win")
    _write(
        snaps / DATE / "2026-07-22T22-00-00Z_other.json",
        make_payload([dict(pick, result="loss")], slate_date=DATE),
    )
    report = grade_desk_loss_feedback(
        tmp_path,
        profit_dir=profit,
        model_dir=model,
        player_dir=props,
        snapshot_dir=snaps,
    )
    row = report["picks"][0]
    assert row["graded"] is False
    assert row["result"] is None
    assert row["ungradedReason"] == "source_result_conflict"


def test_dated_cache_wins_over_a_disagreeing_snapshot(tmp_path: Path):
    report = _grade(tmp_path, _prop(), desk_result="win", cache_result="loss", snapshot_result="win")
    row = report["picks"][0]
    assert row["result"] == "loss"
    assert row["resultSource"] == "player_props_cache"


def test_new_pick_cannot_ship_without_a_link_to_the_settled_record():
    report = {
        "reportId": "desk-grade-abc123",
        "summary": {
            "sourceGradedRecord": {
                "wins": 53,
                "losses": 31,
                "pushes": 1,
                "settled": 84,
                "netUnits": -5.5,
                "stakedUnits": 45.5,
                "roi": -0.12,
                "throughDate": "2026-09-23",
            }
        },
    }
    payload = {
        "models": {
            "mls": {
                "picks": [
                    {
                        "decision": "BET",
                        "pick": "Under 3.5 (Real Salt Lake @ Seattle Sounders FC)",
                        "sport": "MLS",
                        "date": "2026-09-23",
                        "source": "MLS Model",
                    }
                ]
            }
        }
    }

    attach_loss_feedback_link(payload, report)
    pick = payload["models"]["mls"]["picks"][0]
    assert pick["id"].startswith("pick-")
    assert pick["lossFeedback"]["reportId"] == "desk-grade-abc123"
    assert pick["lossFeedback"]["reportPath"] == "data/loss_feedback/desk_grade.json"
    assert pick["lossFeedback"]["settled"] == 84
    require_loss_feedback_link(payload)

    pick.pop("lossFeedback")
    with pytest.raises(LossFeedbackLinkError, match="no link to the settled desk record"):
        require_loss_feedback_link(payload)


def test_existing_pick_id_is_kept_when_the_report_link_is_stamped():
    report = {"reportId": "desk-grade-keep", "summary": {"sourceGradedRecord": {}}}
    payload = {"models": {"mlb_player_props": {"picks": [{"id": "pp_keep", "pick": "Ace Under 0.5"}]}}}
    attach_loss_feedback_link(payload, report)
    assert payload["models"]["mlb_player_props"]["picks"][0]["id"] == "pp_keep"
    assert payload["models"]["mlb_player_props"]["picks"][0]["lossFeedback"]["reportId"] == "desk-grade-keep"


def test_regenerated_pick_keeps_the_id_that_links_its_settled_record():
    current = {
        "picks": [
            {
                "id": "pick-settled-1",
                "result": "loss",
                "pick": "Under 3.5",
                "sport": "MLS",
                "date": "2026-09-23",
                "source": "MLS Model",
            }
        ]
    }
    regenerated = {
        "picks": [
            {
                "pick": "Under 3.5",
                "sport": "MLS",
                "date": "2026-09-23",
                "source": "MLS Model",
                "decision": "BET",
                "units": 1,
            }
        ]
    }
    for preserve in (preserve_model_pick, preserve_feed_pick):
        merged = preserve(current, regenerated)
        shipped = merged["picks"][0]
        assert shipped["id"] == "pick-settled-1"
        assert shipped["result"] == "loss"
        assert shipped["decision"] == "BET"


def test_new_desk_candidate_stores_the_source_pick_id():
    pick = make_pick(slate_date=LIVE_DATE, pick_id="pp_source_1")
    built = desk.build_profit_desk_payload(
        LIVE_DATE,
        make_payload([pick], slate_date=LIVE_DATE),
        None,
        team_history=[],
        prop_history=[],
    )
    assert built["candidates"][0]["sourcePickId"] == "pp_source_1"

    unlabeled = make_pick(slate_date=LIVE_DATE, pick_id="")
    built = desk.build_profit_desk_payload(
        LIVE_DATE,
        make_payload([unlabeled], slate_date=LIVE_DATE),
        None,
        team_history=[],
        prop_history=[],
    )
    assert built["candidates"][0]["sourcePickId"] is None


def test_sync_corrects_a_stale_desk_result_from_the_cache_and_from_a_snapshot(tmp_path: Path):
    pick = _prop(result="win")
    identity = _identity(pick)
    profit = tmp_path / "profit"
    model = tmp_path / "model"
    props = tmp_path / "props"
    snaps = tmp_path / "snaps"
    model.mkdir()
    row = _desk_row(pick, result="win", identity=identity)
    _write(
        profit / f"{DATE}.json",
        {
            "date": DATE,
            "summary": {"liveRecord": {"wins": 1, "losses": 0}},
            "candidates": [row],
            "portfolio": {"live": [dict(row)], "all": [dict(row)]},
        },
    )
    _write(props / f"{DATE}.json", make_payload([dict(pick, result="loss")], slate_date=DATE))

    changed = desk._sync_artifact_results(profit, model, props, snaps)
    assert changed == 1
    synced = json.loads((profit / f"{DATE}.json").read_text(encoding="utf-8"))
    assert synced["candidates"][0]["result"] == "loss"
    assert synced["portfolio"]["live"][0]["result"] == "loss"
    assert synced["summary"]["liveRecord"]["losses"] == 1
    assert synced["candidates"][0]["pick"] == row["pick"]
    assert synced["candidates"][0]["stakeUnits"] == row["stakeUnits"]

    # Dated cache no longer has the row. An agreeing snapshot is the result.
    dropped = _prop(
        pick="Zebby Matthews Under 4.5 Strikeouts",
        player="Zebby Matthews",
        line=4.5,
        market="strikeouts",
        pick_id="pp_zebby",
        result="pending",
    )
    dropped_identity = _identity(dropped)
    pending = _desk_row(dropped, result="pending", identity=dropped_identity)
    _write(
        profit / f"{DATE}.json",
        {"date": DATE, "summary": {}, "candidates": [pending], "portfolio": {"live": [dict(pending)], "all": [dict(pending)]}},
    )
    _write(props / f"{DATE}.json", make_payload([], slate_date=DATE))
    _write(snaps / DATE / "snap.json", make_payload([dict(dropped, result="loss")], slate_date=DATE))
    desk._sync_artifact_results(profit, model, props, snaps)
    synced = json.loads((profit / f"{DATE}.json").read_text(encoding="utf-8"))
    assert synced["portfolio"]["live"][0]["result"] == "loss"


def test_real_desk_rows_are_graded_only_from_stored_results():
    report = grade_desk_loss_feedback(ROOT)
    summary = report["summary"]
    assert summary["deskRows"] == 90
    assert summary["graded"] + summary["ungraded"] == summary["deskRows"]
    published = summary["publishedDeskRecord"]
    assert published["wins"] == 54
    assert published["losses"] == 30
    assert published["pushes"] == 1
    assert published["settled"] == 84
    assert published["netUnits"] == pytest.approx(-4.7926)

    by_pick = {row["pick"]: row for row in report["picks"]}
    langford = by_pick["Wyatt Langford Under 0.5 Walks"]
    assert langford["deskResult"] == "win"
    assert langford["result"] == "loss"
    assert langford["resultSource"] == "player_props_cache"
    assert langford["disagreesWithDesk"] is True
    assert langford["profitUnits"] == pytest.approx(-0.5)

    zebby = by_pick["Zebby Matthews Under 4.5 Strikeouts"]
    assert zebby["deskResult"] == "pending"
    assert zebby["result"] == "loss"
    assert zebby["resultSource"] == "player_props_snapshot"

    for row in report["picks"]:
        if row["graded"]:
            assert row["result"] in {"win", "loss", "push"}
            assert row["resultSource"] in {"model_cache", "player_props_cache", "player_props_snapshot"}
            assert row["resultPath"]
        else:
            assert row["result"] is None
            assert row["profitUnits"] is None
            assert row["ungradedReason"]

    # The source grade is allowed to differ from the published desk tally.
    # It must still be computed, not copied.
    graded = summary["sourceGradedRecord"]
    assert graded["wins"] + graded["losses"] + graded["pushes"] == summary["graded"]
    assert report["reportId"].startswith("desk-grade-")
    indexed = index_settled_sources(
        ROOT / "data" / "model_cache",
        ROOT / "data" / "player_props_cache",
        ROOT / "data" / "player_props_snapshots",
        DATE,
        repo_root=ROOT,
    )
    assert indexed["by_source_pick_id"][("mlb_player_props", "pp_0bd3594da56cfd16a230_consensus")]["result"] == "loss"

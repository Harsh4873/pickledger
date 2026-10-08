from datetime import datetime, timezone

from scripts.capture_closing_lines import _default_slate_date


def test_default_slate_date_stays_on_central_day_after_utc_midnight():
    # 00:30 UTC is still 19:30 on the prior Central date during daylight time.
    now = datetime(2026, 9, 6, 0, 30, tzinfo=timezone.utc)
    assert _default_slate_date(now) == "2026-09-05"


def test_merge_saved_closing_lines_restores_rows_without_duplicates(tmp_path, monkeypatch):
    import json

    from scripts import capture_closing_lines as module

    live = tmp_path / "live"
    saved = tmp_path / "saved"
    live.mkdir()
    saved.mkdir()
    monkeypatch.setattr(module, "CLOSING_LINES_DIR", live)
    shared = {"marketIdentity": "m1", "capturedAt": "2026-10-08T23:00:00Z", "provider": "p"}
    retried_away = {"marketIdentity": "m2", "capturedAt": "2026-10-08T23:05:00Z", "provider": "p"}
    (live / "2026-10-08.json").write_text(json.dumps({"date": "2026-10-08", "rows": [shared]}))
    (saved / "2026-10-08.json").write_text(json.dumps({"date": "2026-10-08", "rows": [shared, retried_away]}))

    assert module.merge_saved_closing_lines(saved) == 1
    rows = json.loads((live / "2026-10-08.json").read_text())["rows"]
    assert [row["marketIdentity"] for row in rows] == ["m1", "m2"]
    assert module.merge_saved_closing_lines(saved) == 0

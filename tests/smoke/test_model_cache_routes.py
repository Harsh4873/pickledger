from __future__ import annotations

import json


def test_load_cached_model_result_reads_static_json(monkeypatch, tmp_path):
    import pickgrader_server as server

    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "2026-06-05.json").write_text(
        json.dumps({
            "date": "2026-06-05",
            "models": {
                "mlb_first_five": {
                    "ok": True,
                    "picks": [{"pick": "Over 5.5 F5"}],
                    "games": [{"matchup": "Giants @ Cubs"}],
                }
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_init_admin_firestore", lambda: None)

    result = server._load_cached_model_result("2026-06-05", "mlb_first_five")

    assert result is not None
    assert result["ok"] is True
    assert result["source"] == "firebase_cache"
    assert result["cache_source"] == "static_json"
    assert result["picks"] == [{"pick": "Over 5.5 F5"}]
    assert result["games"] == [{"matchup": "Giants @ Cubs"}]


def test_cached_model_response_skips_async_launch(monkeypatch):
    import pickgrader_server as server

    cached = {
        "ok": True,
        "source": "firebase_cache",
        "cache_source": "firestore",
        "picks": [{"pick": "Inning 3 - No Run Scored"}],
    }

    def fail_launch(*args, **kwargs):
        raise AssertionError("async job should not launch on cache hit")

    def fail_model(*args, **kwargs):
        raise AssertionError("model should not run on cache hit")

    monkeypatch.setattr(server, "_load_cached_model_result", lambda date, model: cached)
    monkeypatch.setattr(server, "_launch_job", fail_launch)

    result = server._cached_or_model_response(
        "mlb_inning",
        "2026-06-05",
        fail_model,
        ("2026-06-05",),
        async_mode=True,
        force_refresh=False,
    )

    assert result is cached


def test_force_refresh_bypasses_cache_and_launches_async_job(monkeypatch):
    import pickgrader_server as server

    monkeypatch.setattr(server, "_load_cached_model_result", lambda date, model: {"ok": True, "picks": []})
    monkeypatch.setattr(server, "_launch_job", lambda fn, *args: "job-123")

    result = server._cached_or_model_response(
        "mlb_first_five",
        "2026-06-05",
        lambda *_: {"ok": True, "picks": []},
        ("2026-06-05",),
        async_mode=True,
        force_refresh=True,
    )

    assert result == {"ok": True, "job_id": "job-123", "status": "running"}

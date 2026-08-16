from __future__ import annotations

import json

from scripts import train_player_prop_ml as trainer


def _candidate_rows():
    rows = [[0.0] * len(trainer.FEATURE_NAMES) for _ in range(30)]
    labels = [index % 2 for index in range(30)]
    dates = ["2026-06-18"] * 15 + ["2026-06-19"] * 15
    probabilities = [0.5] * 30
    return rows, labels, dates, probabilities, probabilities


def _failed_validation(*_args, **_kwargs):
    return {
        "samples": 15,
        "dates": ["2026-06-19"],
        "model_brier": 0.30,
        "market_brier": 0.25,
        "baseline_brier": 0.28,
        "calibration_gap": 0.10,
    }


def _configure_candidate(monkeypatch, tmp_path):
    model_path = tmp_path / "candidate.joblib"
    metadata_path = tmp_path / "candidate.json"
    monkeypatch.setitem(
        trainer.SPORT_ARTIFACTS,
        "WNBA",
        {"model": model_path, "metadata": metadata_path, "artifact_sport": "WNBA"},
    )
    monkeypatch.setattr(trainer, "_ledger_rows", lambda *_args, **_kwargs: _candidate_rows())
    monkeypatch.setattr(trainer, "_forward_validation", _failed_validation)
    monkeypatch.setattr(trainer, "_fit_classifier", lambda *_args, **_kwargs: object())
    return model_path, metadata_path


def test_rejected_candidate_does_not_replace_an_active_artifact(monkeypatch, tmp_path):
    model_path, metadata_path = _configure_candidate(monkeypatch, tmp_path)
    model_path.write_bytes(b"trusted-model")
    metadata_path.write_text(json.dumps({"active": True}), encoding="utf-8")

    result = trainer._fit_artifact(
        sport="WNBA",
        families=trainer.WNBA_FAMILIES,
        repo_root=tmp_path,
        force=True,
        dry_run=False,
    )

    assert result["candidate_rejected"] is True
    assert result["changed"] is False
    assert model_path.read_bytes() == b"trusted-model"
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {"active": True}


def test_dry_run_never_writes_candidate_artifacts(monkeypatch, tmp_path):
    model_path, metadata_path = _configure_candidate(monkeypatch, tmp_path)

    result = trainer._fit_artifact(
        sport="WNBA",
        families=trainer.WNBA_FAMILIES,
        repo_root=tmp_path,
        force=True,
        dry_run=True,
    )

    assert result["candidate"]["active"] is False
    assert result["changed"] is False
    assert not model_path.exists()
    assert not metadata_path.exists()


def test_dry_run_evaluates_even_when_an_artifact_already_exists(monkeypatch, tmp_path):
    model_path, metadata_path = _configure_candidate(monkeypatch, tmp_path)
    model_path.write_bytes(b"existing-model")
    metadata_path.write_text(json.dumps({"active": True}), encoding="utf-8")

    result = trainer._fit_artifact(
        sport="WNBA",
        families=trainer.WNBA_FAMILIES,
        repo_root=tmp_path,
        force=False,
        dry_run=True,
    )

    assert result["changed"] is False
    assert result["candidate"]["active"] is False
    assert model_path.read_bytes() == b"existing-model"
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {"active": True}

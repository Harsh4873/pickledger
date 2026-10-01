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


def test_precision_fit_skips_empty_or_single_class():
    import pandas as pd

    from scripts.train_player_prop_precision_ml import NUMERIC_FEATURES, CATEGORICAL_FEATURES, _fit

    empty = pd.DataFrame(columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["over_outcome"])
    assert _fit(empty) is None
    single = pd.DataFrame(
        [{**{name: 0.0 for name in NUMERIC_FEATURES}, **{name: "x" for name in CATEGORICAL_FEATURES}, "over_outcome": 1}]
    )
    assert _fit(single) is None


def test_prop_ml_main_exits_nonzero_when_football_skipped_without_allow_skip(monkeypatch):
    from scripts import train_player_prop_ml as train

    def fake_fit(**kwargs):
        sport = kwargs["sport"]
        if sport in {"NFL", "CFB"}:
            return {"sport": sport, "skipped": True, "reason": "synthetic", "path": f"{sport}.joblib", "changed": False}
        return {"sport": sport, "skipped": False, "path": f"{sport}.joblib", "changed": False}

    monkeypatch.setattr(train, "_fit_artifact", fake_fit)
    monkeypatch.setattr(
        train,
        "_parse_args" if hasattr(train, "_parse_args") else "main",
        train.main,
    )
    # Patch argv via parse inside main
    import sys
    monkeypatch.setattr(sys, "argv", ["train_player_prop_ml.py"])
    # Bypass rebuild and call main with patched ArgumentParser
    from types import SimpleNamespace
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(repo_root=train.REPO_ROOT, force=False, dry_run=True, rebuild_ledger=False, allow_skip=False),
    )
    assert train.main() == 2
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(repo_root=train.REPO_ROOT, force=False, dry_run=True, rebuild_ledger=False, allow_skip=True),
    )
    assert train.main() == 0


def test_prop_ml_allow_skip_emits_github_actions_warning(monkeypatch, capsys):
    from scripts import train_player_prop_ml as train
    from types import SimpleNamespace
    import sys

    def fake_fit(**kwargs):
        sport = kwargs["sport"]
        if sport in {"NFL", "CFB"}:
            return {"sport": sport, "skipped": True, "reason": "synthetic", "path": f"{sport}.joblib", "changed": False}
        return {"sport": sport, "skipped": False, "path": f"{sport}.joblib", "changed": False}

    monkeypatch.setattr(train, "_fit_artifact", fake_fit)
    monkeypatch.setattr(sys, "argv", ["train_player_prop_ml.py"])
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            repo_root=train.REPO_ROOT, force=False, dry_run=True, rebuild_ledger=False, allow_skip=True
        ),
    )
    assert train.main() == 0
    err = capsys.readouterr().err
    assert "::warning title=Player prop ML soft-skip::" in err
    assert "NFL" in err and "CFB" in err


class _ProbaModel:
    def predict_proba(self, matrix):
        import numpy as np

        count = len(matrix)
        return np.column_stack([np.full(count, 0.45), np.full(count, 0.55)])


def _dated_rows(counts_by_date: list[tuple[str, int]], *, marker_by_date: dict[str, float] | None = None):
    marker_by_date = marker_by_date or {}
    width = len(trainer.FEATURE_NAMES)
    rows: list[list[float]] = []
    labels: list[int] = []
    dates: list[str] = []
    for record_date, count in counts_by_date:
        marker = marker_by_date.get(record_date, 0.0)
        for index in range(count):
            row = [0.0] * width
            row[0] = marker
            rows.append(row)
            labels.append(index % 2)
            dates.append(record_date)
    probabilities = [0.5] * len(rows)
    return rows, labels, dates, probabilities, probabilities


def _patch_wnba_artifact(monkeypatch, tmp_path, ledger_rows):
    model_path = tmp_path / "published.joblib"
    metadata_path = tmp_path / "published.json"
    monkeypatch.setitem(
        trainer.SPORT_ARTIFACTS,
        "WNBA",
        {"model": model_path, "metadata": metadata_path, "artifact_sport": "WNBA"},
    )
    monkeypatch.setattr(trainer, "_ledger_rows", lambda *_args, **_kwargs: ledger_rows)
    return model_path, metadata_path


def test_published_fit_excludes_final_validation_date(monkeypatch, tmp_path):
    early_date = "2026-06-18"
    final_date = "2026-06-19"
    early_count = 40
    ledger_rows = _dated_rows(
        [(early_date, early_count), (final_date, 12)],
        marker_by_date={early_date: 1.0, final_date: 2.0},
    )
    _patch_wnba_artifact(monkeypatch, tmp_path, ledger_rows)
    fit_calls: list[tuple[list[list[float]], list[int]]] = []

    def _capture_fit(fit_rows, fit_labels):
        fit_calls.append((fit_rows, fit_labels))
        return _ProbaModel()

    monkeypatch.setattr(trainer, "_fit_classifier", _capture_fit)

    result = trainer._fit_artifact(
        sport="WNBA",
        families=trainer.WNBA_FAMILIES,
        repo_root=tmp_path,
        force=True,
        dry_run=True,
    )

    candidate = result["candidate"]
    assert candidate["full_sample_refit"] is False
    assert candidate["published_fit"] == "pre_final_validation_date"
    assert candidate["published_fit_end_exclusive"] == final_date
    assert candidate["published_fit_end_exclusive"] == max(candidate["validation"]["dates"])
    assert candidate["published_fit_samples"] == early_count
    assert candidate["validation_describes"] == "walk_forward_selection_not_published_fit"
    assert final_date in candidate["validation"]["dates"]
    published_rows, published_labels = fit_calls[-1]
    assert len(published_rows) == early_count
    assert len(published_rows) < len(ledger_rows[0])
    assert {row[0] for row in published_rows} == {1.0}
    assert published_labels == ledger_rows[1][:early_count]


def test_published_fit_keeps_earlier_validation_dates(monkeypatch, tmp_path):
    first_date = "2026-06-17"
    middle_date = "2026-06-18"
    final_date = "2026-06-19"
    ledger_rows = _dated_rows(
        [(first_date, 40), (middle_date, 10), (final_date, 8)],
        marker_by_date={first_date: 1.0, middle_date: 2.0, final_date: 3.0},
    )
    _patch_wnba_artifact(monkeypatch, tmp_path, ledger_rows)
    fit_calls: list[list[list[float]]] = []

    def _capture_fit(fit_rows, _fit_labels):
        fit_calls.append(fit_rows)
        return _ProbaModel()

    monkeypatch.setattr(trainer, "_fit_classifier", _capture_fit)

    result = trainer._fit_artifact(
        sport="WNBA",
        families=trainer.WNBA_FAMILIES,
        repo_root=tmp_path,
        force=True,
        dry_run=True,
    )

    candidate = result["candidate"]
    assert candidate["full_sample_refit"] is False
    assert candidate["published_fit_end_exclusive"] == final_date
    assert candidate["published_fit_samples"] == 50
    assert set(candidate["validation"]["dates"]) == {middle_date, final_date}
    published_markers = {row[0] for row in fit_calls[-1]}
    assert published_markers == {1.0, 2.0}


def test_published_fit_uses_all_rows_when_walk_forward_has_no_dates(monkeypatch, tmp_path):
    ledger_rows = _dated_rows([("2026-06-18", 20), ("2026-06-19", 20)])
    _patch_wnba_artifact(monkeypatch, tmp_path, ledger_rows)
    fit_calls: list[int] = []

    def _capture_fit(fit_rows, _fit_labels):
        fit_calls.append(len(fit_rows))
        return _ProbaModel()

    monkeypatch.setattr(trainer, "_fit_classifier", _capture_fit)

    result = trainer._fit_artifact(
        sport="WNBA",
        families=trainer.WNBA_FAMILIES,
        repo_root=tmp_path,
        force=True,
        dry_run=True,
    )

    candidate = result["candidate"]
    assert candidate["validation"]["dates"] == []
    assert candidate["full_sample_refit"] is True
    assert candidate["published_fit"] == "all_rows_no_validation_dates"
    assert candidate["published_fit_samples"] == 40
    assert "validation dates" in candidate["full_sample_refit_reason"]
    assert candidate["validation_describes"] == "walk_forward_selection_not_published_fit"
    assert fit_calls == [40]


def test_published_fit_falls_back_when_pre_holdout_is_too_small(monkeypatch, tmp_path):
    holdout_date = "2026-06-19"
    ledger_rows = _dated_rows([(holdout_date, 36)])
    _patch_wnba_artifact(monkeypatch, tmp_path, ledger_rows)
    fit_calls: list[int] = []

    def _capture_fit(fit_rows, _fit_labels):
        fit_calls.append(len(fit_rows))
        return object()

    monkeypatch.setattr(
        trainer,
        "_forward_validation",
        lambda *_args, **_kwargs: {"samples": 0, "dates": [holdout_date]},
    )
    monkeypatch.setattr(trainer, "_fit_classifier", _capture_fit)

    result = trainer._fit_artifact(
        sport="WNBA",
        families=trainer.WNBA_FAMILIES,
        repo_root=tmp_path,
        force=True,
        dry_run=True,
    )

    candidate = result["candidate"]
    assert candidate["full_sample_refit"] is True
    assert candidate["published_fit"] == "all_rows_insufficient_pre_holdout"
    assert candidate["published_fit_end_exclusive"] == holdout_date
    assert candidate["published_fit_samples"] == 36
    assert "0 samples" in candidate["full_sample_refit_reason"]
    assert fit_calls == [36]


def test_activation_training_floor_counts_only_published_fit_rows(monkeypatch, tmp_path):
    rows = _dated_rows([("2026-06-17", 30), ("2026-06-18", 20), ("2026-06-19", 50)])
    _patch_wnba_artifact(monkeypatch, tmp_path, rows)
    monkeypatch.setattr(trainer, "_fit_classifier", lambda *_args: _ProbaModel())
    monkeypatch.setattr(trainer, "_forward_validation", lambda *_args: {
        "dates": ["2026-06-18", "2026-06-19"], "samples": 70,
        "model_brier": 0.1, "market_brier": 0.25, "baseline_brier": 0.25,
        "calibration_gap": 0.01,
    })
    result = trainer._fit_artifact(
        sport="WNBA", families=trainer.WNBA_FAMILIES, repo_root=tmp_path, force=True, dry_run=True,
    )
    candidate = result["candidate"]
    assert candidate["ledger_samples"] == 100
    assert candidate["published_fit_samples"] == 50
    assert candidate["active"] is False

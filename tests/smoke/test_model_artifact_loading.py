"""Exercise real football artifacts and failure diagnostics without network IO."""

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_cold_concurrent_football_artifacts():
    # A fresh interpreter matters: an already-imported sklearn hides the race.
    code = """
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from NFLPredictionModel.nfl_model import _load_artifacts as nfl
from CFBPredictionModel.cfb_model import _load_artifacts as cfb
barrier = Barrier(3)
def load(loader):
    barrier.wait()
    return loader()
def mls():
    import MLSPredictionModel
    return True
with ThreadPoolExecutor(max_workers=3) as pool:
    assert all(pool.map(load, (mls, nfl, cfb)))
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=45)


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_artifact_failure_includes_original_exception(monkeypatch, tmp_path, caplog, sport):
    if sport == "nfl":
        from NFLPredictionModel import nfl_model as model
        rows = model.load_games(refresh=False)
        assert rows
        monkeypatch.setattr(model, "load_games", lambda: rows)
        monkeypatch.setattr(model, "ARTIFACT_DIR", tmp_path)
    else:
        from CFBPredictionModel import cfb_model as model
        monkeypatch.setattr(model, "ARTIFACT_PATH", tmp_path / "missing.joblib")
    result = getattr(model, f"generate_{sport}_picks")("2026-10-08")
    assert result["ok"] is False
    assert result["picks"] == []
    assert "FileNotFoundError:" in result["error"]
    assert str(tmp_path) in result["error"]
    assert "FileNotFoundError" in caplog.text

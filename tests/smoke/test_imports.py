from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def prepended_sys_paths(*paths: Path):
    original = list(sys.path)
    for path in reversed([str(p) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def test_locked_runtime_imports_still_resolve(monkeypatch):
    if importlib.util.find_spec("config") is None:
        config_stub = types.ModuleType("config")
        config_stub.BDL_API_KEY = ""
        config_stub.RUN_WNBA = False
        monkeypatch.setitem(sys.modules, "config", config_stub)

    imports = [
        ("pickgrader_server", (REPO_ROOT,)),
        ("runlive", (REPO_ROOT,)),
        ("MLBPredictionModel.date_utils", (REPO_ROOT,)),
        ("NBAPredictionModel.run_live", (REPO_ROOT / "NBAPredictionModel", REPO_ROOT)),
        ("WNBAPredictionModel.wnba_picks", (REPO_ROOT,)),
        ("FIFAWorldCupPredictionModel.fifa_world_cup_model", (REPO_ROOT,)),
        ("NBAPlayerBettingModel.run_props", (REPO_ROOT / "NBAPlayerBettingModel", REPO_ROOT)),
        ("NBAPlayoffsPredictionModel.run_live", (REPO_ROOT,)),
        ("models.mlb_inning.mlb_inning_model", (REPO_ROOT,)),
        ("models.mlb_first_five.mlb_first_five_model", (REPO_ROOT,)),
        ("ipl.run_api", (REPO_ROOT,)),
        ("scripts.auto_grade_picks", (REPO_ROOT,)),
        ("scripts.cache_manifest", (REPO_ROOT,)),
    ]

    for module_name, paths in imports:
        with prepended_sys_paths(*paths):
            importlib.import_module(module_name)

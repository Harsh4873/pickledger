"""Test-time stubs so smoke tests don't depend on local-only config files."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WNBA_DIR = REPO_ROOT / "WNBAPredictionModel"

# Make sure repo root + WNBA package dir are importable for tests that load
# WNBA modules with absolute names (the model's own try/except expects either
# relative or top-level `wnba_*` resolution).
for path in (REPO_ROOT, WNBA_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

# `config.py` is a local-only module that production runs depend on. Stub it
# for tests so importing model packages doesn't blow up. We attach a real
# ModuleSpec so importlib.util.find_spec("config") can still resolve it.
if "config" not in sys.modules and importlib.util.find_spec("config") is None:
    stub = types.ModuleType("config")
    stub.__spec__ = importlib.machinery.ModuleSpec(name="config", loader=None)
    stub.BDL_API_KEY = ""
    stub.RUN_WNBA = False
    sys.modules["config"] = stub

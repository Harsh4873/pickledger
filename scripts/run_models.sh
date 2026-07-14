#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
  set -a
  . ".env"
  set +a
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
  elif [ -x "/opt/homebrew/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: python3 not found"
  exit 1
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import json

from pickgrader_server import run_daily_model_caches_to_firestore

result = run_daily_model_caches_to_firestore()
print(json.dumps(result, indent=2, sort_keys=True))
if not result.get("ok"):
    raise SystemExit(1)
PY

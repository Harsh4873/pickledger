#!/bin/zsh
# Mac convenience wrapper — delegates to the portable publisher.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
exec "${SCRIPT_DIR}/forebet_publish.sh" "$@"

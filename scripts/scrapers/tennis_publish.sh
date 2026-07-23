#!/usr/bin/env bash
# Publish Tennis feeds (Scores24 + TennisTonic) from a non-GitHub-Actions IP.
#
# Kept deliberately separate from scores24_publish.sh / forebet_publish.sh: those
# publishers refuse the WHOLE commit if any one feed trips their strict
# full-slate gate, and the tennis slate is large with intentionally partial
# prediction coverage. Isolating tennis here means a tennis miss can never wedge
# the Scores24 MLB/WNBA or Forebet publish. TennisTonic is plain-HTTP (and also
# runs on Actions); Scores24 tennis is Cloudflare-challenged on datacenter IPs,
# so it rides the same Camoufox path as the other Scores24 feeds — a zero-pick
# Scores24 tennis bucket is healthy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "python3 not found" >&2
  exit 1
fi

GH_BIN="$(command -v gh || true)"
if [[ -z "${GH_BIN}" ]]; then
  echo "gh CLI not found on PATH" >&2
  exit 1
fi

DATE_ISO="${TENNIS_DATE:-$(TZ=America/Chicago date +%F)}"
PUBLISH_FEEDS="${TENNIS_PUBLISH_FEEDS:-tennistonic_tennis,scores24_tennis}"
FEED_COOLDOWN="${TENNIS_PUBLISH_FEED_COOLDOWN_SECONDS:-30}"
REQUEST_INTERVAL="${SCORES24_REQUEST_INTERVAL_SECONDS:-12}"
REQUEST_ATTEMPTS="${SCORES24_REQUEST_ATTEMPTS:-1}"
BLOCK_RETRY_ROUNDS="${SCORES24_BLOCK_RETRY_ROUNDS:-2}"
HOST_BLOCK_COOLDOWN="${SCORES24_HOST_BLOCK_COOLDOWN_SECONDS:-90}"
# Same-day resume state shared with the other Scores24 publishers so a cleared
# challenge covers tennis reruns too.
SCORES24_STATE_ROOT="${SCORES24_STATE_ROOT:-${HOME}/.cache/pickledger-scores24}"
export SCORES24_CHECKPOINT_DIR="${SCORES24_CHECKPOINT_DIR:-${SCORES24_STATE_ROOT}}"
export SCORES24_CAMOUFOX_PROFILE_DIR="${SCORES24_CAMOUFOX_PROFILE_DIR:-${SCORES24_STATE_ROOT}/camoufox-profile}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)
      if [[ $# -lt 2 ]]; then
        echo "--date requires a YYYY-MM-DD value" >&2
        exit 2
      fi
      DATE_ISO="$2"
      shift 2
      ;;
    --date=*)
      DATE_ISO="${1#--date=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--date YYYY-MM-DD]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--date YYYY-MM-DD]" >&2
      exit 2
      ;;
  esac
done
if [[ ! "${DATE_ISO}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Tennis publish date must be YYYY-MM-DD; got ${DATE_ISO}" >&2
  exit 2
fi

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pickledger-tennis.XXXXXX")"
TEMP_REPO="${TEMP_ROOT}/repo"
GENERATED_CACHE="${TEMP_ROOT}/tennis-latest.json"

cleanup() {
  rm -rf "${TEMP_ROOT}"
}
trap cleanup EXIT

REMOTE_URL="$(git -C "${REPO_ROOT}" remote get-url origin)"
GIT_NAME="$(git -C "${REPO_ROOT}" config user.name)"
GIT_EMAIL="$(git -C "${REPO_ROOT}" config user.email)"

git clone --quiet --depth 1 "${REMOTE_URL}" "${TEMP_REPO}"
git -C "${TEMP_REPO}" config user.name "${GIT_NAME}"
git -C "${TEMP_REPO}" config user.email "${GIT_EMAIL}"

# Warm a Camoufox session so the Scores24 tennis feed can clear a challenge once
# and reuse it. Harmless (and skipped) when Camoufox is unavailable.
"${PYTHON_BIN}" - <<'PY'
import os
os.environ.setdefault("SCORES24_CAMOUFOX_FALLBACK", "true")
if os.environ.get("SCORES24_CAMOUFOX_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}:
    try:
        from camoufox.sync_api import Camoufox

        with Camoufox(headless=True, humanize=True) as browser:
            page = browser.new_page()
            page.goto("about:blank", timeout=15000)
            page.close()
        print("Tennis Camoufox warmup complete.")
    except Exception as exc:
        print(f"Tennis Camoufox warmup skipped: {exc}")
PY

IFS=',' read -r -a FEED_KEYS <<< "${PUBLISH_FEEDS}"
feed_index=0
for raw_feed_key in "${FEED_KEYS[@]}"; do
  feed_key="$(printf '%s' "${raw_feed_key}" | tr -d '[:space:]')"
  if [[ -z "${feed_key}" ]]; then
    continue
  fi
  if [[ "${feed_index}" -gt 0 ]]; then
    sleep "${FEED_COOLDOWN}"
  fi
  echo "Refreshing ${feed_key} for ${DATE_ISO}."
  SCORES24_BROWSER_FALLBACK=true \
  SCORES24_CAMOUFOX_FALLBACK=true \
  SCORES24_REQUEST_INTERVAL_SECONDS="${REQUEST_INTERVAL}" \
  SCORES24_REQUEST_ATTEMPTS="${REQUEST_ATTEMPTS}" \
  SCORES24_BLOCK_RETRY_ROUNDS="${BLOCK_RETRY_ROUNDS}" \
  SCORES24_HOST_BLOCK_COOLDOWN_SECONDS="${HOST_BLOCK_COOLDOWN}" \
  "${PYTHON_BIN}" "${TEMP_REPO}/scripts/refresh_external_feeds.py" \
    --date "${DATE_ISO}" \
    --feeds "${feed_key}" \
    --sports "tennis" \
    --skip-firestore || echo "Tennis feed ${feed_key} refresh returned non-zero; continuing (soft-launch)."
  feed_index=$((feed_index + 1))
done

if [[ "${feed_index}" -eq 0 ]]; then
  echo "No Tennis feeds selected for publish." >&2
  exit 2
fi

TENNIS_CACHE_FILE="${TEMP_REPO}/data/model_cache/${DATE_ISO}.json"
if [[ ! -f "${TENNIS_CACHE_FILE}" ]]; then
  TENNIS_CACHE_FILE="${TEMP_REPO}/data/model_cache/latest.json"
fi

# Lenient gate: tennis is best-effort/soft-launch, so partial coverage never
# fails the publish. Only require that at least one requested feed produced an
# ok bucket dated for today; otherwise there is simply nothing to publish.
if ! DATE_ISO="${DATE_ISO}" PUBLISH_FEEDS="${PUBLISH_FEEDS}" "${PYTHON_BIN}" - "${TENNIS_CACHE_FILE}" <<'PY'
import json
import os
import sys
from pathlib import Path

date_iso = os.environ["DATE_ISO"]
required = tuple(
    feed.strip()
    for feed in os.environ.get("PUBLISH_FEEDS", "").split(",")
    if feed.strip()
)
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
external = payload.get("external_feeds") if isinstance(payload.get("external_feeds"), dict) else {}
ok_feeds = []
for key in required:
    bucket = models.get(key) if isinstance(models.get(key), dict) else external.get(key)
    if not isinstance(bucket, dict):
        continue
    meta = bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}
    bucket_date = str(bucket.get("date") or meta.get("date") or "").strip()
    if bucket.get("ok") is True and bucket_date == date_iso:
        ok_feeds.append(key)
print("ok tennis feeds:", ",".join(ok_feeds) or "(none)")
sys.exit(0 if ok_feeds else 1)
PY
then
  echo "No ok Tennis feed for ${DATE_ISO}; nothing to publish."
  exit 0
fi

cp "${TENNIS_CACHE_FILE}" "${GENERATED_CACHE}"

for attempt in 1 2 3; do
  git -C "${TEMP_REPO}" fetch --quiet origin main
  git -C "${TEMP_REPO}" reset --hard --quiet origin/main
  MERGE_RESULT="$(
    cd "${TEMP_REPO}"
    "${PYTHON_BIN}" scripts/merge_external_feed_cache_payload.py "${GENERATED_CACHE}"
  )"
  echo "${MERGE_RESULT}"
  DEPLOYABLE="$("${PYTHON_BIN}" -c 'import json,sys; print(str(json.load(sys.stdin)["latestUpdated"]).lower())' <<< "${MERGE_RESULT}")"
  git -C "${TEMP_REPO}" add data/model_cache
  if git -C "${TEMP_REPO}" diff --cached --quiet; then
    echo "Tennis cache already current for ${DATE_ISO}."
    exit 0
  fi
  if [[ "${DEPLOYABLE}" == "true" ]]; then
    (
      cd "${TEMP_REPO}"
      "${PYTHON_BIN}" scripts/build_parlay_cards.py --date "${DATE_ISO}"
    )
    git -C "${TEMP_REPO}" add data/parlay_cards
  fi
  git -C "${TEMP_REPO}" commit -m "chore(feeds): refresh Tennis feeds for ${DATE_ISO}"
  if git -C "${TEMP_REPO}" push origin HEAD:main; then
    if [[ "${DEPLOYABLE}" == "true" ]]; then
      "${GH_BIN}" workflow run deploy-pages.yml --repo Harsh4873/pickledger --ref main
    else
      echo "Skipped Pages deploy until the full ${DATE_ISO} team-model cache is available."
    fi
    echo "Published Tennis feeds for ${DATE_ISO}."
    exit 0
  fi
  echo "Tennis push attempt ${attempt} failed; retrying from latest main."
done

echo "Unable to publish Tennis feeds after three attempts." >&2
exit 1

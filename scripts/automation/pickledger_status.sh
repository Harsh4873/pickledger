#!/usr/bin/env bash
# One-line answer to "is the site actually up to date, or is something wrong?"
#
# Checks the published state rather than the automation's own opinion of itself,
# because the failure mode that matters is silent: if cron never fires, a job
# that reports on its own runs says nothing at all. So the primary signal is the
# committed data's date compared with today, which is wrong whenever the slate
# is stale no matter which link in the chain broke.
set -uo pipefail

REPO="${PICKLEDGER_CRON_REPO:-$HOME/pickledger-cron/repo}"
GH_REPO="${PICKLEDGER_GH_REPO:-Harsh4873/pickledger}"
STATUS_FILE="${PICKLEDGER_CRON_STATUS:-$HOME/pickledger-cron/last_run.json}"
LOG_DIR="${PICKLEDGER_CRON_LOGS:-$HOME/pickledger-cron/logs}"
TODAY="$(TZ=America/Chicago date +%F)"
VERBOSE=false
[[ "${1:-}" == "-v" || "${1:-}" == "--verbose" ]] && VERBOSE=true

problems=()
details=()

# The daily job holds this lock for its whole run, so a held lock is the only
# reliable way to tell "still working" from "never started" — the status file is
# not written until the run ends.
run_in_progress() {
  local lock="${PICKLEDGER_CRON_LOCK:-$HOME/pickledger-cron/.lock}"
  [[ -e "$lock" ]] || return 1
  # flock succeeds => nobody holds it => no run in progress.
  ! flock -n 9 2>/dev/null 9<"$lock"
}

# --- 1. is the committed slate current? -----------------------------------
cache_date=""
if git -C "$REPO" fetch --quiet origin main 2>/dev/null; then
  cache_date="$(git -C "$REPO" show origin/main:data/model_cache/latest.json 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin).get("date",""))' 2>/dev/null || echo "")"
  props_date="$(git -C "$REPO" show origin/main:data/player_props_cache/latest.json 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin).get("date",""))' 2>/dev/null || echo "")"
  head_sha="$(git -C "$REPO" rev-parse --short origin/main 2>/dev/null || echo '?')"
  details+=("slate ${cache_date:-unknown} | props ${props_date:-unknown} | HEAD ${head_sha}")
  # Data dated ahead of today is fine — a late-evening refresh legitimately
  # generates the next Central day. Only behind is a problem.
  [[ -n "$cache_date" && "$cache_date" < "$TODAY" ]] && problems+=("site slate is ${cache_date}, expected ${TODAY}")
  [[ -n "$props_date" && "$props_date" < "$TODAY" ]] && problems+=("props are ${props_date}, expected ${TODAY}")
  [[ -z "$cache_date" ]] && problems+=("could not read the published slate")
else
  problems+=("cannot reach GitHub (network, or gh/git auth expired)")
fi

# --- 2. is the LIVE site serving today's slate? ----------------------------
# Deliberately not "does the newest commit have a deploy": auto-grade commits
# land all day and each is briefly undeployed, so that phrasing cries wolf.
# What matters is the content actually published, so this finds the most recent
# deploy whose `deploy` job really executed — a green run is not a deploy,
# readiness can defer and leave the job skipped — and reads the slate date out
# of the commit that deploy shipped.
if command -v gh >/dev/null 2>&1; then
  live_date=""; live_run=""
  while read -r id sha; do
    [[ -z "$id" ]] && continue
    [[ "$(gh run view "$id" --repo "$GH_REPO" --json jobs \
          --jq '[.jobs[]|select(.name=="deploy")|.conclusion]|join(",")' 2>/dev/null)" == "success" ]] || continue
    live_run="$id"
    live_date="$(git -C "$REPO" show "${sha}:data/model_cache/latest.json" 2>/dev/null |
      python3 -c 'import json,sys; print(json.load(sys.stdin).get("date",""))' 2>/dev/null || echo "")"
    break
  done < <(gh run list --repo "$GH_REPO" --workflow deploy-pages.yml --limit 15 \
            --json databaseId,headSha --jq '.[] | "\(.databaseId) \(.headSha)"' 2>/dev/null)

  if [[ -z "$live_run" ]]; then
    problems+=("no deploy has executed recently — the site may be serving old output")
  elif [[ -z "$live_date" ]]; then
    details+=("live deploy run ${live_run} (slate date unreadable)")
  else
    details+=("live slate ${live_date} (run ${live_run})")
    [[ "$live_date" < "$TODAY" ]] && problems+=("the LIVE site is serving ${live_date}, not ${TODAY}")
  fi
fi

# --- 3. what did the last automated run do? -------------------------------
if [[ -f "$STATUS_FILE" ]]; then
  run_date="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("date",""))' "$STATUS_FILE" 2>/dev/null)"
  run_state="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("state",""))' "$STATUS_FILE" 2>/dev/null)"
  details+=("last refresh ${run_date} ${run_state}")
  [[ "$run_state" != "HEALTHY" && "$run_date" == "$TODAY" ]] &&
    problems+=("last refresh ended ${run_state} (${LOG_DIR}/${run_date}.log)")
  # Silence is the dangerous case: no run recorded for today means cron did not
  # fire, which no self-reporting job can tell you about. But "in progress" and
  # "never started" look identical from the status file alone — the outcome is
  # only written at the end — so ask the lock before calling it missed. The full
  # refresh takes about an hour, which would otherwise mean a false alarm every
  # single morning.
  if [[ "$run_date" != "$TODAY" ]]; then
    if run_in_progress; then
      details+=("a refresh is running right now")
    else
      problems+=("no automated run recorded today (cron may not have fired)")
    fi
  fi
elif run_in_progress; then
  details+=("a refresh is running right now")
else
  details+=("no automated run recorded yet")
fi

# --- verdict ---------------------------------------------------------------
if [[ ${#problems[@]} -eq 0 ]]; then
  echo "OK  — ${details[*]}"
  exit 0
fi
echo "PROBLEM — ${#problems[@]} issue(s):"
for p in "${problems[@]}"; do echo "  - $p"; done
$VERBOSE && { echo "context:"; for d in "${details[@]}"; do echo "  $d"; done; echo "log: ${LOG_DIR}/${TODAY}.log"; }
$VERBOSE || echo "  (run 'pl-status -v' for context, or /refresh in Claude Code to fix)"
exit 1

# Model refresh recovery

`model-cache-refresh.yml` has **no schedule trigger**. It accepts
`workflow_call` from Daily Refresh and `workflow_dispatch` from recovery or
manual routines. Daily Refresh owns the nominal 06:30 and 13:00
America/Chicago slots and calls models, player props, and external feeds in
sequence before requesting Pages deployment. Individual writers remain
manually dispatchable.

GitHub cron can arrive late or be dropped. Daily Refresh and the freshness
guard share that scheduler; more cron entries do not supply an independent
clock. The scheduled times are desired refresh windows, not promised run times.

The guard now checks `generatedAt` against the latest scheduled model window:
06:30 and 13:00 America/Chicago. The Daily Refresh coordinator uses the same
timezone-aware schedule, keeping both local times fixed through daylight saving.
External feed updates to `updatedAt` do not count as a model refresh. Core models
must also satisfy the guard's health/date policy for the current Central day;
the existing NBA transport-outage exception reports `degraded` without repeated
recovery. A same-day warmup before the current window is insufficient.

The lightweight guard runs on its cron, after other data workflows complete,
and when requested by the local Scores24 publisher. The macOS backup clock
requests that same guard every 15 minutes between 06:30 and 19:00 Central.
It requires the Mac to be awake, online, and logged in with an authenticated
`gh`; after sleep, its next execution checks the current window. GitHub Actions
still runs the models, so an Actions outage can delay recovery.

Install or update the local clock from the maintained checkout:

```sh
python3 scripts/automation/install_model_refresh_guard.py
```

The generated launch agent and local paths stay outside the public repository.
Logs are in `~/Library/Logs/PickLedger/`. The installer is safe to rerun.
Remove it with `launchctl bootout gui/$(id -u)/bet.harsh.pickledger.model-refresh-guard`
and delete its plist from `~/Library/LaunchAgents/`.

Recovery requested with `--remote` is serialized through
`model-cache-freshness-guard`; direct writer/coordinator dispatches do not pass
through that guard. A queued or running model refresh or Daily Refresh
coordinator prevents another recovery dispatch. Failed or cancelled attempts
have a 20-minute cooldown and at most three manual/recovery attempts per window.
Exhausting recovery fails the guard visibly. The next window resets the budget.
The guard recovers player props only when models are fresh, degraded under the
existing NBA exception, or waiting out a retry cooldown. It does not queue props
behind a pending model run in the shared `pick-cache-writer` group, where a
second pending writer replaces the first. A model outage therefore need not
prevent props recovery during cooldown.
If the current window ran but a core model failed, recovery reruns just the
failed models. A missed window requests the full Daily Refresh coordinator,
including props, CFB/NFL external feeds, and deployment. A healthy same-day
morning cache does not satisfy the afternoon window.

Scores24, tennis, and CFB share an ESPN scoreboard client with three bounded
attempts for temporary failures and invalid JSON. It prefers HTTPS and switches
between ESPN's existing public HTTP/HTTPS endpoints after a 403, since their
edges can behave differently on local and hosted runners. No credentials are
sent with scoreboard requests. Only a valid `events` list can
confirm an off-day; tennis requires responses from both ATP and WTA. If ESPN
remains unavailable, existing cache fallback and optional-feed failure handling
still apply. This avoids publishing a network error as an empty sports slate.

Verification uses the source smoke tests, frontend tests/build, data upcheck,
and Actions job results. A successful Pages workflow is only a deployment when
the `deploy` job itself ran successfully.

## Schedule investigation, 2026-10-09

The operator's offline `gh` evidence distinguishes an obsolete expectation
from a real scheduler problem:

- The model writer has had no cron since the available history's squashed root
  on September 15. Its last roughly 40 standalone runs are all dispatches.
  The external `SCHEDULED_TASKS.md` inventory's 12:45, 14:05, 15:30, and 20:30 UTC
  model slots do not exist in this checkout. That inventory is outside this
  worktree and still needs correction by its owner.
- Daily Refresh's 06:30 Central slot (11:30 UTC during daylight time) arrived
  3.5–8 hours late; its 13:00 slot arrived 3–6 hours late. The morning slot had
  not appeared by 16:15 UTC on October 9. Frozen Staking Status was 4–8 hours
  late and weekly Calibration Refresh was 6–8.5 hours late.
- High-frequency schedules largely failed to appear: Auto-Grade produced only
  3–7 schedule events/day out of 96 nominal slots, and the freshness guard only
  4–6 out of 50. Some guard deliveries arrived outside its configured hours.
  A missing event may still arrive late, but the multi-day deficit cannot be
  explained as a reliable queue of all nominal slots.
- For the observed Daily Refresh runs, `created_at == run_started_at`;
  the hours were lost before run creation, not waiting on concurrency. Few
  model cancellations were observed, all dispatches. Daily commits rule out
  public-repository inactivity disabling as the explanation. The evidence
  establishes scheduler delay/loss, not GitHub's internal reason for it.
- On October 9, the freshness guard dispatched a successful model refresh at
  13:55 UTC. The local watchdog's 14:56 UTC kick was redundant: its test for a
  run "after 14:00 UTC" rejected an already satisfied morning window.

| Candidate | Assessment |
| --- | --- |
| Move to odd minutes | No demonstrated remedy here. Off-hour and odd-minute schedules already suffer the same hours of delay. Keep the Central windows and the guard's matching `SLOTS` unchanged. |
| Add backup cron slots | Still the same clock. Existing frequent schedules mostly fail to appear; extra slots can also deliver redundant work hours later. No evidence of timely recovery from this change. |
| Add a GitHub cron watchdog | Already present as Model Cache Freshness Guard. Its cron is delayed too, so another cron watchdog cannot meet the morning deadline. Completion events help reconcile work that ran; local dispatch supplies the independent clock for work that never started. |
| Change concurrency | Does not repair event delivery. Keep `pick-cache-writer` and `cancel-in-progress: false` to protect running writers. There is only one pending slot, so a newer pending writer can still replace an older one. Sequential coordination and duplicate checks address that separate risk. The `pages` group only cancels deployments. |
| Skip late scheduled refreshes | Potentially useful, but deferred. A model-only check cannot justify skipping the full coordinator's props/feeds/deploy. Selective model refreshes also advance top-level `generatedAt` while preserving other buckets, so that timestamp alone cannot prove every model was refreshed in the window. A safe skip needs sufficient publication/coverage evidence and must retain explicit manual refreshes. |

## Off-GitHub watchdog contract

Keep an independent, awake, authenticated host checking every 15 minutes during
06:30–19:00 America/Chicago, every day (the existing macOS installer implements
that window). The first check may dispatch immediately if the scheduled run has
not appeared. Use the existing remote entry point for a model/props check:

```sh
python3 scripts/automation/ensure_model_refresh.py --remote --dispatch
```

This explicitly dispatches the hosted guard, bypassing GitHub's cron delivery.
The guard syncs current `main`, checks cache freshness and both the standalone
model and Daily Refresh run lists, and applies the existing retry limits.
Do not use a stale local checkout's cache as the freshness signal. Add
`--local-clock --external-feeds` only on a host configured for the local feed
publishers, as the macOS installer does. Neither clock configuration nor any
remote workflow was changed as part of the offline October 9 investigation.

The existing 09:20 Central weekday watchdog should perform this same
reconciliation instead of blindly dispatching the model writer. Its criterion
is today's Central cache date, a valid `generatedAt` at or after the latest due
06:30/13:00 Central slot and no later than now, plus required model health/date
checks. Morning's boundary is 11:30 UTC during daylight time and 12:30 UTC in
standard time; 14:00 UTC has no role. A successful run title, recent `updatedAt`,
or same-day cache date alone is insufficient. Accept the existing documented
NBA degraded state without changing the model or repeatedly retrying it.

If a model/coordinator run is active, wait and recheck it. If data is missing or
stale, the guard dispatches Daily Refresh; if this window is current but a core
model failed, it dispatches only the failed model subset. Respect the 20-minute
cooldown and three dispatch attempts per window. Record the guard/recovery run
IDs, check the resulting committed cache, and surface exhaustion or an active
run stuck beyond its job timeouts instead of piling on more dispatches.

Separately verify that Pages' actual `deploy` job succeeded and its deployed
revision includes the current cache. If only deployment is missing, recover
deployment rather than rerunning models. This clock fixes the scheduling
dependency, not an Actions execution outage. Keep the 09:20 check until another
independent clock with this contract is confirmed operational; it can then be
an alert/check rather than a second unconditional model dispatch.

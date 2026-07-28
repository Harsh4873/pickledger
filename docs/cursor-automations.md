# Cursor Automations for PickLedger

Use **two scheduled cloud automations** on repo `Harsh4873/pickledger` / branch `main`. Enable **GitHub** tool access and ensure `gh` is authenticated in the cloud environment.

Delete or replace draft automations named `Harsh's Automation` if they have zero runs.

For Codex upkeep in both tracks: never open the deployed website or a browser, run source/tests/upcheck checks only, verify NBA Summer League appears as the `nba_summer` in-house cache bucket during Summer League dates, and verify Player Props rankings stay split by the eight published model buckets: Season, All Time, Hot (L10), and Matchup (H2H) for both MLB and WNBA.

## 1. Scores24 publish (required — GitHub Actions cannot scrape Scores24)

**Schedule (UTC cron):** `30 14 * * *` and `30 20 * * *` (~9:30 AM and 3:30 PM America/Chicago during CDT).

**Instructions:**

```
Run scripts/scrapers/scores24_publish.sh from the PickLedger repo root.
Use `scripts/scrapers/scores24_publish.sh --date YYYY-MM-DD` when backfilling a missed slate.

Never open the deployed website or a browser to verify output.

Run the Codex upkeep guard above before the final summary.

After the script finishes, report:
- exit code
- whether a commit was pushed
- Scores24WNBA, Scores24MLB, and Scores24FIFAWorldCup pick counts for today (America/Chicago)
- any scrape or push errors

If Scores24 blocks the cloud IP, say so clearly in the run summary. Do not add AI co-author lines to commits.
```

## 1b. Forebet publish (required — GitHub Actions get Cloudflare-challenged)

**Schedule (UTC cron):** `40 14 * * *` and `40 20 * * *` (~9:40 AM and 3:40 PM America/Chicago during CDT — shortly after Scores24).

**Instructions:**

```
Run scripts/scrapers/forebet_publish.sh from the PickLedger repo root.
Use `scripts/scrapers/forebet_publish.sh --date YYYY-MM-DD` when backfilling a missed slate.

Never open the deployed website or a browser to verify output.

Run the Codex upkeep guard above before the final summary.

After the script finishes, report:
- exit code
- whether a commit was pushed
- ForebetMLB, ForebetWNBA, and ForebetMLS pick counts for today (America/Chicago) with officialMatchups vs matchedPicks
- any scrape, Cloudflare, or push errors

If Forebet Cloudflare-challenges the cloud IP, say so clearly in the run summary. Do not add AI co-author lines to commits.
```

## 1c. Tennis publish (soft-launched — Scores24 tennis needs a non-Actions IP)

**Schedule (UTC cron):** `45 14 * * *` and `45 20 * * *` (~9:45 AM and 3:45 PM America/Chicago during CDT — shortly after Forebet).

**Instructions:**

```
Run scripts/scrapers/tennis_publish.sh from the PickLedger repo root.
Use `scripts/scrapers/tennis_publish.sh --date YYYY-MM-DD` when backfilling a missed slate.

Never open the deployed website or a browser to verify output.

Run the Codex upkeep guard above before the final summary.

After the script finishes, report:
- exit code
- whether a commit was pushed
- TennisTonic and Scores24Tennis pick counts for today (America/Chicago) with officialMatchups vs matchedPicks
- any scrape, Cloudflare, or push errors

Tennis is soft-launched and best-effort: a large singles slate with only partial
prediction coverage is healthy, and a zero-pick Scores24Tennis bucket is normal
(Scores24 Cloudflare-challenges the cloud IP). TennisTonic (plain HTTP) also runs
on Actions via external-feed-refresh, so this publisher's main job is the
Scores24 tennis odds and a local TennisTonic fallback. Do not add AI co-author
lines to commits.
```

## 2. Production health check (optional daily sanity)

**Schedule (UTC cron):** `0 21 * * *` (~4:00 PM America/Chicago during CDT).

**Instructions:**

```
Production upcheck for PickLedger. Never open the deployed site or a browser.

Sync main, run npm run upcheck, and python3 -m pytest tests/smoke/test_player_props.py tests/smoke/test_grader_dry_run.py tests/smoke/test_static_viewer.py -q.

Run the Codex upkeep guard above and confirm Player Props rankings are model-bucketed with applicable-sport records, not duplicated whole-slate consensus records.

Inspect latest GitHub Actions runs for model-cache-refresh, player-props-refresh, external-feed-refresh, auto-grade, and deploy-pages. The model-cache refresh should include `nba_summer` alongside NBA, NBA Playoffs, WNBA, MLB, and FIFA.

If today's model cache or player-props cache is missing or unhealthy, dispatch the matching workflow with gh and wait.

If code fixes are required: test, commit without AI/co-author taglines, push as the logged-in GitHub user, and dispatch deploy-pages.yml.

Summarize health, bucket counts, workflow status, and any blockers.
```

---

## Local daily refresh (cron, 06:53 America/Chicago)

`scripts/automation/daily_local_refresh.sh`, installed in the owner's crontab on
the machine that holds the publisher credentials.

**Why it is not a GitHub `schedule:` trigger.** Two independent reasons:

1. **Scrapers cannot run in Actions.** Scores24 and Forebet Cloudflare-403 every
   GitHub-hosted runner IP, which is why the runbook makes them local-only. No
   cloud-hosted trigger can publish them — the host has to have a
   non-datacenter IP and Camoufox.
2. **`schedule:` is unreliable here.** GitHub treats cron as best-effort. Over
   2026-07-25…28 the writers failed to fire on the 25th and the 28th, and the
   27th's was cancelled on the shared `pick-cache-writer` group. Each time the
   site silently served the previous day, because deploy readiness correctly
   refuses to promote stale data. The scheduled workflows are still in place;
   this job is the belt to their braces.

**Order.** Team models and props are dispatched first because they create the
day's cache entry; only then can an external-feed publish promote `latest.json`
(see `merge_external_feed_cache_payload.REQUIRED_TEAM_MODEL_KEYS`), so each
publisher afterwards can ship its own feeds live instead of stranding them in
the dated file. Then external feeds, then a deploy whose `deploy` **job** is
confirmed to have executed — a green run is not a deploy.

**Operational notes.**
- Requires the machine to be powered on and networked at 06:53, and `gh` to stay
  authenticated. It exits non-zero on anything but a verified deploy.
- `flock` prevents overlapping runs.
- A failing publisher degrades that feed only; the run continues, because the
  team models and the deploy still matter.
- Logs: `~/pickledger-cron/logs/<date>.log`, plus `cron.log` for scheduler-level
  output. Check these first when the morning slate looks wrong.
- It automates the mechanical half of `/refresh` only. It cannot diagnose a
  novel defect — when it reports anything but `HEALTHY`, run `/refresh` and read
  the log.
- Dry run: `daily_local_refresh.sh --dry-run` (verifies wiring, takes no action).

### Knowing whether it worked

`scripts/automation/pickledger_status.sh`, symlinked to `~/bin/pl-status`:

```
pl-status        # one line: OK, or the specific problems
pl-status -v     # adds context and the log path
```

It checks the **published** state, not the automation's own opinion of itself —
because the dangerous failure is silent. If cron never fires, a job that reports
on its own runs says nothing at all, and the site just keeps serving yesterday
while looking completely normal. So the checks are:

1. Is the committed slate (and props) dated today? Data dated *ahead* of today is
   healthy — a late-evening refresh legitimately generates the next Central day —
   so only "behind" counts.
2. Is the **live** site serving today? This resolves the most recent deploy whose
   `deploy` job actually executed and reads the slate date out of the commit that
   deploy shipped. Deliberately not "does the newest commit have a deploy":
   auto-grade commits land all day and each is briefly undeployed, which would
   cry wolf constantly.
3. Did an automated run record itself today, and how did it end? No record at all
   means cron did not fire — the one failure a self-reporting job can never
   surface.

Exit code is 0 for OK and 1 for any problem, so it can be used in a check.

**Optional push alert.** Set `PICKLEDGER_ALERT_WEBHOOK` to a Discord-compatible
webhook and the daily job will message it on any non-HEALTHY outcome. Without it
the run is silent by design and `pl-status` is the only way to notice.

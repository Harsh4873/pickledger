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

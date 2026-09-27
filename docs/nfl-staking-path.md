# NFL in-house staking path

Status as of 2026-09-27: **no staking approval**. `data/calibration/staking_approvals.json` stays `{ "approvals": [] }` until the unused holdout actually clears.

The team model still mints BET/LEAN from the frozen totals rule. The publication gate demotes those rows to visible PASS @ 0u and keeps `shadow_decision` / `shadow_units` for research grading.

## Why NFL cannot clear a real approval yet

1. **Sample size.** The frozen rule (Under residual >= 1.0) fires on a small slice of each slate. Two live Sundays produced on the order of eight BET/LEAN totals, not 100 independently priced settled rows.
2. **Mined historical bands.** `NFLPredictionModel/artifacts/metadata.json` searched 2012-2025 closing prices to pick the Under 1.5 BET / Under 1.0 LEAN ladder. That walk-forward is selection, not an unused holdout.
3. **Late Sunday publication.** Daily refresh starts at 6:30 AM and 1:00 PM Chicago. NFL forecasts are generated early, then the trusted clock was rewritten at write time after MLB finished (often 20:42 UTC). 1pm ET kickoffs are 17:00 UTC, so those rows failed `published_at_not_before_game_start` or never entered the ledger. PASS capture for forecast audit only started around 2026-09-21, so most Sunday research rows are missing.
4. **Version churn.** Ledger `model_version` preferred `prediction_model_version` (`nfl:` + a hash of NFL code plus `pickgrader_server.py`). Unrelated server edits split the scorecard. Approvals now key on the fitted label `nfl_v1_epa_elo_market_anchored`.
5. **Quote timestamps.** nflverse posted odds were marked `market_priced` so the live DraftKings overlay skipped them. Financial eligibility requires an observed quote clock. Generation now stamps `market_retrieved_at` from the games.csv fetch, and the NFL bucket replaces odds when a pregame ESPN/DK attach still matches the line.
6. **Inspected live dates.** The 2026-09-25 evidence review already looked at certified NFL rows. Dates through 2026-09-26 are not an unused holdout.

## Frozen candidate (do not re-search)

File: `data/calibration/nfl_staking_freeze.json`

- Fitted version: `nfl_v1_epa_elo_market_anchored`
- Market: `totals`
- Rule: Under residual magnitude >= 1.5 is BET 0.5u; Under in [1.0, 1.5) is LEAN 0.25u; juice cap -125
- Holdout starts: `2026-09-27T00:00:00Z` (after freeze). Walk-forward 2012-2025 and live dates through 2026-09-26 remain selection.

Incumbent on that window is unapproved PASS @ 0u. Challenger is the frozen rule on the same games and quotes.

## How an approval is earned

```bash
python scripts/nfl_staking_approval.py --backfill --output /tmp/nfl-staking-status.json
python scripts/nfl_staking_approval.py --write
```

`--write` is a no-op unless the holdout has:

- unused_during_selection
- >= 100 independently priced settled challenger actions (shadow BET/LEAN after the gate, or live BET/LEAN if an approval already exists)
- ROI > 0 and clustered lower 95 > 0
- no material paired Brier regression versus the observed market (needs >= 20 paired samples)

The writer calls the same `_approved()` check the publication gate uses. It will not hand-write a row that fails that check.

Daily refresh already captures NFL snapshots and preserves generation clocks across a slow sibling-model job. Catch up missing historical NFL cache rows with `--backfill`. Leave approvals empty until this script succeeds.

Replay of features as-of tip is the existing NFL serving path (`features_for_date` folds a game's result only after emitting its row). Do not train a new fitted version into this freeze without starting a new unused holdout after that freeze.

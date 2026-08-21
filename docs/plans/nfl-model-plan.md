# NFL Model — build plan (drafted 2026-07-19, launched 2026-08-20)

Goal: a trained NFL model that publishes priced moneyline, spread, and total picks automatically on NFL game days. The committed 2026 schedule begins on 2026-09-09; preseason is intentionally excluded from the model dataset and training contract.

## Data spine (all free, keyless, verified live)

1. **nflverse `games.csv`** (github.com/nflverse/nfldata) — every game 1999-present with final scores, **closing spread and total**, moneylines, rest days, roof/surface, starting QBs, division flags. This is both the training target set and the market baseline. Verified reachable.
2. **nflverse play-by-play parquet** (nflverse-data releases) — EPA per play 1999-present, the standard for team-strength features; weekly auto-updates in season.
3. **ESPN NFL scoreboard** (`football/nfl`) — daily slate + DraftKings ML/spread/total via the existing `market_odds.py` machinery (add `"NFL": ("football","nfl")`), and live scores through the existing auto-grader (add the `SPORT_TO_ESPNSLUG` entry).

## The exact ML training design

- **Training window:** 2002-2025 regular + postseason (~6,400 games). 2020 gets a COVID flag feature, preseason games excluded from training entirely.
- **Features, computed strictly as-of game date (no lookahead):**
  - *Team strength:* offensive/defensive EPA per play and success rate, season-to-date with exponential decay (half-life ~6 games), pass/rush splits, early-down EPA, explosive-play rate.
  - *Situational:* rest-day differential (Thursday/short week/bye), home field, divisional flag, travel/timezone, week number, dome/surface, temperature+wind for totals.
  - *Continuity:* starting-QB-change flag from the games file's QB columns (the single biggest single-player effect in NFL).
  - *Market anchor:* opening/closing spread and total as features — the model learns **residuals over the market**, the same market-anchored philosophy as MLBPredictionModel v2 and parlay engine v6.
- **Three heads (mirroring the repo's proven MLB v2 stack):**
  1. *Moneyline:* `HistGradientBoostingClassifier` → home-win probability → **isotonic calibration** fit on out-of-fold predictions only.
  2. *Spread:* `HistGradientBoostingRegressor` on (actual margin − market spread); cover probability via the residual distribution (empirical σ ≈ 13.2).
  3. *Total:* same residual approach on (actual total − market total), σ ≈ 13.5, weather/pace-weighted.
- **Validation protocol:** strict walk-forward by season — train ≤ season N, test N+1, rolled 2015→2025. Metrics: log loss, Brier, calibration curves, and **simulated flat-bet ROI vs closing lines** (the only honest test). Expectation set honestly: beating closing lines consistently is near-impossible; the promotion bar is calibrated probabilities within ~1-2% Brier of market plus positive ROI on the top-edge-decile picks in validation. Edge thresholds for BET/LEAN are chosen from those validation ROI curves, not invented.
- **Artifacts:** joblib models + metadata JSON (feature contract, train window, per-season validation metrics, version) in `NFLPredictionModel/artifacts/`, trained by a manual-dispatch workflow like `mlb-train.yml`. Serving loads artifacts; no training in the daily cron.

## Launch status

- New bucket `nfl`, source rows split as **NFL ML / NFL Spread / NFL Total** (consistent with every other variant).
- NFL is active in the daily model refresh, the site filter, freshness guard, merge contract, and production upcheck.
- The model emits BET/LEAN decisions only at real posted prices. These rows feed the pregame ledger, grading, records, and site views like the other active team models.
- An empty NFL bucket is valid on a date without a scheduled regular/postseason game. The first game in the committed 2026 schedule is 2026-09-09, so no August rows are expected.

## Registration sweep

`run_nfl_model` in pickgrader_server (+ `SPORT_TO_ESPNSLUG["NFL"]`), refresh jobs + cron default, merge DEPLOYED/ALIAS keys, `market_odds` SPORT_LEAGUES + bucket keys, team_prop_pregame_ledger key, evaluator contract, parlay SOURCE_LABELS, data.ts labels + market split, ESPN live scores, tests, freshness guard, and upcheck required sets.

## Phases

1. **Now+50min (implementation start):** NFLPredictionModel package — nflverse downloader with local cache, feature builder, training + walk-forward backtest scripts, serving path, registration sweep, shadow wiring, smoke tests.
2. **This week:** train v1 artifacts, produce the walk-forward validation report, commit artifacts.
3. **August:** activate the daily pipeline and validate empty off-slate buckets ahead of the regular season.
4. **Sept 9, opening slate:** automatically publish the first priced NFL picks.

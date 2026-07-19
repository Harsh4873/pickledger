# NFL Model — build plan (drafted 2026-07-19, shadow-first)

Goal: a trained NFL model whose walk-forward ledger is warm by Week 1 (2026-09-10), publishing **nothing** to the site until an explicit go-live. Preseason (August) is pipeline rehearsal only.

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

## Shadow mode (the "don't publish" mechanism)

- New bucket `nfl`, source rows split as **NFL ML / NFL Spread / NFL Total** (consistent with every other variant).
- The model emits real BET/LEAN decisions at real DraftKings prices — required, because the pregame ledger and walk-forward qualification only count BET/LEAN rows with observed prices — but the sport joins a frontend `SHADOW_SPORTS` set so **nothing renders anywhere on the site**. Grading and ledger accumulation are server-side and continue regardless of display.
- Preseason rows are graded for pipeline rehearsal but flagged (`season_type: preseason`) and excluded from walk-forward qualification; the warm-up ledger that matters starts Week 1... which is why shadow BET/LEANs begin the moment real September lines post.
- Go-live = remove NFL from `SHADOW_SPORTS` + add `nfl` to the consensus-gate config with MLB-grade thresholds (≥30 walk-forward samples, positive calibrated edge, real prices). By then the record exists and is inspectable before a single pick shows publicly.

## Registration sweep (soft-launch, mechanical)

`run_nfl_model` in pickgrader_server (+ `SPORT_TO_ESPNSLUG["NFL"]`), refresh jobs + cron default, merge DEPLOYED/ALIAS keys, `market_odds` SPORT_LEAGUES + bucket keys, team_prop_pregame_ledger key, evaluator contract, parlay SOURCE_LABELS, data.ts labels + market split + SHADOW_SPORTS, tests. NOT in freshness guard or upcheck required sets (cannot block deploys).

## Phases

1. **Now+50min (implementation start):** NFLPredictionModel package — nflverse downloader with local cache, feature builder, training + walk-forward backtest scripts, serving path, registration sweep, shadow wiring, smoke tests.
2. **This week:** train v1 artifacts, produce the walk-forward validation report, commit artifacts.
3. **August:** preseason shadow rehearsal; weekly nflverse refresh wiring; threshold tuning from validation curves.
4. **Sept 10, Week 1:** go-live review — validation report + shadow record decide publication.

# In-house model audit — 2026-09-19

Scope: the in-house team models that publish to `data/model_cache/`
(`cfb`, `nfl`, `mlb_new`, `mlb_inning`, `mlb_first_five`, `mlb_team_total`,
`mls`, `wnba`, `nba`, `nba_playoffs`). Tennis was out of scope and untouched.

Evidence: the 108 dated cache files (2026-06-04 … 2026-09-19, ~15k rows),
`data/calibration/team_prop_pregame_ledger.json`, `data/profit_desk/`, the
committed training artifacts, and walk-forward backtests re-run on the newest
nflverse / SportsDataverse data. Every ROI below is flat 1u at the recorded
posted price; rows priced at a house-assumed number are called out as such and
never counted as money. Bankroll dollars are not modelled anywhere.

The recurring theme is not "the models need more data". It is that stakes
were being minted where the model has no measured accuracy (moneylines the
market prices better, residual heads with 49-51% direction rates, prices that
no book posted), while the one or two segments where each model *is* accurate
were either gated off or invisible. The fixes therefore move every stake
behind a segment that was validated at posted prices, and make PASS an
explicit, visible research state instead of an empty board.

## Priority 1 — CFB and NFL

### NFL: before

`nfl_v0_games_ewma` (July 2026) had three heads trained on points-scored
EWMAs. Its own walk-forward report, re-run at the recorded closing prices:

| live rule | picks 2012-25 | hit | flat ROI | seasons positive |
| --- | --- | --- | --- | --- |
| spread residual ≥ 2.5 (LEAN/BET) | 1,332 | 49.5% | −3.4% | 6/14 |
| total residual ≥ 2.5 | 1,472 | 51.1% | −0.7% | 7/14 |
| moneyline edge ≥ 2.5 pts (LEAN/BET) | 2,074 | — | −1.2% | 3/14 |

The moneyline head's out-of-fold Brier was 0.2208 against 0.2125 for a
spread-only logistic — worse than the market it was betting against. The
first live slate (2026-09-13) published 6 LEANs, all six lost, and three
publish-integrity bugs compounded it: rows were re-decided at 20:42Z for
17:00Z kickoffs (a refresh after the 1pm games had started), spread/total
rows carried a hard-coded −110 labelled `pricing_type: market`, and the
pooled cross-sport calibrator shifted probabilities ~3-4 points and left
rows on the board as LEAN at 0u with negative edge.

### NFL: after (`nfl_v1_epa_elo_market_anchored`)

- Team state adds margin-of-victory Elo and EWMA EPA/play (offense, defense,
  pass/rush matchups, CPOE) from nflverse weekly team stats, built strictly
  as-of each game; the moneyline is a market-anchored logistic whose
  walk-forward Brier (0.2115) sits at market parity (0.2111 / 0.2113).
- `nfl_train.py` searches a deliberately tiny segment space (direction ×
  residual threshold) at recorded prices and writes `decision_policy` into
  `artifacts/metadata.json`. The bar is ≥100 picks, ≥+3% flat ROI and ≥70%
  of seasons positive; a discovered segment is capped at LEAN.
- The only qualifying segment is **Under with residual ≤ −1.0**: 306 picks,
  60.3%, +17.1%, 12 of 14 seasons positive (robust to alpha 10/50/200 and to
  dropping the line feature; the all-unders baseline is −0.9%). Moneyline and
  spread publish as research PASS with the model's side and probability.
- Serving skips games that have kicked off (the merge keeps pre-kickoff
  rows), prices spread/total rows from the nflverse odds columns or marks them
  unpriced, converts Eastern kickoff to UTC, stamps `calibration_excluded`, and
  reports scheduled / started / unpriced / staked counts in the bucket note.
- Dry run for 2026-09-20: 14 games → 42 rows, 2 LEAN unders, 39 rows priced
  by the DraftKings attach, both staked rows certified in the pregame ledger.

### CFB: before

`cfb_v1_market_free_bivariate` deliberately excluded the market from its
features. Out of fold (2021-25) the originator's disagreement with the
spread-implied win probability was anti-predictive — its side won 34-39%
when it disagreed by 5+ points — and its spread/total direction rates sat at
47-53%. The isotonic spread/total calibrators collapsed to a plateau
(calibrated Brier 0.2495 / 0.2497 ≈ no information), so serving fell back to
raw probabilities and forced PASS: the 2026-09-19 board was 57 PASS plus six
overconfident favourite ML BETs.

### CFB: after (`cfb_v2_market_anchored_totals`)

- The originator stays market-free and keeps producing the research forecast
  and the win-aligned spread card (the TAMU rule is unchanged).
- Market-anchored heads join the artifact: a logistic moneyline (Brier 0.179
  vs 0.190 originator / 0.178 market) and HGB residual heads for spread and
  total. Only these heads may stake, and only through a validated segment.
- Total residual `|r| ≥ 3` qualifies: 689 graded, 55.9% vs the 52.4%
  break-even, 4 of 5 seasons above break-even, and monotone (52.8% at ≥2,
  59.6% at ≥4, 62.0% at ≥6). It publishes at LEAN (0.25u) when both total
  prices are posted and the selected price is no heavier than −125. Moneyline
  and spread publish as PASS research.
- Same-day re-run (17:00Z, 13 started games skipped): 9 games → 27 rows,
  1 LEAN (Over 39.5, residual +3.6), 26 PASS.

## Priority 2 — the other in-house models, ranked

| rank | model | what is broken | class | evidence | shipped fix |
| --- | --- | --- | --- | --- | --- |
| 1 | `mls` moneyline | 8-16-1 / −44.5% flat at posted prices (ledger 37 rows / −42.0%); 33% win rate against a 57.6% break-even; the training backtest never cleared its own −250 cap's break-even | edge | `mls_segments.py`, `ledger_roi.py` | moneyline publishes as PASS research; gate result kept as `model_decision` |
| 2 | `mls` totals/handicaps | gate ignored model-vs-market edge; edge < 0 rows 24-27-1 / −24.6%, edge ≥ 0 totals 53-32-1 / +7.5%, stable both halves | edge gate | `mls_devig.py`, `mls_gates.py` | `_decision` refuses a negative edge |
| 3 | `mlb_inning` | 1,362 rows, 156.5u staked, 0 posted prices; own 53.8% hit rate below the stamped −120's 54.5% break-even | publish | `gate_price_bugs.py` | rows publish as research (PASS, `market_priced: false`, `model_decision` kept) |
| 4 | `mlb_new` totals | market line was the model's output on 1,901/1,901 rows; 100% of September rows were O/U 8.5; real-priced ROI −1.9% → −6.3% once the line froze; 3-5pp band −6.9% vs ≥5pp +13.2% | publish + gate | `provenance_deep.py`, `fix_backtests.py` | lineless totals skipped, runner line kept at an explicit assumed price, LEAN 3pp → 5pp, gate blocks `model_output` lines |
| 5 | shared calibration | fit on publication-selected rows, shrunk toward a pooled prior (team_total group 68% prior); worsened Brier for `mlb_new` and `mlb_team_total`; stale-snapshot read left 16 WNBA + NFL rows LEAN at 0u | calibration | `calibration_provenance.py`, `plumbing.py` | current-row semantics, identity bootstrap for every team model |
| 6 | all team models | rows first published / re-decided after kickoff (98 MLB, 32 WNBA; WNBA post-tip upgrades 17-1); prices captured after kickoff (26% of priced MLS rows) | publish | `late_publish_recent.py`, `plumbing.py` | kickoff freeze in the refresh pipeline; wall-clock price capture |
| 7 | `mls` / `fifa_world_cup` handicaps | soccer per-side handicap line never parsed → 0 of 165 MLS spread rows had provenance, 106 staked rows invisible to the ledger | plumbing | `plumbing.py` | `market_odds` reads `pointSpread.<side>.close.line` |
| 8 | certified ledger | a model-published `market_probability` was accepted as odds provenance (259 MLS + 31 WNBA records; 17 certified at the +100 placeholder) | evidence | `ledger_check.py` | explicit provenance required; placeholder-equal odds rejected |
| 9 | `nba` (October) | confidence-only BET at `odds: null` (the lone live row, 1.13u, lost); invented −110 on unpriced lines; PASS totals at 1u; no `certification_timing` (0 of 5,135 ledger records) | publish | `pickgrader_server.py` review | provisional decision labelled unpriced, assumed price labelled, PASS = 0u, NBA joins the certified/frozen team keys |
| 10 | `mlb_first_five` | 69 of 107 dates all-PASS — correctly. f5_side −49% walk-forward ROI, f5_total raw probability is noise (AUC 0.507); every model-vs-price bucket loses | edge (none) | `ledger_state.py`, `beat_market.py` | none; keep dark. Unreplaced proxy prices are now demoted |
| 11 | `mlb_team_total` | passes validation but was blocked by the borrowed shrink; raw probabilities under-confident (0.5425 vs 0.5550 outcome); only positive slice (Under, 3-4pp) is July-only | calibration | `calibration_fit.py`, `fix_backtests.py` | identity bootstrap lets its own gate decide; no threshold change |
| 12 | `wnba` | clean pregame book since the v2 totals rebuild is 42-35 / +3.8%; v1 totals (21 rows, −63.8%) are already retired | marginal | `wnba_clean.py`, `wnba_current.py` | publish-integrity fixes only (see below) |

Two findings that changed during verification and are recorded so they are
not re-litigated:

- **WNBA claimed-edge ceiling — not shipped.** The audit's −61.5% bucket is
  on the *calibrated* (displayed) edge of 10-15 points (12 rows, 3-9). On the
  model's raw edge the pattern reverses (raw 15-20 points: 11-6, +22.7%), so a
  ceiling inside the WNBA assessors would have removed winning picks while the
  losing bucket is 12 rows. Revisit at n ≥ 30 on the calibrated edge, where
  the effect actually lives (`scripts/pick_calibration.py`).
- **`mlb_new` moneyline needs no loosening.** The 30 rows the gate staked at a
  posted price returned +13.3%, better than every mechanical edge band; the
  market still beats the model outright (Brier 0.2415 vs 0.2461, AUC 0.596 vs
  0.546), so the gate is the selector, not the model.

## Shared pipeline rules added

`scripts/refresh_model_cache.py` now runs, in order: stamp timing →
**freeze started games** → merge → attach posted prices → calibrate →
consensus gate → **demote unpriced stakes**. A refresh after kickoff can no
longer publish or re-decide a live game (the merge keeps the pre-kickoff
rows), and a BET/LEAN whose price is still missing or still equal to the
model's placeholder after the attach becomes PASS at 0u with
`source_decision` preserved.

## Not done, deliberately

- Retraining the MLB v2 artifacts (`training_end_date = 2025-09-20`; the
  whole 2026 season is out of sample). It is warranted, but only after this
  branch lands: the totals model's target is `total − market line` and it
  cannot be trained or served honestly while the served line was a hardcoded
  8.5. Entry point: `MLBPredictionModel/build_historical_dataset.py` then
  `train_model_v2.py`.
- Re-running `MLSPredictionModel/mls_train.py` with prices attached so
  `backtest.json` carries ROI beside hit rate, restricted to the ±0.5
  handicap lines ESPN actually posts. Refitting the same family will not
  close a gap the walk-forward already measured; the gates above are the
  yield.
- Rebalancing `train_pick_calibration.fit_platt` (prior strength 80 at a
  30-row minimum) and admitting certified PASS rows to its training pool.
  The identity bootstrap removes the damage; the refit is a separate change.

## Smoke check after merging

```bash
# regenerate today's in-house buckets into the cache (no Firestore)
python3 scripts/refresh_model_cache.py --models nfl,cfb --skip-firestore
python3 - <<'PY'
import json, collections
d = json.load(open("data/model_cache/latest.json"))
for key in ("nfl", "cfb"):
    b = d["models"].get(key, {})
    print(key, b.get("note"), b.get("coverage"))
    print(collections.Counter((p.get("market"), p.get("decision")) for p in b.get("picks") or []))
PY
python3 -m pytest tests/smoke/test_nfl_model.py tests/smoke/test_cfb_model.py \
  tests/smoke/test_publish_integrity.py tests/smoke/test_pick_calibration.py -q
```

Expected: a non-empty NFL/CFB bucket on any slate day, every staked row
carrying `decision_reason: segment:…` and a posted price, every other row
`PASS` at 0u, and `[kickoff-freeze]` / `[unpriced-demotion]` counters in the
refresh log. Retrain with the manual `NFL Model Training` / `CFB Model
Training` workflows; both write the validated `decision_policy` into the
artifact metadata, so a retrain that finds no qualifying segment publishes
research only.

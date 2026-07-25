# MLS Model (v2)

Trained Dixon-Coles engine behind the site's **MLS ML**, **MLS Spread**, and
**MLS Total** variants. One bivariate score grid per game prices all three
markets; every number the model publishes traces to a walk-forward-validated
artifact committed in `artifacts/`.

## Why v2 — the v1 audit

v1 was the FIFA World Cup roster-power heuristic pointed at `usa.1`. The audit
found, and v2 fixes:

| v1 shortcoming | v2 fix |
| --- | --- |
| Never trained or validated; hand-set constants throughout | Weighted-MLE Dixon-Coles fit on 6,054 MLS matches (2012-present); hyperparameters walk-forward validated, results reported on an untouched 2023+ test walk |
| Player power collapsed to the club table (every MLS player's league is `usa.1`) via hundreds of roster/athlete/club API calls per run | Team ratings come from match results; a slate run makes ~3 HTTP requests |
| Flat home-goal bumps (+0.20 / −0.15) bolted on outside the model | Home advantage estimated in the fit (≈ +0.26 log-goals, consistent with MLS's league-leading home edge) |
| Independent Poisson scores | Dixon-Coles low-score dependence (fitted ρ ≈ −0.04), where draw and handicap pricing lives |
| Integer total lines dropped push mass without renormalizing | Push-aware pricing; published probabilities are push-conditioned like the prices they sit next to |
| No posted total → priced a fabricated 2.5 line | Totals and spreads price posted lines only |
| Ad-hoc feedback loop nudging thresholds off tiny cache samples | Trained calibration (1X2 vector scaling + total-rate affine) fit on out-of-sample walks |
| Venue multipliers from ≤14-day samples | Dropped; nothing ships that validation could not support |

## Data

- **Training + archive:** football-data.co.uk `new/USA.csv` — every MLS match
  since 2012 with Pinnacle/average/best closing 1X2 prices (100% coverage).
  Committed compactly as `artifacts/mls_matches.json.gz`, keyed by ESPN team
  id so training ratings apply directly to the live slate.
- **Serving refresh:** the same workbook re-downloaded at run time (non-strict
  name mapping) plus an ESPN scoreboard backfill for the last few days, then a
  full refit at the artifact hyperparameters — the exact procedure the
  validation walk scored. `meta.ratings_through` reports the effective date.

## Training protocol (`mls_train.py`)

Walk-forward everywhere: every prediction uses only matches strictly before
its date. Warm-up 2012-2018; validation walk 2019-2022 (hyperparameters,
calibrations, blend weight, gates); test walk 2023-2026 (reported, never
tuned). Selected: half-life 365 days, ridge 2e-2.

Held-out (2023+) results, n=1,821:

- 3-way log loss: **1.051** calibrated model vs **1.025** devigged Pinnacle
  close (uniform baseline 1.099). The model is competitive but behind the
  sharp close — as expected, and the design treats it that way.
- Challengers on identical walks: logistic regression 1.058, gradient
  boosting 1.138. The Dixon-Coles incumbent ships because it won, not by
  assumption (`metrics.json: challengers`).
- Score-grid market calibration: P(over 2.5) predicted 0.560 vs empirical
  0.589; P(over 3.5) 0.337 vs 0.355; handicap cover pools within ~2pp of
  stated at every threshold (`metrics.json: grid_markets_test`).

## What the backtest licenses — and what it does not

Flat-stake 1X2 vs closing prices: the best validation edge gate (+16% val ROI)
inverts to **−12.6% on test**. Picks where the model exceeds the devigged
market by >10pp hit 32% at a 34% market bar. Every relative model-vs-market
filter anti-selected model blind spots (`backtest.json: edge_gate_rejected`).

So the model **does not claim to beat the market**, and decisions are
confidence gates, tennis-model style:

- Published probability = calibrated model blended with the devigged live
  market at weight 0.6 (validated: within 0.002 nats of market-only log loss).
- **BET** ≥ 0.60 blended, **LEAN** ≥ 0.55 (moneyline; grid markets 0.58/0.545),
  team sides only, decided tiers only at posted prices no shorter than
  **−250** (a fixed cap — relative EV filters anti-select; extreme juice
  carries no betting information either way).
- Held-out hit rates at the shipped gates: **BET pool 67.8%** (n=90),
  **all decided 63.2%** (n=337), 2023-2026 (`backtest.json:
  confidence_gates.test_table_price_capped`).
- `edge` (model minus devigged market, pp) is published as information and
  never required. Unpriced rows cap at LEAN. Teams under 8 effective decayed
  games PASS.

## Files

- `mls_data.py` — workbook/ESPN parsing, franchise map, archive, dedupe.
- `mls_core.py` — weighted-MLE fit (analytic gradients), score grid,
  push-aware market pricing, calibrations.
- `mls_train.py` — full protocol; regenerates every artifact:
  `python -m MLSPredictionModel.mls_train --csv USA.csv`
- `mls_model.py` — serving: `generate_mls_picks(date_iso)`.
- `artifacts/` — match archive, model config + calibrations + gates,
  `metrics.json`, `backtest.json`.

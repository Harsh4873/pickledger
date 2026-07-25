# Tennis model

In-house ATP/WTA singles match-winner model: surface-aware Elo and Weighted Elo
ratings feeding a calibrated classifier, trained walk-forward on 26 years of
tour results and evaluated against bookmaker closing prices.

## Layout

| File | Role | Dependencies |
| --- | --- | --- |
| `tennis_data.py` | Downloads and normalises the historical archive; builds the match spine | stdlib (`xlrd` only for pre-2013 seasons) |
| `tennis_core.py` | Elo / surface-Elo / WElo engine, as-of feature builder, ratings snapshot | stdlib |
| `tennis_infer.py` | Scores the exported JSON model | stdlib |
| `tennis_model.py` | Daily serving: ESPN slate → picks | stdlib + `requests` (via the shared tennis scraper) |
| `tennis_train.py` | Walk-forward training, tuning, calibration, export | numpy, scikit-learn |
| `tennis_backtest.py` | Betting simulation against closing prices | stdlib |

Only training needs scikit-learn. The trained model is exported to plain JSON —
tree nodes, isotonic knots, blend coefficients — and scored by `tennis_infer.py`
with the standard library alone, verified against scikit-learn at train time to
machine precision. The daily cache refresh therefore cannot be broken by a
library upgrade, which is the failure mode a pickled estimator invites.

## Data

`tennis-data.co.uk`, the archive the Weighted-Elo literature is built on: every
ATP and WTA main-tour singles match with the final scoreline, both players'
rankings and points, and closing prices from up to ten bookmakers.

- ATP 2000–present, WTA 2007–present
- ~118,000 matches, ~3,000 players
- ~82% carry Pinnacle's closing price, ~92% Bet365

Raw workbooks and the derived spine live in `data/tennis/` and are **not**
committed — they are third-party data, regenerated on demand:

```bash
python -m TennisPredictionModel.tennis_data      # download + build the spine
```

Two source defects are corrected on the way in, both caught by the spine's QA
and covered by tests:

- The `2005w`/`2006w` URLs serve the **men's** workbook byte-for-byte. Taken at
  face value they inject 5,818 mislabelled ATP matches into the WTA ratings.
- A handful of rows carry typo'd years (the 2026 Iaşi Open final is stamped
  2029). One such row sorts to the end of the chronological replay and poisons
  the snapshot's `through_date`, which is what the daily job uses to decide how
  much history to catch up on.

## Method

**Ratings.** Three systems run over one chronological pass:

- Elo with the tennis scale factor `K = k_scale / (matches + offset) ** shape`
  (Kovalchik 2016; the shape FiveThirtyEight's tennis model uses).
- Independent per-surface Elo, blended with the overall rating at serve time.
- WElo (Angelini, Candila & De Angelis, *EJOR* 297(1), 2022): the same update
  scaled by the winner's share of games, so a 6-0 6-0 win moves ratings twice as
  far as a 7-6 7-6 win. Symmetric by construction.

Rather than adopt the published constants, `k_scale`, `k_offset`, `k_shape` and
the surface blend are grid-searched on the development seasons. The search
range is deliberately wider than the published values, because the first sweep
selected its own lower corner — a hyper-parameter pinned to a boundary means
the grid chose it, not the data.

**Features.** 38 as-of features: rating gaps (overall, surface, weighted,
blended), rank and points, career and surface experience and win rates, recent
form, rating momentum, head-to-head, rest days, three fatigue windows, games
already played at this event, recent-retirement flags, and interactions that let
a linear model express "rating gaps matter more over five sets".

Every player-dependent feature is a difference, and prediction is explicitly
symmetrised, so `P(a beats b) + P(b beats a) == 1` holds exactly. Models are
also trained on both orientations of every match, which removes any residual
preference for the first slot.

**No leakage.** The replay emits a match's feature row and *only then* folds the
result into both players' state. Walkovers never move ratings; retirements do,
but with a neutral margin weight, and they are excluded from the training set.

**Market independence.** The model never sees a betting price. A model with the
market as an input converges to the market and can only track it; keeping it
market-free makes the comparison against the closing line a real test. The two
are combined afterwards by a logistic forecast combination fitted out-of-sample,
which doubles as an encompassing test.

## Protocol

| Seasons | Role |
| --- | --- |
| 2000–2011 | Burn-in. Replayed to warm the ratings; no rows emitted. |
| 2012–2021 | Development. Walk-forward by season; picks hyper-parameters, model family, calibration and blend weights. |
| 2022–present | Held out. Touched once, at the end, by the same walk-forward loop. |

Walk-forward means each season is scored by a model trained only on strictly
earlier seasons — never a random split, which would let a model train on a
player's future.

## Results

Held-out 2022–2026, 22,912 matches, walk-forward. Full report in
`artifacts/metrics.json`; betting simulation in `artifacts/backtest.json`.

### Forecast accuracy

| Model | Log loss | Brier | Accuracy | ECE |
| --- | --- | --- | --- | --- |
| Surface-blended Elo | 0.6189 | 0.2154 | 65.1% | 2.27% |
| WElo | 0.6217 | 0.2166 | 64.7% | 2.38% |
| **This model (calibrated)** | **0.6122** | **0.2125** | **65.6%** | **1.34%** |
| Bookmaker closing price (de-vigged) | 0.5841 | 0.2007 | 68.4% | 1.47% |

The model beats both Elo variants on every metric and is the best-calibrated
forecaster in the table — better calibrated than the market itself. These
numbers also replicate the published benchmarks closely: the most recent
comparison on a 2022–2025 test set reports Pinnacle at 68.7% / 0.198 Brier,
WElo at 65.8% / 0.217 and Elo at 65.3% / 0.217.

Calibration is tight across the whole range — predicted vs. realised win rate
never diverges by more than about a point in any decile:

| Predicted | 0.1–0.2 | 0.3–0.4 | 0.5–0.6 | 0.7–0.8 | 0.9–1.0 |
| --- | --- | --- | --- | --- | --- |
| Actual | 0.165 | 0.337 | 0.543 | 0.749 | 0.962 |

### It does not beat the market

Two independent results say so, and they should be read together:

**The encompassing test.** Combining the model with the de-vigged closing price
in an out-of-sample logistic regression puts a weight of **1.05 on the market
and 0.02 on the model** (development window). The combination's Brier score on
the held-out seasons (0.20063) is indistinguishable from the market's alone
(0.20066). Refitted on all 59,438 priced out-of-fold predictions — the version
stored in the artifact — the model's weight is **−0.04**, i.e. not merely zero
but slightly the wrong sign, which is what a coefficient that is really zero
looks like in a finite sample. Formally: the market encompasses the model.
Whatever the model knows, the price already contains.

This is also why `blend_with_market` never fires in production: serving has no
price to blend with. It exists so the encompassing test has something to
measure.

**The betting simulation.** Backing the model's pick, flat stakes, held-out
seasons:

| Confidence gate | Bets | Hit rate | ROI @ Pinnacle | ROI @ Bet365 | ROI @ best price |
| --- | --- | --- | --- | --- | --- |
| ≥ 0.50 (all) | 22,912 | 65.6% | −4.17% | −5.74% | −2.13% |
| ≥ 0.58 | 15,397 | 71.6% | −3.03% | −4.71% | −1.40% |
| ≥ 0.65 | 10,379 | 76.4% | −2.60% | −4.07% | −0.99% |
| ≥ 0.70 | 7,177 | 79.5% | −3.01% | −4.22% | −1.37% |
| ≥ 0.75 | 5,019 | 82.7% | −2.73% | −3.56% | −0.86% |

Hit rate rises with confidence exactly as calibration predicts, and ROI stays
negative everywhere. Note what the ROI floor is: about −2.7% against Pinnacle,
which is roughly Pinnacle's tennis margin. On the picks it is most confident
about, the model is approximately as accurate as the market — it just cannot
out-run the hold. No segment escapes either; ATP, WTA, every surface and every
tier are negative.

Betting the model's *edge* rather than its confidence is worse, not better:
staking every disagreement with the price returns −8.8% at Pinnacle and gets
monotonically worse as the edge threshold rises (−18.3% at a 15-point edge),
with win rates in the 25–38% range. That is the favourite-longshot bias doing
its work: a truthfully calibrated model sees "value" on longshots precisely
where the market shades them, and loses.

### What this model is for

It is a well-calibrated forecaster, not an edge. It is published as unpriced
picks graded on their result, where a 79.5% hit rate at the BET gate is the
claim — and it is a real one. It is not evidence that these are profitable
bets, and nothing in the pipeline claims an edge against a price.

That conclusion is also the one the literature reaches. Kovalchik (2016) found
no published model matching bookmaker accuracy; the 2025 graph-neural-network
study concludes "the market itself encompasses our model." A tennis model that
appeared to beat closing Pinnacle would be far more likely to have a leak than
an edge.

## Retraining

```bash
python -m TennisPredictionModel.tennis_train --refresh
```

Or dispatch `.github/workflows/tennis-train.yml` (manual only, never on the
daily cron). Rewrites four artifacts:

| Artifact | Purpose |
| --- | --- |
| `tennis_model.json` | The exported model, calibration and blend weights |
| `tennis_ratings.json.gz` | Pruned Elo/WElo snapshot the daily job rolls forward |
| `tennis_tournaments.json` | Tournament → surface/tier/best-of, for classifying a live slate |
| `metrics.json`, `backtest.json` | Held-out evaluation |

The daily job does **not** retrain. It loads the snapshot and replays only the
matches that finished since — one current-season workbook per tour — so a
retrain is needed when the model changes or the snapshot drifts a season behind.

## Serving

`generate_tennis_picks(date_iso)` rates the official ESPN ATP+WTA singles slate.
Rows carry a real calibrated probability and a confidence-gated decision, and
are **unpriced by design**: tennis moneylines are outside the shared
market-odds attachment, whose scoreboard parser expects the team-sport shape
(`event.competitions`) rather than tennis's `event.groupings[].competitions`.
So `odds` and `edge` are null and `pricing_type` is `unpriced` — nothing claims
an edge against a price that was never observed.

Picks grade through the existing `grade_tennis_picks` path against ESPN's
competitor-level `winner` flag, exactly like the TennisTonic and Scores24 tennis
feeds.

## References

- Angelini, G., Candila, V. & De Angelis, L. (2022). *Weighted Elo rating for
  tennis match predictions.* European Journal of Operational Research 297(1),
  120–132. — The WElo update and margin weight implemented here, and the source
  of the ROI result this backtest tests against.
- Kovalchik, S. (2016). *Searching for the GOAT of tennis win prediction.*
  Journal of Quantitative Analysis in Sports 12(3), 127–138. — Compares eleven
  published models against bookmaker predictions; the K-factor curve
  `250/(N+5)^0.4` and the finding that no model matched the bookmakers.
- Sackmann, J. (2019). *An Introduction to Tennis Elo*, Heavy Topspin. —
  Surface-specific Elo blending, inactivity handling, and the observation that
  Elo beats rankings but not betting odds.
- *Capturing Intransitive Dominance in Tennis Forecasting: A Graph Neural
  Network Approach* (arXiv:2510.20454). — 2022–2025 benchmark table used above,
  and the encompassing result: "adding [the model] to recalibrated Pinnacle
  gives no Brier-score improvement."
- Shin, H. S. (1993). — The de-vig method implemented in `shin_no_vig`, which
  attributes bookmaker margin to insider trading and so corrects the
  favourite-longshot bias that proportional de-vig leaves in place.

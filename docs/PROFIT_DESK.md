# Profit Desk

## Decision

Profit Desk is the site's price-first decision screen, on its own tab after
Parlays. It does not ask which pick is most likely to win. It asks whether the
available evidence supports a positive return **at the current executable
price**, after uncertainty and portfolio limits. Best Bets (the heuristic
daily shortlist) remains a separate, unchanged tab.

Policy `profit_desk_policy_v2` stakes real units through two qualification
lanes and abstains otherwise:

- **EDGE (1.0u flat)** — strict segment-level market-alpha qualification;
- **VALUE (0.5u flat)** — source-level flat-ROI qualification at posted prices;
- everything else is a watchlist or rejection with exact blockers, and a slate
  with zero qualified picks correctly says `Sit out`.

Live staking begins `2026-07-11`. Earlier slates rebuild as zero-stake
research (`research_backfill`), so a live record can never be backfilled.

## What v1 got right, and why v2 replaced it

The launch policy (`profit_desk_v1_shadow`) was methodologically correct but
structurally impossible to satisfy:

1. it demanded a per-pick `policy_version` that no upstream pipeline writes —
   the selection policy is this engine, so v2 stamps its own version instead;
2. it treated scraped odds feeds as unpriced because provenance markers were
   missing per record, even though those feeds exist to republish bookmaker
   prices; v2 declares that provenance in a source registry (Tier C) and takes
   the feed capture time as the price-observation upper bound;
3. it required a model version even for feeds that have no model; v2 uses the
   source identity as the stable era, keeping explicit model versions where
   they exist;
4. it excluded one-sided posted prices from all evidence; v2 measures them
   against their own vigged break-even — a stricter baseline than no-vig — so
   flat-ROI claims stay conservative;
5. it assumed externally scraped picks could not be graded; the repository's
   grader demonstrably settles hundreds of rows per feed.

v1's statistical machinery (market baseline, hierarchical shrinkage,
uncertainty penalty, chronological stability) is retained unchanged.

## Evidence audit behind the v2 thresholds

Measured on July 11, 2026 against strictly prior settled rows with executable
posted prices and fresh pregame timestamps:

- `mlb_player_props`: 317 rows over 20 distinct dates, +7.8% flat ROI, both
  chronological halves positive, Pr(profitable) ≈ 0.83 → qualifies (VALUE);
- `sportsgambler_mlb`: 299 rows, +2.4% ROI, Pr ≈ 0.67 → watchlist (below the
  0.70 gate; earns its way in prospectively or not at all);
- `scores24_mlb`, `sportytrader_mlb`, all WNBA feeds: flat or negative → the
  gates reject them (WNBA props ran −24% ROI);
- in-house team models price picks with assumed −110 odds
  (`market_total_source: model_output`) → permanently blocked from staking
  until they carry real market prices.

The VALUE thresholds (150 rows, 15 dates, positive ROI, stable halves,
Pr ≥ 0.70) sit exactly in the gap that separates the one provably positive
source from the marginal and negative ones.

## Evidence tiers

| Tier | Requirement | Allowed use |
| --- | --- | --- |
| A: certified executable | Immutable trusted publication before start plus observed executable odds | EDGE and VALUE evidence |
| B: posted two-sided | Fresh paired pregame quote with an exact no-vig conversion | EDGE and VALUE evidence |
| C: posted one-sided | Real offered price without the opposing price (includes registry-declared scraped feeds) | VALUE evidence against its own break-even; never a market-alpha claim |
| D: assumed or synthetic | Assumed, proxy, default, synthetic, stale, or unattributed price | Context only; never profit evidence |

## Selection algorithm

For a candidate with offered decimal odds `d`:

1. Compute the exact break-even probability `p_break_even = 1 / d`.
2. Baseline: an explicit or derived two-sided no-vig probability when the
   market is complete; otherwise the one-sided break-even itself. A one-sided
   implied probability is never mislabeled as no-vig.
3. For strictly earlier verified observations, calculate residuals
   `residual_i = outcome_i - baseline_i`.
4. Shrink the source residual toward zero (prior weight 40 rows), then shrink
   the narrower segment (source + model era + market family + direction +
   probability band) toward the source estimate (prior weight 25 rows).
5. Estimate uncertainty with a conservative variance floor and compute
   `p_est`, `EV = p_est * d - 1`, the one-sided 90% lower bound, and
   `Pr(EV > 0)`.
6. **EDGE lane** (all required): Tier A/B fresh price; ≥100 source rows;
   ≥40 segment rows in the same model era; ≥20 distinct prior dates; both
   chronological halves nonnegative; `Pr(EV > 0) ≥ 0.80`; conservative
   probability ≥ break-even + 2 points; conservative EV > 0.
7. **VALUE lane** (all required): Tier A/B/C fresh executable price;
   ≥150 source rows; ≥15 distinct prior dates; positive source flat ROI; both
   source chronological halves nonnegative; source-level `Pr(EV > 0) ≥ 0.70`.
8. Rank qualified picks (EDGE first, then conservative EV, probability of
   positive EV, evidence depth). Keep at most three per mode and one market
   per canonical game. EDGE stakes 1.0u, VALUE 0.5u, flat.

Raw model probability, model rank, recent win rate, and consensus are display
context only; they never create edge or ranking position.

## Live measurement

Each dated artifact freezes its selections at build time. As the caches grade,
a result-sync pass updates only the outcomes on frozen artifacts — never the
selections, stakes, or ranks — and the newest artifact carries the cumulative
stake-weighted live record (`summary.liveRecordToDate`) since the live
cutover. Rejected candidates remain visible with their exact blockers so
coverage and false-negative behavior stay measurable.

If the gates stop holding — a source's chronological halves go negative, its
ROI turns nonpositive, or its probability of profit decays below the lane
threshold — qualification lapses automatically on the next build and the desk
returns to `Sit out`. That is the design working, not a bug.

## Closing line value and alerts

When a pick settles, the sync pass also attaches the last pregame-captured
price for its exact side as `closing` (with `clv`, entry price versus that
closing observation). Beating the close is the fastest available signal that
selections carry real edge, and the cumulative live record reports `avgClv`
alongside ROI. A `Profit Desk Notify` workflow announces newly published live
picks through the `PROFIT_DESK_WEBHOOK_URL` repository secret (silent until
the secret exists), and never re-pings an unchanged card.

Published past slates are frozen: rebuilds may sync outcomes and closing
prices onto them but never re-run selection, so hindsight can never edit the
record.

## Qualification leaderboard

Each artifact exports per-source VALUE-gate progress (`sources`), rendered on
the Rankings tab: settled priced rows, distinct dates, flat ROI, half
stability, and probability of profit against each threshold. Sources climb
onto the card — and fall off it — purely by these numbers.

## Statistical references

- Glenn Brier, [Verification of Forecasts Expressed in Terms of Probability](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2), 1950.
- Tilmann Gneiting and Adrian Raftery, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437), 2007.
- J. L. Kelly Jr., [A New Interpretation of Information Rate](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x), 1956.

Kelly sizing remains deliberately disabled: stakes are flat per lane. Any
future fractional-Kelly output should be calculated from the conservative
probability and capped; it is a risk policy, not evidence that the probability
estimate is correct.

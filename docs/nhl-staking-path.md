# NHL in-house staking path

Status: **research only**. The NHL artifact has no as-of, independently priced NHL game history or unused holdout. There is no NHL entry in `data/calibration/staking_approvals.json`. Without an earned exact approval, every published NHL pick remains **PASS at 0 units**.

## Frozen research candidate

`nhl_poisson_v2_20252026_shadow_ev_v1` uses last completed regular-season goal rates. For each regular-season moneyline, ±1.5 puck line, and game total, the publisher requires a complete two-sided observed quote. It chooses the side with higher model expected profit per unit at that price. A valid pregame quote clock, both team ratings, and juice no worse than -125 are required for a shadow action:

| Model expected profit per unit | Shadow action |
| --- | --- |
| At least 0.10 | BET, 0.5u |
| At least 0.05 | LEAN, 0.25u |
| Below 0.05 | PASS, 0u |

These thresholds were specified without a historical price search. They are **unvalidated research thresholds**, not evidence of positive return. Visible decisions stay PASS @ 0u; candidate rows carry `source_decision`, `source_units`, `shadow_decision`, `shadow_units`, and `staking_policy: awaiting_approved_holdout`. Team totals and player props remain priced PASS research rows without shadow actions. Preseason publishes no sides.

The DraftKings board is the live quote source. It only attaches a game total or puck line when the two selections share the same posted number, and records retrieval time. The NHL generator stamps its own trusted pregame clock. The existing model-cache refresh, pregame ledger, and team-model evaluator can then retain and grade the shadow rule by fitted version without relying on a mutable serving hash.

## Evidence needed before any live stake

Start an unused chronological holdout only after this candidate is deployed. Do not treat inspected or training-window games as unused. Grade certified independently priced settled rows from the immutable pregame ledger, grouped by game, market, and `nhl_poisson_v2_20252026_shadow_ev_v1`. Compare ROI and a game-clustered lower 95% bound to zero and check calibration against the observed market. The existing publication gate requires at least 100 such actions, positive ROI and clustered lower bound, and no material calibration regression before an exact fitted-version/market approval could allow live stakes. An approval must be written from actual reviewed evidence; this change writes none.

## Prospective approval workflow

The implemented freeze is `data/calibration/nhl_staking_freeze.json`. It fixes
this candidate's source/artifact fingerprint and a holdout start of
`2026-10-01T00:00:00Z`, after all inspected September dates. Deploy before that
start; if deployment misses it, prospectively move the freeze/start before
collecting evidence. Never label an already inspected window unused.

```bash
python3 scripts/frozen_staking_approval.py nhl --output /tmp/nhl-status.json
# Optional recovery of actual cache publications, never synthetic prices:
python3 scripts/frozen_staking_approval.py nhl --backfill
# After reviewing the report and unused-window attestation:
python3 scripts/frozen_staking_approval.py nhl --write
```

The scheduled `frozen-staking-status.yml` workflow uploads reports only. It
also shows each market's settled and pending counts in the job summary; it
cannot write approvals. `awaiting_holdout_evidence` with zero independently
priced settled rows and null ROI means the gate has no October result yet.
`gate_clear_review_required` still requires explicit human evidence review.
The explicit writer reuses the NFL gate/statistics
without changing NFL behavior. Each market qualifies separately: at least
100 certified independently priced settled actions, ROI > 0, game-clustered
lower 95% > 0, and at least 20 paired observations with model Brier no more
than 0.01 worse than market. Only first qualifying event/market observations
count; repeated refreshes cannot multiply the sample. NHL binary calibration
uses probability conditional on no push; financial grading retains pushes.
Exclusion counts are market specific. A record with a different fitted version
cannot consume the first publication slot for the frozen candidate.

The current refresh and auto-grader capture and settle immutable ledger rows.
The report rejects missing fingerprints, unpriced/uncertified rows, invalid
quote clocks, earlier dates and wrong fitted versions. Missing history is
reported as zero observations and null ROI, never reconstructed at today's
prices. Source/artifact changes require a new freeze and unused window.

Publication retains shadow decisions/stakes until an exact fitted-version,
market and frozen-rule approval clears. Promotion also requires a valid
current pregame price. Team totals and player props remain outside this rule.
No approval is supplied by this implementation; holdout evidence must accrue.

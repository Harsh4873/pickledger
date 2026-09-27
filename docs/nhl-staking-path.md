# NHL in-house staking path

Status: **research only**. The NHL artifact has no as-of, independently priced NHL game history or unused holdout. There is no NHL entry in `data/calibration/staking_approvals.json`. Every published NHL pick remains **PASS at 0 units**.

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

Run the audit report with `python3 scripts/team_prop_model_evaluator.py --forward-since <prospectively fixed UTC timestamp>`. A future NHL-specific approval workflow should review the frozen rule and holdout before proposing any approval row. Changing ratings, thresholds, or selection logic starts a new fitted candidate and a new unused holdout.

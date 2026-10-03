# October 3 continuation audit

## Publication regression

The new approval import in `scripts/build_profit_desk.py` worked when imported
by tests but failed under the direct script command used by refresh workflows:
`ModuleNotFoundError: No module named 'scripts'`. External Feed Refresh run
37142284460 failed at that command before committing its rebuilt caches.
The script now adds the repository root to the import path, consistent with
the other command-line entrypoints. A subprocess regression test removes
PYTHONPATH and runs it from a different directory.

## Changed MLS candidate

Dev Checks run 37142818734 had six failures after the MLS retrain. The frozen
fingerprint still identified the previous calibration artifact. The existing
approval gate correctly rejected that mismatch; removing the guard would
have allowed evidence for one model to authorize a different model.

The original freeze is archived unchanged. The retrained candidate has a new
content-bound rule and a prospective window beginning October 4 at midnight
America/Chicago. No staking approval was granted. Synthetic test clocks are
separate from production window dates; separate tests check the real artifact
fingerprint and reject the superseded rule and pre-window observations.

## What the committed results establish

At inspection, the October 3 Profit Desk artifact recorded 53 wins, 31 losses,
one push, and five pending entries: -5.5885 units on 45.5 settled staked units,
or -12.2824% ROI. These are the published portfolio's units, not verified
personal account transactions. Its win rate is not evidence of profitability.
The approvals file was empty.

The NFL metadata reports 18 September holdout games, with model moneyline
Brier 0.25175 versus market 0.24845 (lower is better). This small sample does
not establish an advantage. CFB metadata records 58 holdout games but does
not report a separate scored holdout result.

MLS reports test log loss 1.05319 versus market 1.02768 over 1,987 matches.
This is the broader historical walk-forward test using validation-fitted
calibration, not a new September-only holdout of the shipped calibration.
Do not present it as the latter. The shipped calibration was refitted using
pre-September-21 out-of-sample predictions. Prospective approval remains a
separate requirement.

## Operational limits found during the audit

The initial upcheck passed through its intentional external-feed fallback,
while warning that October 3 player props were missing and the NBA model
bucket was absent. A passing deploy/readiness check is not proof that every
model is fresh. Scores24 college-football coverage was 22 of 25 official
matchups. Tennis reported a July 20 archive cutoff and unrated players.

Use workflow results and a fresh upcheck to distinguish recovered publication
from remaining provider/model coverage gaps. Do not weaken evidence gates to
fill an empty card, manufacture missing prices, or treat retraining as proof
that a strategy is profitable. Historical selection and recorded outcomes
remain intact.

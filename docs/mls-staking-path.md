# MLS prospective staking path

Status: awaiting unused holdout; no approval written. Historical walk-forward
and the inspected live window through September 30 are selection evidence.
The artifact explicitly fits serving calibration on validation plus test;
those dates cannot support a new unused-holdout claim.

`data/calibration/mls_staking_freeze.json` freezes the existing grid candidate:
as-of daily Dixon-Coles ratings, committed calibration and 0.6 market blend,
confidence gates 0.58 BET / 0.545 LEAN, nonnegative model edge, implied-price
cap 0.7143, minimum 8 effective games and existing 0.25–1u sizing. Moneyline
remains research because its posted-price evidence failed. Total and spread
must earn separate approvals. Daily ratings use only results before the
slate date; changing that algorithm or serving calibration invalidates the
content fingerprint. Accumulating prior results is part of the frozen rule.

The unused window starts `2026-10-01T00:00:00Z`. Deploy before then or fix a
new prospective start before collecting evidence. Do not backdate a freeze.

```bash
python3 scripts/frozen_staking_approval.py mls --output /tmp/mls-status.json
python3 scripts/frozen_staking_approval.py mls --backfill
# Explicit review step; a no-op unless all real evidence gates clear:
python3 scripts/frozen_staking_approval.py mls --write
```

The shared NHL/MLS status workflow only uploads reports. Approval requires
100 independently priced settled certified first event/market actions,
positive ROI and game-clustered lower 95%, unused-during-selection attestation,
and at least 20 paired Brier samples with no regression greater than 0.01.
Quotes must have a valid observed pregame clock. The regular cache refresh
captures shadow actions and the existing grader settles ledger records.
Backfill captures actual missing publications; it never invents odds or clocks.
Missing evidence produces an awaiting report with null ROI.

MLS now uses its fitted version for ledger and approval identity. Its
`calibration_excluded` flag follows approval state instead of an unconditional
model-key blacklist, including in ledger eligibility. MLS owns its existing
vector/total calibration; universal calibration takes an audit snapshot and
does not apply a second pooled transform. Eligible approved observations can
enter the calibration outcome ledger for evaluation. Any newly fitted serving
calibration needs a new frozen candidate and prospective holdout.

Until exact fitted-version/market/frozen-rule approval exists, candidate
BET/LEAN stays visible PASS at zero units with preserved source/shadow fields.
No approval rows, historical prices, or holdout metrics are supplied here.

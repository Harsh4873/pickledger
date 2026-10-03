# September 30 desk gate diagnostics

October 3 follow-up: the MLS retrain changed its artifact and invalidated the
fingerprint described below. The replacement freeze and its new prospective
window are documented in `mls-staking-path.md`. The September 30 findings below
describe the earlier candidate, not the retrained model.

## NHL and MLS October holdout status

The frozen candidate fingerprints still match their committed source files.
The NHL and MLS approval evaluators currently find zero independently priced
settled rows after the October 1 UTC holdout start. ROI and confidence bounds
are null, every market is `awaiting_holdout_evidence`, and
`staking_approvals.json` remains empty. The scheduled status workflow reports
these counts without writing an approval. Its per-market candidate exclusions
show historical rows separately from actual holdout evidence.

## MLB inning prices

Two live ESPN DraftKings propBets boards (events 401907972 and 401907897)
were inspected September 30. They exposed F5 moneyline/run-line/total and
team total runs, but no individual-inning or NRFI market. The model's second,
fourth or fifth inning scoreless pick cannot use a first-inning or F5 price.

The direct DraftKings MLB league 84240 board was also checked. Its three
remaining pregame events (34746408, 34746417, 34746424) each exposed 77
markets in the event category endpoint, with no full-inning run market.
The started Phillies/Braves event exposed live **half-inning** markets;
those are neither pregame nor the model's full-inning contract and are rejected.

The odds adapter accepts explicitly labeled full-inning O/U 0.5 or Run Scored
Yes/No quotes if supplied. It requires exact inning and selection, rejects
team/half-inning markets and conflicting duplicate quotes, and never guesses
Over/Under from row order. This is guarded future feed support, not a claim
that ESPN currently supplies those markets. A different provider requires
an adapter based on its actual schema and posted executable prices.

Without a match, the publisher preserves the source decision as research
PASS at 0u. With a match it replaces assumed odds, records provenance and the
selected/opposite prices, devigs only complete pairs, and rechecks positive
EV at the actual price. The independent staking-approval gate still applies;
price availability alone is not proof that the model deserves live stakes.

## WNBA probability versus price

The September 30 moneyline rows use identity calibration; their negative
edges are genuine model/market disagreements, not a global calibration bug.
The current fitted spread and total calibrators shrink the model probabilities
(they are retained). However, the universal edge calculation previously
ignored `market_no_vig_selected_probability`, using stale generator fields or
vigged -110 break-even .5238 instead. Dallas +4.5 had calibrated probability
about .5163 versus fair .510834: positive .55pp, not displayed -.75pp. That
small corrected edge still fails the existing action threshold.

The old same-matchup SQLite query also lacked a date constraint. Serving now
prefers the requested date's ESPN board and accepts only same-date database
moneylines as fallback. Its generic single spread/total price lacks selection
identity, so that fallback no longer prices a specific side.

WNBA now uses the current attached no-vig selected-side baseline while
retaining the learned probability and decision thresholds. ESPN parsing also
keeps home/away spread and Over/Under prices separate; an away or Under pick
cannot inherit the opposite side's price. Missing prices no longer default
to fabricated -110. Complete pairs define edge = model probability minus
no-vig probability; one-sided prices remain conservative break-even baselines.
No calibrator was refitted on these inspected live examples, and no positive
edge or bet is promised by correcting the arithmetic.

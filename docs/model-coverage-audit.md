# Source and model coverage audit — September 21, 2026

## Changes

MLB full-game inference now consumes observed pregame ESPN scoreboard totals, both side prices, provider, capture time and game start before predicting. The parser preserves the exact line and selected-side price. Ambiguous doubleheaders are excluded until an event-ID join is available. A direct September 21 run returned observed totals for all three games; the previously committed cache had only three moneylines.

Tennis replays official completed singles results when the workbook archive is stale. It excludes future results, qualifiers and unknown court metadata. A September 21 run applied 1,722 results and advanced ratings from July 20 to September 20 with no failed result downloads. This does not repair missing ranking/points inputs: the archive still returns HTTP 403 and 24 players are unrated. Source health exposes those gaps even when cache generation succeeds.

Football now handles grouped ESPN rosters and paginated prop feeds. NFL can use the existing paired Action Network quote / dated ESPN history baseline when the primary feed produces no candidates. Both football baselines have stable rank metadata and remain uncalibrated PASS projections with zero stake. The paired-quote join requires both teams, kickoff, athlete identity, book, exact line and both prices. No fabricated odds are introduced.

Source-health warnings run in all three refresh jobs and the local upcheck; main receives the same development checks as pull requests. These warnings distinguish fresh output from incomplete inputs.

The repaired production refresh published 38 NFL PASS projections. A later scheduled refresh encountered an upstream HTTP 504; failed empty football refreshes now retain earlier same-day, uncalibrated, zero-stake research from immutable snapshots. Original quote timestamps and outage diagnostics remain intact. Healthy empty slates and prior-day picks are not carried forward.

## Coverage and evidence limits

- MLB player props: the September 21 cache had 1,615 candidates across three games, 4,272 scored variant rows and 24 published research PASS rows. Sparse actionable output is principally a qualification gate, not absence of candidate generation. Current artifacts qualify hits and RBIs; strikeouts, combined hits/runs/RBIs and other markets fail existing policies. Strikeouts show 6/13 validation and 7/12 holdout wins. Missing under prices also exclude rows. Counts of rejection reasons are variant-level and must not be interpreted as unique players. Publication also applies per-player and per-game concentration limits. None of these gates was loosened.
- NFL player props: the inspected ESPN game exposed 1,113 rows over two pages; the inspected first page had no usable prices. The repaired fallback produced 24 PASS candidates for Giants–Rams: two passing-yards, four rushing-yards, nine receiving-yards and nine receptions. It joined all quoted athletes; three player/stat combinations lacked sufficient prior history. Other prop markets remain unsupported by this baseline.
- CFB player props: all 25 official September 26 games matched the alternative schedule, but no eligible paired player quotes were available at inspection. This is an input gap; no missing prices or players are invented. Nonempty nested-roster fixtures verify publication and ranking.
- NFL and CFB team models support moneyline, spread and full-game total. The September 21 NFL cache has all three for its one game; CFB has no game that day. Dedicated team totals, halves and quarters are not implemented by these models. The raw NFL feed contained team-total rows without usable prices, so those do not establish deployable market coverage.

## Certified evaluation and forward window

The certified ledger now captures zero-stake forecasts for every deployed team model, including MLS, WNBA, MLB team totals, NBA, Tennis and IPL. A prediction fingerprint includes implementation and fitted artifacts; it is stamped only on fresh inference, before kickoff preservation. Historical rows are not retroactively assigned a new version. Evaluation retains certification, revision and price exclusions, adds exact line/selection/offered-price groups, and compares model and market on the same priced sample. PASS forecasts contribute probability scoring, never actionable ROI.

Auto-Grade stores a downloadable evaluation artifact on every run. The prospective window is fixed at **2026-09-22T00:00:00Z**. Reports remain insufficient below 100 priced, settled actionable observations per version/market; reaching 100 only requests review and never approves promotion. Historical gate selection is not an untouched holdout. Do not change this boundary after seeing results.

No new model or staking policy is promoted by this repair. MLB full-game training ends in 2025 and warrants a later retraining experiment once observed-input coverage is stable. Tennis ranking inputs remain incomplete. NFL/CFB player baselines lack native calibrated betting models; their projections stay research-only. Existing selected historical segments and high win rates do not establish current profitability. Evaluate retraining against frozen temporal splits and the prospective versioned window before replacing serving artifacts.

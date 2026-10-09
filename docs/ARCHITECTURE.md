# Architecture

## Production Path

```text
model/feed Actions
      |
      v
committed JSON in data/
      |
      +--> scheduled ESPN auto-grader --> committed results
      |
      +--> Profit Desk builder --> dated live decision artifacts
      |
      v
Vite static build --> GitHub Pages
```

The production viewer is intentionally static. It does not load Firebase, authenticate users, or call the optional Python backend.

## Frontend Contract

- `src/data.ts` loads every dated file listed in `data/model_cache/index.json`.
- `src/data.ts` loads every dated file listed in `data/player_props_cache/index.json`.
  Public buckets are `mlb_player_props`, `nba_player_props`, `wnba_player_props`,
  `nfl_player_props`, and `cfb_player_props`. NFL/CFB boards may be empty
  (off-day or unpriced) without blocking MLB publication.
- `src/data.ts` loads the compact, precomputed files listed in
  `data/profit_desk/index.json`; the browser never invents a profit score from
  raw picks.
- Picks receive deterministic browser IDs so client-side ESPN grades can be stored locally.
- `src/main.ts` renders Home, Search, Rankings, Best Bets, Parlays, and the
  Profit Desk from the same pick collection plus the precomputed desk artifact.
- Rankings are calculated from committed results across all manifest dates.
- Best Bets is the heuristic daily shortlist. Profit Desk is its own tab after
  Parlays: it requires observed, fresh pricing, uses strictly prior evidence,
  applies shrinkage and an uncertainty penalty, and stakes only through its
  qualification lanes (EDGE 1.0u, VALUE 0.5u) — otherwise it publishes 0u and
  says so. See `docs/PROFIT_DESK.md`.
- The Refresh button checks ESPN for pending games and stores temporary local grades. The scheduled grader remains authoritative because it writes results into repository JSON.

## Writer Contract

The model-cache, player-prop, external-feed, and calibration workflows share
the `pick-cache-writer` concurrency group. The high-frequency auto-grade and
closing-line workflows have dedicated groups: GitHub retains only one pending
run per group, so putting them in the heavy-writer queue can silently evict a
pending daily refresh. Those dedicated writers resync `main`, recompute after a
rejected push, and retry; the cache mergers preserve committed grades and game
times when a heavy writer publishes concurrently.

Model and feed refreshes:

1. Generate JSON without Firestore writes.
2. Reset to the latest `main`.
3. Freeze started games: in-house team rows whose kickoff has already passed
   at refresh time are dropped from the generated payload so the merge keeps
   the rows published before kickoff. A late refresh can never publish or
   re-decide a live game.
4. Merge the generated payload.
5. Attach real pregame market prices (`scripts/market_odds.py`): both
   moneylines, total over/under prices, spreads, and MLB first-5-innings
   markets from the ESPN scoreboard/prop feeds. Scraped picks keep their own
   executable odds and gain a verifiable two-sided baseline; in-house model
   picks with assumed prices have them replaced by the real observed price
   for their exact market and line. Captured pregame prices are preserved by
   the merge layer once a game goes live. Capture is gated on the wall clock,
   not only on the provider's status flag.
6. Preserve existing `result`, `start_time`, and `game_start_time` fields for matching picks.
7. Demote unpriced stakes: after the attach, an in-house BET/LEAN whose price
   is still missing or still equal to the model's own placeholder becomes
   PASS at 0u (`unpriced_demoted`, `source_decision` preserved). A stake nobody
   can place is research.
8. Commit and push as the triggering GitHub actor.

NFL and CFB mint stakes only through the graduated `decision_policy` bands
that their training scripts validated out of fold at recorded prices — BET is
the strongest validated band, LEAN the next band that still clears its own
bar (see `docs/model-audit-2026-09.md`). Every other row, including all
moneyline and spread rows for both sports, publishes as visible PASS research
with a `confidence_label`; in-house PASS cards stay on the board whenever the
published side is the model's favourite.

For the audited in-house team-model buckets (`mlb_new`, `mlb_first_five`,
`mlb_inning`, `fifa_world_cup`, and `nba_summer`), the model refresh also
stores an immutable first-publication/revision record in
`data/calibration/team_prop_pregame_ledger/`. Certification requires a
trusted per-pick publication timestamp earlier than the scheduled start.
Legacy cache rows remain visible but are not promoted into certified evidence.

The ledger uses compact daily JSON shards selected by `slate_date`, then the
date of `game_start_time` or `published_at`, with `undated` as the fallback.
Each line wraps an unchanged record with its global insertion sequence;
`index.json` stores the original root metadata and shard counts. Dates split
into numbered parts before 8 MB. Load through
`scripts.team_prop_pregame_ledger.load_team_prop_pregame_ledger()` to retain
the original payload shape and ordering.

Run `python -m scripts.team_prop_pregame_ledger --migrate` to migrate or retry
a migration after updating the checkout. The loader also unions a recreated
legacy `data/calibration/team_prop_pregame_ledger.json` by record ID. Settlement
retractions outrank old binary grades; attached results outrank pending copies;
explicit record update times break ties, then the sharded copy wins. A root
publication timestamp is not a grade timestamp. Missing or damaged shards
raise an error so incomplete history cannot silently enter approvals.

The writer atomically replaces only files whose bytes changed, installs the
index last, and removes the monolith after success. Interrupted initial
migrations retain the monolith and can be rerun. Model refresh saves the whole
directory, then uses `--merge-from SAVED_REPO_ROOT` after resyncing to union
concurrent records and grades. Calibration refresh excludes this input ledger
from its artifact restore. `python scripts/check_data_file_sizes.py` checks
versioned and new nonignored JSON under `data/` against 50 MB, and ledger files
against 10 MB; Dev Checks and model publication run it.

Quote and publication clock changes alone do not create a material revision
when certification and eligibility are unchanged. Actual prices, picks,
features, model versions, provenance, and eligibility changes still count;
historical snapshots and their hashes are never rewritten by this comparison.

The certified ledger separates three concepts:

- forecast evaluation, which includes certified PASS/LEAN/BET outcomes;
- market/ROI evaluation, which additionally requires observed executable odds;
- calibration training, which additionally requires explicit eligibility and
  continues to exclude FIFA.

`scripts/team_prop_model_evaluator.py` reads only this ledger and reports
chronological metrics by model version and market, including Brier score, log
loss, calibration bins, verified-price ROI, market benchmarks, and retained
feature-contract coverage.

The universal probability calibrator has a versioned training contract. Rows
whose probabilities are owned by the player-prop ML policy remain available
for evaluation but are explicitly ineligible to train the separate shared
Platt layer. A training-contract change invalidates the prior mapping and
forces evaluation against a clean identity champion.

`scripts/cache_manifest.py` updates the dated-cache manifest whenever model or feed caches are written or merged.

## NBA player props

`nba_player_props` scores ESPN posted DraftKings markets with the four heuristic
variants (season, all-time, hot L10, matchup H2H) over ESPN gamelogs. There is no
trained NBA prop artifact: `player_props/ml.py` routes NBA to its own
`nba_player_props_ml.joblib` (absent, so it scores from the market-anchored
projection baseline and never borrows the WNBA artifact), and the season/history
consensus gate has no NBA entry, so no NBA market can publish or stake. Priced
NBA rows still flow to `research_candidates` and the Profit Desk player-prop
research shortlist at 0u. ESPN's season phase is stamped as `season_type`;
preseason or unverified-season rows are labeled `NBA PRESEASON — research/entertainment
only`, single only, never in parlays or Edge/Prop Doubles, at most two, ranked
after regular-season rows. Activating an NBA market needs a trained season/history
artifact whose chronological validation and later holdout both clear the same
70% accuracy and sample floors the other sports use.

## NFL/CFB player props

NFL player props use the ESPN posted-market consensus path. CFB uses the public
Action Network scoreboard, book registry and two-sided player markets, joined to
ESPN by both teams, kickoff and an unambiguous roster name. Public cache keys are
`nfl_player_props` and `cfb_player_props`. Covered markets:

- passing yards / TDs / completions, interceptions
- rushing yards / attempts / TDs
- receiving yards / receptions / TDs

There is no synthetic-line fallback and no basketball-artifact borrowing.
Until native NFL consensus joblibs exist, that board remains empty. CFB publishes
historical baseline projections as PASS with zero stake and explicit uncalibrated
status. They use up to 12 dated games from the current and prior season, weighted
by 0.85 per game of age, with at least four observations. A rolling historical MAE
is reported separately from betting calibration; it is not evidence of a betting
edge. Opponent strength, transfer role changes and participation probabilities
are not modeled. The board says so in each row. Zero catches and zero rushing
yards are retained; passing priors exclude games without a passing attempt.

CFB rejects ambiguous player/game joins, mismatched line/side/book pairs, opening
or consensus prices, live games, and undated or same-day/future outcomes. Rows
retain ESPN game/player IDs for grading and provider IDs plus retrieval time for
source inspection. The existing refresh, merge, immutable archive and grading
pipeline carries these rows; baseline PASS rows are not limited by staking caps.
The market-history job grades immutable pregame CFB snapshots against final ESPN
box scores, feeding the existing native training corpus without relying on the
missing ESPN `propBets` endpoint. Post-kickoff captures never enter that corpus.
Per-game diagnostics distinguish unavailable markets, unmatched players and
insufficient history. An outage is a soft-fail: other sports and Pages proceed.
Dates stamp `America/Chicago`. CFB scoreboards use ESPN FBS `groups=80`.

## Deployment Contract

`.github/workflows/deploy-pages.yml` runs on every push to `main`. It first checks that today's model and player-prop caches are complete. Incomplete daily refreshes defer deployment without failing; ready data is built, copied into `dist/`, and deployed to GitHub Pages.

## Verification

```bash
npm run build
npm run typecheck
python3 -m pytest tests/smoke/test_static_viewer.py -q
python3 scripts/auto_grade_picks.py
```

Visual browser inspection is intentionally left to the repository owner.

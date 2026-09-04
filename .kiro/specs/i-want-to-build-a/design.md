# Design — College Football (CFB) Prediction Model

This design evolves the existing `CFBPredictionModel/` in place, adds a separate
CFB player-props subsystem inside `player_props/`, fixes the current model's
shortcomings, and introduces a variant bake-off with a formal model-selection
test, a roster-change training adjustment, and a promotion gate. It references
the real code as it exists today.

## 1. Current state (what we build on)

`CFBPredictionModel/` today:

- `cfb_core.py` — the market-free data spine and strict as-of feature builder.
  - Data: SportsDataverse ESPN-derived schedules (`load_schedule_season`),
    resolved betting releases + multi-book median history
    (`load_betting_season`, `load_line_odds`), and the live ESPN FBS scoreboard
    (`load_live_slate`, `groups=80`).
  - `FEATURE_NAMES` = 16 market-free features (EWMA off/def, `elo_diff`,
    `schedule_strength_diff`, rest, `week`, `neutral_site`, `conference_game`,
    `early_season`). Sportsbook lines are targets/labels only.
  - `TeamState` holds EWMA offense/defense + Elo, with an `OFFSEASON_DECAY`
    (0.58) applied at each `roll_season`. `build_dataset` emits one record per
    finished FBS-vs-FBS game **before** updating state (no lookahead);
    `serving_rows`/`features_for_slate` build the live slate the same way.
- `cfb_train.py` — walk-forward (2021–2025) over two families (`ridge`,
  `hist_gradient_boosting`), selecting the family with the lowest combined
  margin+total MAE, then fits final margin+total regressors, isotonic
  calibrators (moneyline/spread/total) on OOF predictions, and a bivariate
  Gaussian residual covariance. Writes `artifacts/cfb_model.joblib` +
  `metadata.json` with `shadow_mode: true`, `promotion_status: "not_qualified"`.
- `cfb_model.py` — `generate_cfb_picks(date_iso)` loads artifacts, builds the
  slate, and emits three market rows per game (`h2h`, `spread`, `totals`) with
  `_decision` (EV+probability thresholds), assumed -110 for spread/total,
  observed prices for moneyline. Every row carries `shadow_mode: true`.
- Registered as `server.run_cfb_model` and included in
  `refresh_model_cache.py::_model_jobs["cfb"]` and the daily default model list.
  `market_odds.SPORT_LEAGUES["CFB"]`, `odds_api.SPORT_KEYS["CFB"]`, and
  `pickgrader_server.SPORT_TO_ESPNSLUG["CFB"]` all exist.

Shortcomings this design fixes: near-coin-flip edge (thin features, no
roster-change handling), hand-picked family selection (MAE only, no formal
test), no player props, and no promotion path off shadow mode.

## 2. High-level architecture

```
CFBPredictionModel/                 team markets (evolved in place)
  cfb_core.py        + roster-continuity data & feature, richer features
  cfb_features.py    NEW  opponent-adjusted efficiency, pace, travel, continuity
  cfb_variants.py    NEW  variant registry (algo × feature-set specs)
  cfb_selection.py   NEW  walk-forward eval + LRT / AIC-BIC model selection
  cfb_train.py       rewritten to run the bake-off and record full metadata
  cfb_model.py       serving: winner artifact, SEC-coverage gate, promotion flag
  cfb_promotion.py   NEW  per-market promotion gate from committed graded history
  artifacts/         cfb_model.joblib (+ selection_report in metadata.json)

data/cfb/
  conferences.json   NEW  team_id -> conference (SEC membership source of truth)
  returning_production/{season}.csv  NEW  roster-continuity inputs (committed)

player_props/                       props subsystem (separate)
  api.py             + cfb_scoreboard, cfb_team_roster, cfb_player_gamelog,
                       cfb_boxscore  (ESPN football/college-football)
  football_cfb.py    NEW  generate_cfb_candidate_model(api, date_iso)
  ml.py              + CFB stat families in MARKET_FAMILY_NAMES,
                       SPORT_ARTIFACTS["CFB"]
  generator.py       + build_variant_buckets(sport="CFB", ...)
  artifacts/         cfb_player_props_ml.joblib (+ metadata) once trained

pickgrader_server.py + CFB football player-stat extractor + prop-fetch branch
scripts/train_cfb_player_props_ml.py  NEW  prop-model training (mirrors NFL/MLB)
.github/workflows/cfb-train.yml       extended: bake-off + prop training
tests/smoke/test_cfb_model.py         extended for all new behavior
```

The team model and the prop model are independent: they train separately, ship
separate artifacts, publish into separate cache buckets (team → `model_cache`
`cfb`; props → `player_props` payload `CFB` bucket), and are gated separately.

## 3. Team model — fixing shortcomings

### 3.1 Richer, still market-free features (`cfb_features.py`)

New features appended to `FEATURE_NAMES` (all derived from completed games
strictly before kickoff, preserving the no-lookahead contract; **no sportsbook
field is ever added**):

- **Opponent-adjusted offensive / defensive efficiency** — points-per-drive
  style ratings adjusted for opponent strength via a lightweight ridge/iterative
  adjustment over the season-to-date graph (two features: `home_adj_off_eff`,
  `home_adj_def_eff` differenced against away).
- **Pace / possessions** — estimated drives or plays per game EWMA
  (`pace_diff`), which conditions totals.
- **Home / travel factor** — travel distance bucket + altitude/home indicator
  (`travel_factor`), replacing the flat neutral/home Elo bump with a graded
  term.
- **Roster-continuity factor** (see 3.2) — `home_continuity`, `away_continuity`,
  `continuity_diff`.

`TeamState` gains the efficiency/pace accumulators; `_feature_row` emits the new
names. `FEATURE_NAMES` stays the single serving contract asserted in tests and
stored in metadata.

### 3.2 Roster-change adjustment (training)

Two complementary mechanisms, both recorded in metadata:

1. **Per-team continuity factor.** From committed `data/cfb/returning_production/`
   (returning starters / returning production share per team-season, sourced
   from a public returning-production dataset). At each `roll_season`, a team's
   carried-over EWMA/Elo is shrunk toward the league mean by
   `1 - continuity` instead of the flat `OFFSEASON_DECAY`: a team that returns
   most of its roster keeps more of its prior rating; a gutted roster reverts
   harder. The raw continuity value is also exposed as a feature (3.1).
2. **Season-recency sample weighting.** In the bake-off fits, older seasons get
   an exponential recency weight `w = decay^(current_season - game_season)` so
   the model leans on recent, roster-relevant games without discarding history.
   Default decay is a variant hyperparameter chosen by the bake-off.

Both are pure training-time constructs; serving is unchanged beyond the new
features. If `returning_production` data is missing for a season the continuity
factor falls back to the existing flat `OFFSEASON_DECAY` (documented, not
silent — recorded in metadata as `continuity_fallback: true`).

### 3.3 Variant bake-off + formal selection (`cfb_variants.py`, `cfb_selection.py`)

`cfb_variants.py` defines a **registry of variant specs**, each a
`{name, family_factory, feature_subset, sample_weighting}`:

- Families: `ridge` (nested baseline), `hist_gradient_boosting`, and a `ridge`
  with the expanded feature set (nested superset of the baseline ridge — enables
  a true LRT).
- Feature subsets: `core16` (today's features) and `extended` (core + 3.1).
- Sample weighting: `flat` and `recency`.

`cfb_selection.py` runs **one shared walk-forward split** (same seasons/folds
for every variant) and produces, per target (margin, total):

- Held-out MAE, spread/total direction rate, Brier (as today) for each variant.
- **Nested comparison (LRT).** For nested pairs (baseline ridge ⊂ extended
  ridge on the same folds), compute the Gaussian log-likelihood of held-out
  residuals under each model and the likelihood-ratio statistic
  `LR = 2(ll_full - ll_reduced)`, referenced to a χ² with df = added params;
  report the statistic and p-value. The extended model is preferred only if the
  LRT is significant at a documented α (default 0.05).
- **Non-nested comparison.** Between different families, select on held-out
  walk-forward metric (combined margin+total MAE, tie-break by direction rate),
  with **AIC/BIC** reported as a secondary criterion. AIC/BIC use the held-out
  Gaussian log-likelihood and the effective parameter count (for HGB, an
  effective-df estimate from leaf count is recorded; documented as approximate).

Selection rule, in order:
1. Choose the best feature set for the linear family by LRT.
2. Compare the LRT-selected linear model against the boosted family on held-out
   metric; pick the better, AIC/BIC as tie-break.
3. Fix the seed so selection is reproducible (Requirement 5.6).

`cfb_train.py` calls the selection, fits the **winner** on all records (with the
winner's weighting), computes calibrators + residual covariance exactly as
today, and writes `cfb_model.joblib` for the winner only. `metadata.json` gains a
`selection_report`:

```json
"selection_report": {
  "variants": [{"name","feature_set","family","weighting",
                "walk_forward":{...},"aic","bic","loglik"}],
  "nested_tests": [{"reduced","full","lr_statistic","df","p_value","alpha","selected"}],
  "selected": "<variant name>",
  "selection_basis": "lrt|walk_forward_metric|aic_bic_tiebreak"
}
```

`selected_family` is retained for backward compatibility with the existing test.

### 3.4 SEC-first coverage & promotion (`cfb_model.py`, `cfb_promotion.py`)

- `data/cfb/conferences.json` maps `team_id -> conference`; a helper
  `is_sec_game(game)` returns true when either team is SEC. Coverage scope is a
  module constant / env (`CFB_PUBLISH_SCOPE = "sec" | "fbs"`), default `sec`.
- `cfb_promotion.py::market_promotion_status()` reads **committed graded
  history** (the forecast-audit ledger / `team_prop_pregame_ledger`) and, per
  market (`h2h`, `spread`, `totals`), returns `publishable` when
  `graded_n >= MIN_N` **and** the graded hit rate clears the break-even for the
  prevailing price (moneyline: observed-price break-even; spread/total: 52.4% at
  -110), else `shadow`. Thresholds are documented constants.
- In `generate_cfb_picks`, each row's `shadow_mode` becomes
  `not (in_scope and market_publishable)`. In-scope + gated markets publish
  (`shadow_mode: false`); everything else keeps `shadow_mode: true` and flows to
  the audit ledger exactly as today. The market-free contract and all existing
  row fields are unchanged.

Metadata records `publish_scope` and each market's `promotion_status` + metrics.

## 4. Player-props subsystem (separate)

### 4.1 Data access (`player_props/api.py`)

Add ESPN football/college-football methods to `DirectApiClient`, mirroring the
existing basketball/mlb methods and their retry+cache behavior:

- `cfb_scoreboard(date_iso)` → games (`groups=80`).
- `cfb_espn_prop_bets(event_id)` → posted player-prop lines when ESPN exposes
  them; when absent, the generator projects and prices at the framework default
  (-110), matching how other sports fall back.
- `cfb_player_gamelog(player_id, season)` and `cfb_team_roster(team_id)` for
  projections.
- `cfb_boxscore(event_id)` / summary for grading (4.3).

### 4.2 Candidate model (`player_props/football_cfb.py`)

`generate_cfb_candidate_model(api, date_iso) -> list[dict]` builds per-player
projections for the required stat families and emits picks **only** via
`schema.build_pick(...)` — reusing `normal_probability`, `kelly`,
`decision_and_stake`, `market_fair_probability` unchanged, so CFB props inherit
the exact staking/decision/no-vig-gating logic other sports use.

Stat families (added to `ml.MARKET_FAMILY_NAMES`): `passing_yards`,
`rushing_yards`, `receiving_yards`, `receptions`, `passing_touchdowns`,
`rush_rec_yards`. Each `stat_key` maps to a family via `market_family_for_stat`.

Projections come from player game-log EWMA vs. opponent defense-allowed rates,
with a per-player sigma for `normal_probability`. Same SEC-first scope helper
applies (props for non-SEC players stay unpublished until scope expands).

### 4.3 Wiring & ranking (`generator.py`, `ml.py`)

- `generator.generate_payload` adds
  `**build_variant_buckets(sport="CFB", date_iso=date_iso, base_model=cfb_candidates)`;
  the existing `assign_ml_ranks` / `select_top_props` (top-N, max-per-player)
  then applies unchanged.
- `ml.SPORT_ARTIFACTS["CFB"]` → `cfb_player_props_ml.joblib` +
  `cfb_player_props_ml_metadata.json`. Until a native CFB artifact validates,
  CFB borrows the NFL/analog artifact **capped at LEAN** via the framework's
  existing cross-sport rule (documented, not new logic).

### 4.4 Training (`scripts/train_cfb_player_props_ml.py`)

Mirrors `train_player_prop_ml.py`: assemble CFB player game logs → build the
`FEATURE_NAMES` feature vectors → walk-forward-by-season fit → write
`player_props/artifacts/cfb_player_props_ml*.{joblib,json}` with `active`,
`feature_names`, `market_families`, `validation` (hit rate / calibration), and a
`training_fingerprint`. Refuses on insufficient sample.

## 5. Grading (`pickgrader_server.py`)

- **Team markets** already work via `SPORT_TO_ESPNSLUG["CFB"]` +
  `fetch_scoreboard` + `grade_pick` (the pick text encodes market/line). No
  change beyond ensuring published CFB rows carry `grade_supported: true` (they
  already do) and correct team names in `pick`.
- **Player props** need a new **football player-stat extractor**
  `_extract_cfb_player_stat(summary, player, stat_key)` reading the ESPN
  college-football box-score/summary endpoint, plus a `sport_key == "CFB"`
  branch in the prop-fetch fallback so `grade_player_prop_pick` can resolve
  actuals for the CFB stat families. Grading semantics (exact line → push, else
  direction; canceled/postponed → push) are the shared engine's, unchanged.

## 6. Scheduling & integration

- **Daily team refresh:** already wired (`_model_jobs["cfb"]`, default model
  list). No change needed; publication now depends on the promotion gate.
- **Daily prop refresh:** `player-props-refresh.yml` already calls
  `generate_payload`, so adding the CFB bucket there is automatic once
  `generator.py` includes it.
- **Training workflow:** extend `.github/workflows/cfb-train.yml` (currently
  `workflow_dispatch`) to run the team bake-off **and**
  `scripts/train_cfb_player_props_ml.py`, run smoke tests, and upload both
  artifact sets. Cadence stays manual/dispatchable for now (in-season retrains
  are run on demand), matching the other per-model training workflows.
- **Up-check:** add `cfb` to `site_upcheck.REQUIRED_MODEL_KEYS` **only once CFB
  is promoted**, so a missing refresh is detected without failing while CFB is
  still shadow-only. (Gate this behind the promotion status to avoid false
  alarms.)
- **Frontend:** `src/data.ts` already labels CFB (`CFB ML/Spread/Total`) and
  `src/main.ts` maps the league; verify the CFB primary filter renders published
  rows and that shadow rows remain filtered. No new market types are introduced.

## 7. Data contracts & schemas

- **Team pick row:** unchanged shape from today's `_row`; only `shadow_mode`
  (now gated) and the enlarged `features` map change. Downstream merge, calib,
  and market-odds attach are untouched.
- **Prop pick:** canonical `schema.build_pick` output (`scope: "player"`,
  `sport: "CFB"`, `stat_key`, `line`, `projection`, `probability`, `edge`,
  `decision`, `units`, `result: "pending"`).
- **`metadata.json` (team):** existing keys + `selection_report`,
  `roster_adjustment` (`{method, recency_decay, continuity_source,
  continuity_fallback}`), `publish_scope`, and per-market `promotion_status`.
- **Artifacts:** `cfb_model.joblib` (winner only) unchanged in structure
  (`margin_model`, `total_model`, `calibrators`); prop artifacts as in 4.4.

## 8. Testing (`tests/smoke/test_cfb_model.py` + new prop tests)

Extend/keep the existing suite and add:

1. **No-lookahead** preserved with the new features (state updated only after a
   record is emitted; slate features use only prior games).
2. **Market-free contract**: `FEATURE_NAMES` contains no sportsbook field even
   after expansion (existing assertion, extended).
3. **Bake-off/selection**: on a synthetic dataset, `cfb_selection` emits a
   `selection_report` with variant metrics, a computed LRT statistic + p-value
   for the nested pair, AIC/BIC for non-nested, a `selected` variant, and
   **reproducible** selection under a fixed seed.
4. **Roster adjustment**: continuity shrink behaves monotonically (higher
   continuity ⇒ less reversion) and falls back cleanly when data is missing.
5. **Promotion gate**: a market below `MIN_N` or below break-even stays shadow;
   one meeting both publishes; evaluation reads committed history only.
6. **SEC scope**: with scope=`sec`, only SEC games publish; others stay shadow.
7. **Prop schema**: `generate_cfb_candidate_model` output conforms to
   `build_pick` and maps every `stat_key` to a market family.
8. **Grading**: a completed CFB game grades team picks win/loss/push; a CFB prop
   with a fetched box score grades to a real result (extractor smoke).
9. **Registration**: `run_cfb_model`, `SPORT_TO_ESPNSLUG`, `SPORT_LEAGUES`,
   `SPORT_KEYS`, `SPORT_ARTIFACTS["CFB"]`, and frontend labels are present.
10. **Non-regression**: the full existing smoke suite passes unchanged.

## 9. Risks & mitigations

- **Returning-production data availability/quality.** Mitigation: committed
  dataset with a documented flat-decay fallback; continuity is one signal among
  many, not a hard dependency.
- **HGB effective-df for AIC/BIC is approximate.** Mitigation: LRT is the
  primary rule for nested linear comparisons; AIC/BIC is only a non-nested
  tie-break and is labeled approximate in metadata.
- **ESPN CFB prop lines may be sparse.** Mitigation: project-and-price at the
  framework default when a posted line is absent, exactly like other sports;
  no-vig gating still applies when a real two-sided price exists.
- **Premature promotion.** Mitigation: gate reads committed graded history with
  a minimum sample and break-even threshold; default scope is SEC-only and
  default state remains shadow until the gate is met.
- **Thin early-season signal / roster churn.** Mitigation: recency weighting +
  continuity reversion; early-season games remain flagged (`early_season`).

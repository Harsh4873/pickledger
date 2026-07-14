# NBA Playoffs Prediction Model

This model is the playoff-specific NBA workflow for PickLedgerPro. It reuses shared data acquisition where practical, but its pace, margin, total, injury, series-state, and probability calculations are playoff-specific rather than cloned from the regular NBA model.

## Verification Gate

`run_live.py` only emits picks for games that:

- appear on ESPN's NBA postseason scoreboard (`seasontype=3`),
- have not started yet,
- have team statistics available through the NBA API,
- have current market moneylines available from the ESPN scoreboard odds payload.

When a game fails those checks, the runner prints the reason and does not emit a pick.

## Data Sources

- ESPN scoreboard: playoff schedule, series headline, game status, venue, moneyline, spread, and total.
- NBA API: season team efficiency, recent form, rosters, and schedule/rest context.
- Existing NBA injury feed: current injury statuses and on/off impact adjustment.

## Playoff Logic

`run_live.py` now applies a separate playoff layer for:

- series state, including Game 1, Game 2, closeout, and elimination-game handling,
- slower pace and higher halfcourt weighting as a series advances,
- shorter rotations, star minute inflation, and reduced bench influence,
- stricter injury weighting for expected absences,
- stronger playoff home court,
- matchup-specific halfcourt efficiency, turnover, and rebounding adjustments,
- coaching and repeated-matchup adjustments after teams have seen each other.

## Execution

Run for a selected date:

```bash
../.venv/bin/python run_live.py --date 2026-04-23
```

The backend route is `/run-nba-playoffs-model`, and the Firestore cache key is `nba_playoffs`.

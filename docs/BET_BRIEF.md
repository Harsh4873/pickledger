# Bet Brief (Harsh's scheduled picks)

The bet brief turns Profit Desk into one actionable card for Harsh Dave:
max 1-2 BET THIS per day plus an explicit PASS list, with his personal
ledger snapshot attached. It never invents odds, edges, or P&L, and it
never places bets. Harsh places every ticket manually.

## Schedule

`.github/workflows/bet-brief.yml` runs twice daily:

- Morning: 14:00 UTC (9am America/Chicago in daylight time), after Daily Refresh
- Afternoon: 20:00 UTC (3pm America/Chicago in daylight time), a refresh that
  drops started games and re-runs the same gate

Manual runs: Actions, Bet Brief, Run workflow, with optional `slot`
(auto, am, pm, manual) and `date` (YYYY-MM-DD backfill).

Each run writes `data/bet_briefs/YYYY-MM-DD-<slot>.json` plus `.md`, refreshes
`data/bet_briefs/latest.json` and `latest.md`, updates `index.json`, commits as
the triggering actor through the `pick-cache-writer` queue, and triggers a
Pages deploy so `harsh.html` stays current.

## Climb filter

Input is `data/profit_desk/latest.json` (or the dated file for backfills).
The filter applies in this order:

1. Hard exclusions: NO tennis ever, tier AVOID, price not fresh pregame, no
   observed executable price, missing odds (never guessed).
2. Juice cap: odds at or past -200 are out unless huge +EV, defined as
   conservative EV above 0.05 with probability of positive EV at least 0.80
   (checked on both the headline and VALUE estimates).
3. Started games: anything starting within 10 minutes of the run cutoff
   (`--as-of`, default now) is off the board.
4. Qualified first: Profit Desk EDGE then VALUE picks, max 2, ranked by desk
   rank. These are the only picks labeled qualified.
5. Climb fallback: when the desk qualifies zero (a valid sit out), take max 1
   single from decision BET rows that also survive a proven-negative screen
   (sources at or below -5% flat ROI over 50+ rows are out). Rank by sport
   priority (NFL, CFB, MLB, WNBA, MLS, then the rest) and cleanest price
   (closest break-even to -110). LEAN-only slates sit out.
6. Singles only for climb (Novig, Onyx, best-available per Policy 9/22 correction: each book gets action on day with slate, labeled qualified, weak-day lean, or last-resort; still prefer qualified EDGE/VALUE; still skip proven-negative and juice past about -200; do not double Novig lean onto Onyx).
   Never parlays in climb. Stake guide is Novig first (about $1 = 1u), Onyx
   as backup (about $0.50 = 1u). ReBet and Fliff are daily-refresh longshot
   lane only, every day per Policy 9/22
   (see `docs/DAILY_REFRESH_LONGSHOT.md`), never climb singles.

Every PASS row cites its real reason: the exclusion above, or the desk
decision and tier, the source flat ROI over rows and dates with Prob+EV, and
the top desk blockers.

## Personal ledger

`data/personal_ledger.json` is Harsh's book-by-book record. 9/19 history was
reconstructed from the desk file on 9/20; snapshot moment is the BOOK RESET
9/19 about 8:14 PM CT.

- Books: Novig (active primary, best-available, $1/u, snapshot $30, cashout
  $50), Onyx (active, best-available distinct from Novig, $0.50/u, snapshot $5 post-loss, cashout
  $20), ReBet (active daily-refresh, every day, $0.25 soft/u, snapshot $1 refresh 9/22,
  cashout $20), Fliff (active daily-refresh promo, every day, $0.50 soft/u,
  snapshot $2 refresh 9/22, cashout $50). Policy 9/22 correction: all four books get best-available every
  day; Novig and Onyx best-available (no longer surefire-only silence; labeled qualified, weak-day lean, or last-resort; never same pick on both).
- Running bankroll = snapshot bankroll + settled P&L on or after the snapshot
  date, excluding preSnapshot history. Pending bets never move settled
  bankroll. Pre-snapshot 9/19 tickets carry `preSnapshot: true` (visible in
  lifetime P&L, never subtracted from post-reset snapshots).
- ReBet $1 and Fliff $2 refresh daily, play every day per Policy 9/22;
  ledger running for those books is informational, Boss confirms the promo
  balance each morning. Never OUT or HOLD for good while the promos live.
  Longshot lane only, see `docs/DAILY_REFRESH_LONGSHOT.md`.
- Units = stake dollars / unit dollars. Historical units reflect the policy
  in force at placement. Null only for pre-unit-policy tickets.
- Wins and losses require confirmed American odds from Harsh or the ticket.
  Inferred odds stay flagged unconfirmed until Boss confirms. Unknown odds
  stay null until confirmed.

CLI:

```bash
python3 scripts/personal_ledger.py --ledger data/personal_ledger.json summary
python3 scripts/personal_ledger.py --ledger data/personal_ledger.json add \
  --date 2026-09-21 --book Novig --sport NFL \
  --selection "Under 43.5 (CAR @ ATL)" --odds -110 --stake 1.0
python3 scripts/personal_ledger.py --ledger data/personal_ledger.json settle \
  --id pl-20260919-003 --result win --odds 221
```

All four books are active; no OUT or HOLD gate applies. Novig and Onyx take
climb singles only, best-available (thin boards surface best distinct 1 per book if posted pregame non-tennis candidate exists, labeled qualified, weak-day lean, or last-resort; still skip proven-negative and juice past about -200, no tennis; do not double Novig lean onto Onyx; do not return zero tickets for active book when slate has playable candidate). ReBet and Fliff take daily-refresh
longshots only, every day. Cashout targets (Policy 9/22): Novig $50, Fliff
$50, ReBet $20, Onyx $20.

## Local brief runs

```bash
python3 scripts/generate_bet_brief.py --slot am
python3 scripts/generate_bet_brief.py --slot pm --stdout | head -n 40
python3 scripts/generate_bet_brief.py --slot am --as-of 2026-09-20T14:00:00Z
python3 -m pytest tests/smoke/test_bet_brief.py tests/smoke/test_personal_ledger.py -q
```

## Frontend

`harsh.html` (linked as HARSH from the main board) fetches
`data/bet_briefs/latest.json` and `data/personal_ledger.json` from the
deployed site and renders today's BET THIS plus the ledger. The deploy
workflow copies both data paths into the Pages artifact.

## Guardrails

- Prices are quoted from Profit Desk evidence (posted book lines). If the
  Novig ticket differs, the brief says to pass or resize.
- Climb singles are labeled not qualified unless the desk qualified them.
- Nothing in this pipeline calls a sportsbook, places a bet, or sizes one
  from invented edge. The Grok bot replacement is this scheduled brief plus
  the ledger, not automation that bets.

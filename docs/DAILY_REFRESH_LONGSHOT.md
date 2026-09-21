# Daily-refresh longshot lane (ReBet and Fliff)

ReBet $1 and Fliff $2 refresh daily. This lane spends each refresh on one
chalk-stack longshot parlay per book per day. It is separate from the
Novig and Onyx climb singles (those stay singles only, never parlays).
Harsh places every ticket manually; nothing here auto-places.

## Method: chalk-stack with filters

Stack heavy favorites (chalk) on the moneyline only. No spreads, totals,
or props in this lane. No same-game or correlated legs.

1. Model win-chance filter: each leg must have model probability at least
   0.65 in the committed PickLedger cache (`data/model_cache/latest.json`).
   Below 0.65, the leg is out, even if the payout looks nice.
2. Juice reasonableness: skip legs heavier than -500 (that is, -501 or
   shorter like -950 or -2100). They add little payout for the added loss
   risk. Prefer -150 to -400.
3. Edge filter (A and M lesson): compute edge as model probability minus
   market break-even. Drop any leg with edge below -0.05. Among the rest,
   rank by edge (best first) and take the top legs per the caps below.
   The 9/19 Onyx 10-leg ML parlay at +259 lost; Texas A and M was the
   worst-edge landmine in that stack. Lesson: cap legs and drop any
   worst-edge leg, even if dropping it lowers the payout.
4. Leg caps: ReBet 2 to 3 legs (cap 3), Fliff 2 to 4 legs (cap 4), prefer
   3. Never more than 4. Ten-leg sprays are out.
5. Pregame only: each leg must have a fresh executable posted price
   (observed within 24h, start more than 10 minutes out). If any leg
   moved more than 10c against the posted price on the book ticket, drop
   that leg or pass the ticket.
6. One ticket per book per day, full refresh ($1 ReBet, $2 Fliff). No
   chase, no second ticket after a loss. If no legs pass the filters,
   pass and note why; the balance refreshes tomorrow anyway.
7. Confirm the book ticket before placing. Book parlay pricing may differ
   from the multiplied posted legs. Log the book combined odds, never the
   computed estimate, as the ticket odds.

## Next slate tickets (9/21): PASS both, do not bet

Status as of 2026-09-21 14:27 UTC: the committed PickLedger slate is 9/21
(`data/model_cache/latest.json` dated 2026-09-21 11:56Z,
`data/profit_desk/latest.json` dated 2026-09-21 12:14Z). Morning refresh
has run. Only 10 ML legs exist on the slate, and only 2 pass the filters.
Insufficient for two non-overlapping 2+ leg plus-money longshots, so both
books PASS. No ticket is listed.

ReBet ($1): PASS. Only 1 fully qualifying leg (LA ML) plus 1 conditional
(Dallas Wings, missing start_time in source bucket). Best 2-leg combo is
about -103 (not a plus-money longshot). Do not bet.
Fliff ($2): PASS. No legs left for a second non-overlapping 2+ leg ticket.
Do not bet.

Passing legs (2, ranked by edge):

- LA ML (NYG at LA, nfl) -285, prob 0.7266, break-even 0.7403, edge -0.0137.
  Start 2026-09-22T00:15Z (Sep 21 7:15 PM CT), pregame, fresh from the 11:56Z
  refresh, odds via nflverse_posted_lines. Fully qualifies.
- Dallas Wings ML (Dallas at Phoenix, wnba) -218, prob 0.659, break-even
  0.6855, edge -0.0265. Start missing in the wnba bucket (same game
  2026-09-22T02:00Z in forebet_wnba and scores24_wnba, Sep 21 9 PM CT),
  market updated 11:56Z fresh via DraftKings/ESPN. Conditional: needs
  pregame confirm, no start_time in source bucket.

Failed legs (8, all below the 0.65 win-chance filter):

- Toronto Blue Jays ML -110 (0.54), Blue Jays ML -107 (0.4767), Nationals ML
  +129 (0.4767), Giants ML -102 (0.4954), Atlanta Dream ML -120 (0.515), plus
  3 F5 MLs: Orioles -125 (0.4733, edge -0.082), Tigers -154 (0.4419, edge
  -0.164), Twins -120 (0.4628, edge -0.083). All out. F5 MLs are first-five
  derivatives, not full-game MLs, and fail anyway.

Best available 2-leg math (DO NOT BET, shown for transparency): LA -285 (dec
1.3509) plus Dallas -218 (dec 1.4587) gives combined dec 1.9705, about -103.
$1 pays about $1.97, $2 pays about $3.94, hit chance 0.7266 times 0.659 =
47.9 percent, EV about -0.056. Not a plus-money longshot (needs 3+ legs for
plus money, only 2 available), and Dallas start needs confirm. So PASS both.
Balances refresh tomorrow.

Build command used (same filter as Method, all buckets scanned for ML):

```bash
python3 -c "
import json
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
d = json.load(open('data/model_cache/latest.json'))
print('cache date', d.get('date'))
for bucket in ['nfl', 'mlb_new', 'wnba', 'mls', 'cfb']:
    b = d.get(bucket, {})
    for p in b.get('picks', []):
        if 'ML' not in str(p.get('pick', '')):
            continue
        try:
            prob = float(p.get('probability'))
            odds = int(p.get('odds'))
        except (TypeError, ValueError):
            continue
        if prob < 0.65 or odds <= -501:
            continue
        be = abs(odds) / (abs(odds) + 100.0)
        edge = prob - be
        if edge < -0.05:
            continue
        print(bucket, p.get('pick'), odds, round(prob, 4), 'edge', round(edge, 4), p.get('start_time'))
"
```

Take the passing legs ranked by edge, split into two non-overlapping
tickets per the caps (ReBet 3, Fliff 4, prefer no shared legs so one
landmine does not wipe both), compute the combined decimal as the product
of leg decimals (decimal = 1 + 100/abs(odds) for favorites), confirm the
book combined price, then log.

## Expired 9/20 examples (DO NOT BET)

Method illustration only, built from committed 9/20 live data. All legs
kicked 9/20 and are final. Do not place these.

Sources: `data/model_cache/2026-09-20.json` (NFL bucket, odds via
nflverse_posted_lines and ESPN Scoreboard DraftKings, captured
2026-09-20T11:45Z). Probabilities are model `probability` (calibrated).
Break-even for favorites: abs(odds)/(abs(odds)+100). Edge = prob minus
break-even. Combined decimal = product of leg decimals. American from
decimal: (decimal - 1) * 100 for decimal above 2. Combined hit chance =
product of leg probs (independence assumption, no same-game).

Dropped landmines on this slate (all failed a filter, A and M lesson):

- Minnesota Lynx ML -2100, prob 0.849: heavier than -500, edge -0.106, out.
- New York Liberty ML -1200, prob 0.758: heavier than -500, edge -0.165, out.
- SF ML -950, prob 0.8817: heavier than -500, out (thin payout for the risk).
- CAR ML -155, prob 0.6418: below the 0.65 win-chance filter, out despite
  positive edge. SEA ML -218 (0.642), DAL ML -225 (0.649), GB ML -192
  (0.6274): all below 0.65, out.

### ReBet expired example: 3 legs, +164, $1 to win $1.64 (DO NOT BET)

- NE ML (PIT at NE) -230, prob 0.6959, break-even 0.6970, edge -0.0011
- BAL ML (NO at BAL) -380, prob 0.7752, break-even 0.7917, edge -0.0165
- CHI ML (MIN at CHI) -218, prob 0.6616, break-even 0.6855, edge -0.0239

Decimals: 1.4348 * 1.2632 * 1.4587 = 2.644. American about +164.
Potential payout on $1: about $2.64. Model-implied hit chance:
0.6959 * 0.7752 * 0.6616 = 0.357 (about 36 percent). Combined EV:
2.644 * 0.357 - 1 = about -0.06. Kicks 9/20 17:00Z. Final. Do not bet.

### Fliff expired example: 4 legs, +188, $2 to win $3.76 (DO NOT BET)

- PHI ML (PHI at TEN) -325, prob 0.7404, break-even 0.7647, edge -0.0243
- KC ML (IND at KC) -265, prob 0.6868, break-even 0.7260, edge -0.0392
- TB ML (CLE at TB) -425, prob 0.7677, break-even 0.8095, edge -0.0418
- LAC ML (LV at LAC) -340, prob 0.7308, break-even 0.7727, edge -0.0419

Decimals: 1.3077 * 1.3774 * 1.2353 * 1.2941 = 2.880. American about +188.
Potential payout on $2: about $5.76. Model-implied hit chance:
0.7404 * 0.6868 * 0.7677 * 0.7308 = 0.285 (about 29 percent). Combined EV:
2.880 * 0.285 - 1 = about -0.18. Kicks 9/20 17:00Z and 20:05Z plus 9/21
00:20Z. All started or final as of 9/21 04:37Z. Do not bet.

No shared legs between the two examples, so one landmine does not wipe
both. Rank order by edge among the seven qualifying NFL legs: NE, BAL,
CHI, PHI, KC, TB, LAC.

## Recording longshots

Log the book combined odds from the ticket, never the computed estimate.
For a pending ticket with legs known but book combined not yet confirmed:

```bash
python3 scripts/personal_ledger.py --ledger data/personal_ledger.json add \
  --date 2026-09-21 --book ReBet --sport NFL \
  --selection "3-leg chalk: NE ML + BAL ML + CHI ML" --stake 1.0 \
  --legs 3 --odds-unknown \
  --notes "Daily-refresh longshot. Legs pregame, book combined TBD."
```

Once Harsh confirms the book combined (example +164):

```bash
python3 scripts/personal_ledger.py --ledger data/personal_ledger.json settle \
  --id pl-20260921-001 --result win --odds 164 \
  --notes "Book combined +164 confirmed from ticket."
```

Losses need the book combined too (ledger requires odds for wins and
losses). If the book will not show a combined until placement, keep the
ticket pending with --odds-unknown until Harsh reads it off the ticket.
ReBet and Fliff are active daily-refresh; no --allow-hold gate applies.

## Guardrails

- Prices and probs come only from committed PickLedger caches or the book
  ticket. Unknown or future odds stay TBD, never guessed.
- Climb singles (Novig, Onyx) stay singles only. Parlays live only in this
  daily-refresh lane.
- Nothing here places bets. Harsh places every ticket manually.

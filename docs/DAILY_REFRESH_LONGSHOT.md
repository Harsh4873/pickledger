# Daily-refresh longshot lane (ReBet and Fliff)

ReBet $1 and Fliff $2 refresh daily, play every day per Policy 9/22 (cashout
targets ReBet $20, Fliff $50; Novig $50, Onyx $20; Novig and Onyx weak-day
leans when playable, still prefer qualified and singles). This lane spends
each refresh on one 2-leg longshot per book per day (Method 9/22 fix: fewer
legs, higher bar, no padding). It is separate from the Novig and Onyx singles
(those stay singles only, never parlays). Harsh places every ticket manually;
nothing here auto-places.

## Method: chalk-stack with filters (Method 9/22 fix in effect)

Method 9/22 fix (one weak leg killed both 9/21 tickets; misses were relaxed
legs, real chalk covered): fewer legs (exactly 2 each), higher bar (prob
0.60 plus, edge 0.00 plus, price -125 to -200), rank by prob, verified starts
only, no padding with coin-flips to field two tickets. Replaces the 9/20
method below where they differ; unchanged rules still apply.

Stack favorites on the moneyline only. No spreads, totals, props, F5
derivatives, or same-game/correlated legs in this lane. No tennis.

1. Model win-chance filter: each leg must have model probability at least
   0.60 in the committed PickLedger cache (`data/model_cache/latest.json`).
   Below 0.60, the leg is out. Up from relaxed 0.47 that allowed 0.47-0.55
   coin-flips on 9/21; realistic for MLB slate where 0.65 plus executable
   legs are rare, still excludes coin-flips below 0.60.
2. Edge filter (positive only): compute edge as model probability minus
   market break-even. Drop any leg with edge below 0.00. No negative-edge
   landmines. Up from -0.05. Rank qualifying legs by prob (highest win chance
   first, not edge) to prioritize hitting. The 9/19 Onyx 10-leg at +259 lost
   with Texas A and M as the worst-edge landmine; the 9/21 best-effort used
   negative-edge legs (-0.01 to -0.04) and Boss spread-variants lost by relaxed
   legs. Lesson: fewer legs, positive edge only, drop landmines even if payout
   drops.
3. Price filter (real chalk only): each leg must be -125 to -200 inclusive.
   Required (not preferred). Excludes too-light coin-flips like -102/-107/-120
   and too-heavy thin payout like -500 plus. Up from skip heavier than -500
   (preferred -150 to -400). 9/21 weak MLs (-102, -107, -120) all fail this;
   9/21 heavy (-285, -218) also fail as too heavy for longshot value.
4. Leg caps (fewer legs): ReBet exactly 2 legs, Fliff exactly 2 legs. No 3-4
   leg padding. Down from ReBet 2-3 and Fliff 2-4 (prefer 3). Two legs means
   fewer landmines and higher hit chance than 3-4.
5. Pregame only with verified start: each leg must have a fresh price in
   today cache (observed within 24h via refresh) and a verified start_time
   more than 10 minutes out. No inferred starts; drop the leg if start is
   missing. Price source is any cache odds with verified start
   (posted_market preferred, forebet reference allowed with book confirm since
   9/22 posted_market MLs are all weak; never invented). If any leg moved more
   than 10c against on the book ticket, drop it or pass.
6. One ticket per book per day, full refresh ($1 ReBet, $2 Fliff). No chase,
   no second ticket after a loss. No padding: if fewer than 4 legs qualify,
   field one 2-leg ticket (best 2 by prob) and pass the other book; never add
   below-bar legs to force two tickets. Every-day means play when legs qualify,
   refresh daily, not forced. If zero legs qualify, pass both and note why.
7. Confirm the book ticket before placing. Book parlay pricing may differ
   from the multiplied reference legs. Log the book combined odds, never the
   computed estimate, as the ticket odds.

## Next slate tickets (9/22, Method 9/22 fix)

Status as of 2026-09-22 14:49 UTC: slate is 9/22 (`data/model_cache`
11:48Z, `data/profit_desk` 12:36Z). Morning refresh has run. Four legs
clear the Method 9/22 fix bar (prob 0.60 plus, edge 0.00 plus, price -125
to -200, verified starts, ML only, no tennis). Split 2 plus 2, no shared
legs, no padding. Harsh places manually, confirms book per-leg and combined
prices, no auto-place. Computed odds are estimates from reference legs; log
the book combined from the ticket, never the estimate. If a leg moved more
than 10c against on the book, or started, drop it or pass.

ReBet ($1, 2 legs, about +192): Dodgers ML -125 + Tigers ML -161. $1 pays
about $2.92, hit about 48.3 percent. No shared legs with Fliff.
Fliff ($2, 2 legs, about +179): Mariners ML -149 + Rangers ML -149. $2 pays
about $5.59, hit about 40.9 percent. No shared legs with ReBet.

Legs (4 forebet MLs, ranked by prob per Method 9/22 fix, all pregame as of
14:49Z, all 8 to 11.5 hours out):

- Dodgers ML (SD at LAD) -125, prob 0.71, break-even 0.5556, edge +0.1544.
  Start 2026-09-23T02:10Z (Sep 22 9:10 PM CT), verified pregame. Price is
  forebet reference (no book source in cache); mlb_new posted_market Dodgers
  -120 for same game confirms similar real-price range. Book leg price to
  confirm.
- Tigers ML (WSH at DET) -161, prob 0.68, break-even 0.6169, edge +0.0631.
  Start 2026-09-22T22:40Z (Sep 22 5:40 PM CT), verified pregame. Forebet
  reference; mlb_new posted_market Tigers -157 confirms range. Book to confirm.
- Mariners ML (HOU at SEA) -149, prob 0.66, break-even 0.5984, edge +0.0616.
  Start 2026-09-23T01:40Z (Sep 22 8:40 PM CT), verified pregame. Forebet -149;
  mlb_new posted_market Mariners -149 same price confirms executable. Book to
  confirm.
- Rangers ML (NYM at TEX) -149, prob 0.62, break-even 0.5984, edge +0.0216.
  Start 2026-09-23T00:05Z (Sep 22 7:05 PM CT), verified pregame. Forebet -149;
  mlb_new posted_market Rangers -149 same price confirms executable. Book to
  confirm.

Ticket math (from reference legs, confirm book combined):

- ReBet: 1.80 * 1.6211 = 2.9180, about +192. $1 pays about $2.92. Hit 0.71
  times 0.68 = 48.3 percent. EV about +0.409 (positive, higher bar works).
- Fliff: 1.6711 * 1.6711 = 2.7926, about +179. $2 pays about $5.59. Hit 0.66
  times 0.62 = 40.9 percent. EV about +0.142 (positive).

Why these 4 (Method 9/22 fix: prob 0.60 plus, edge 0.00 plus, price -125 to
-200, exactly 2 each, rank by prob, verified starts, no padding): all 4 clear
higher bar (probs 0.62-0.71, all positive edges, all real chalk -125 to -161,
all verified pregame starts, 4 distinct games, no shared legs). No coin-flip
padding (all above 0.60, none 0.47-0.55). Fewer legs (2 each, not 3-4).

Excluded (honest): all posted_market MLs fail prob below 0.60 (best are Sky
0.651 but edge -0.059 fails 0.00, Fever 0.513, Yankees 0.516, etc.) or edge
negative or price outside -125 to -200. WNBA heavy chalk Mystics -1600
(0.844), Aces -1800 (0.816), Valkyries -950 (0.795) excluded (heavier than
-200, edges -0.09 to -0.13, thin payout). Forebet D-backs -200 (0.61) fails
edge -0.057. Forebet Brewers +110, Guardians +110, Giants +110 are underdogs
(not -125 to -200). Forebet Angels -110, Yankees -125 (0.56), Orioles -125
(0.52) fail prob below 0.60. All F5 MLs are derivatives, excluded (best F5
prob 0.557 Tigers, edge -0.059 fails). All tennis excluded (no tennis, 30
tennis MLs with no odds/prob anyway).

## Expired 9/21 best-effort MLs (not placed, DO NOT BET)

Status as of 2026-09-21 15:03 UTC: the committed PickLedger slate is 9/21
(`data/model_cache/latest.json` dated 2026-09-21 11:56Z,
`data/profit_desk/latest.json` dated 2026-09-21 12:14Z). Morning refresh
has run. Boss override 15:00 UTC: build best-effort longshots even though
full filters say PASS (only 2 pass, best 2-leg about -103). Tickets below
relax win-chance and start verification as noted. Harsh places manually,
confirms the book combined price, no auto-place. Computed odds are estimates
from posted legs; log the book combined from the ticket, never the estimate.
If a leg moved more than 10c against on the book, or started, drop it or pass.

ReBet ($1, 3 legs, about +290): Giants ML -102 + LA ML -285 + Dallas Wings
ML -218. $1 pays about $3.90, hit about 23.7 percent. No shared legs with
Fliff. See legs and caveats.
Fliff ($2, 2 legs, about +255): Atlanta Dream ML -120 + Blue Jays ML -107.
$2 pays about $7.09, hit about 24.6 percent. No shared legs with ReBet. See
legs and caveats.

Legs (5 executable full-game MLs, ranked by edge per method, all pregame as
of 15:03Z, all 452 to 657 min out):

- Giants ML (Twins vs Giants, mlb_new) -102, prob 0.4954, break-even 0.5050,
  edge -0.0096. Start 2026-09-22T01:45Z (Sep 21 8:45 PM CT), pregame, updated
  11:56Z fresh via posted_market. Relaxed: prob below 0.65, lighter than
  preferred -150 to -400.
- LA ML (NYG at LA, nfl) -285, prob 0.7266, break-even 0.7403, edge -0.0137.
  Start 2026-09-22T00:15Z (Sep 21 7:15 PM CT), pregame, fresh from the 11:56Z
  refresh via nflverse_posted_lines. Fully qualifies (price has no timestamp,
  certification 11:56Z).
- Dallas Wings ML (Dallas at Phoenix, wnba) -218, prob 0.659, break-even
  0.6855, edge -0.0265. Start missing in the wnba bucket (same game
  2026-09-22T02:00Z in forebet_wnba and scores24_wnba, Sep 21 9 PM CT),
  market updated 11:56Z fresh via DraftKings/ESPN. Relaxed: start inferred,
  needs book pregame confirm.
- Atlanta Dream ML (Atlanta at New York, wnba) -120, prob 0.515, break-even
  0.5455, edge -0.0305. Start missing in the wnba bucket (same game
  2026-09-22T00:00Z in forebet_wnba and scores24_wnba, Sep 21 7 PM CT),
  market updated 11:56Z fresh. Relaxed: prob below 0.65, start inferred,
  lighter than -150.
- Blue Jays ML (Blue Jays vs Orioles, mlb_new) -107, prob 0.4767, break-even
  0.5169, edge -0.0402. Start 2026-09-21T22:35Z (Sep 21 5:35 PM CT), pregame,
  updated 11:56Z fresh via posted_market. Relaxed: prob below 0.65, lighter
  than -150.

Ticket math (from posted legs, confirm book combined):

- ReBet: 1.9804 * 1.3509 * 1.4587 = 3.9025, about +290. $1 pays about $3.90.
  Hit 0.4954 * 0.7266 * 0.659 = 23.7 percent. EV about -0.074. Contains 1
  relaxed prob leg (Giants) plus 1 conditional start (Dallas).
- Fliff: 1.8333 * 1.9346 = 3.5467, about +255. $2 pays about $7.09. Hit
  0.515 * 0.4767 = 24.6 percent. EV about -0.129. Both legs relaxed prob,
  Atlanta start inferred.

Relaxed filters (honest, Boss override):

1. Win-chance 0.65 relaxed to 0.47 to get 5 legs (only LA and Dallas pass
   0.65). Giants, Atlanta, Blue Jays included by override. Why: thin slate
   (10 ML legs total, only 2 pass), need min 4 for two 2+ leg tickets. Hit
   chances drop to about 24 percent each.
2. Start verification relaxed for Dallas and Atlanta (no start_time in wnba
   bucket, inferred from same-game entries in other buckets). Why: data gap,
   same-game starts exist in cache. Must confirm pregame on the book; if
   started, drop the leg or pass the ticket.
3. Preferred juice -150 to -400: Giants -102, Atlanta -120, Blue Jays -107
   are lighter than -150 (coin-flip range, not heavy chalk). Allowed by the
   hard cap (none heavier than -500) but outside preferred. Noted.

Kept: ML only, pregame only (all more than 10 min out), edge at or above
-0.05 (all 5 pass), no same-game within a ticket (5 distinct games: MIN at
SF, NYG at LA, DAL at PHX, ATL at NYL, TOR at BAL), no shared legs (3 plus 2
uses all 5), fresh executable prices (11:56Z, within 24h, posted_market or
nflverse, no invented odds), confirm book combined.

Excluded: Nationals +129 (underdog, not chalk, excluded despite +0.04 edge),
Toronto forebet -110 (no price source, not executable, same game as Blue Jays
-107, used the executable version), 3 F5 MLs (derivatives, fail prob and edge:
Orioles -125 edge -0.082, Twins -120 edge -0.083, Tigers -154 edge -0.164).

Build command used (same filter as Method, all buckets scanned for ML, then
Boss override relaxes win-chance to 0.47 for this slate):

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

## Expired 9/21 spread-variants (settled LOSS Sep 22, DO NOT BET)

Boss placed both 9/21 longshots as SPREAD variants on Sep 21, not the ML
best-effort above. Both lost by one leg Sep 22 per Boss. Logged as settled
losses in `data/personal_ledger.json` (`pl-20260921-001` ReBet $1 LOSS,
Wings -5.5 missed, Giants and Rams hit; `pl-20260921-002` Fliff $2 LOSS,
Jays -1.5 missed, Dream hit) with combined odds TBD (Boss did not send),
payouts never invented. ReBet and Fliff reset to 9/22 refresh ($1 and $2
fresh, losses in lifetime only).

- ReBet $1 (3 legs, `pl-20260921-001`): Giants ML -102 + Rams -6.5 + Wings
  -5.5. Combined TBD, payout TBD. Legs: Giants ML -102 (Boss confirmed,
  cache -102 prob 0.4954 start 01:45Z), Rams -6.5 (Boss line, cache LA -6.5
  -115 prob 0.5216 start 00:15Z as reference, book leg price TBD), Wings -5.5
  (Boss line, no direct cache price, opposite Mercury +5.5 -108 as reference
  only, start inferred 02:00Z, needs pregame confirm).
- Fliff $2 (2 legs, `pl-20260921-002`): Dream -2.5 + Blue Jays -1.5. Combined
  TBD, payout TBD. Legs: Dream -2.5 (Boss typed -2.4, treat as -2.5 typo, flag
  for confirm; no direct cache price, opposite NYL +1.5 -110 different line as
  reference only, start inferred 00:00Z, needs pregame confirm), Blue Jays -1.5
  (Boss line, no MLB run line price in cache, book leg price TBD, game start
  22:35Z).

No shared legs between the two placed tickets (Giants/Rams/Wings vs
Dream/Blue Jays, 5 distinct games). Both were pregame as of placement. Spreads
that were real chalk covered (Giants, Rams, Dream); misses were relaxed legs
(Wings -5.5, Jays -1.5). Lesson applied in Method 9/22 fix above (fewer legs,
higher bar, no padding).

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

#!/usr/bin/env python3
"""Personal ledger CLI for Harsh Dave.

Records only what Harsh confirms. Never invents odds, edges, or P&L.
Never places bets. It only tracks open and settled tickets plus running
bankroll per book.

Ledger math:
  running bankroll = snapshot bankroll + settled P&L on or after snapshot date
Pending bets never move settled bankroll. Units = stake / unit dollars.
Unknown odds stay null until Harsh or the ticket confirms them.

Usage:
  python3 scripts/personal_ledger.py --ledger data/personal_ledger.json summary
  python3 scripts/personal_ledger.py --ledger data/personal_ledger.json add \
    --date 2026-09-21 --book Novig --sport NFL --selection "Under 43.5 (CAR @ ATL)" \
    --odds -110 --stake 2.0 --notes "Brief pick"
  python3 scripts/personal_ledger.py --ledger data/personal_ledger.json add \
    --date 2026-09-21 --book Novig --sport MLS --selection "Under 3.5 (TBD)" \
    --stake 2.0 --odds-unknown
  python3 scripts/personal_ledger.py --ledger data/personal_ledger.json settle \
    --id pl-20260920-003 --result win --odds -110
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

RESULTS = ("win", "loss", "push", "void", "pending")
SETTLE_RESULTS = ("win", "loss", "push", "void")


def american_profit(stake: float, odds: int, result: str) -> float:
    if result in ("push", "void", "pending"):
        return 0.0
    if result == "loss":
        return round(-stake, 2)
    if odds > 0:
        return round(stake * odds / 100.0, 2)
    return round(stake * 100.0 / abs(odds), 2)


def load_ledger(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def save_ledger(path: Path, ledger: dict) -> None:
    ledger["updatedAt"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "w") as fh:
        json.dump(ledger, fh, indent=2)
        fh.write("\n")


def book_map(ledger: dict) -> dict:
    return {b["book"].lower(): b for b in ledger.get("books", [])}


def settled_profit_for_book(ledger: dict, book: str, snapshot_date: str | None) -> float:
    total = 0.0
    for bet in ledger.get("bets", []):
        if bet.get("book", "").lower() != book.lower():
            continue
        if bet.get("status") in (None, "pending"):
            continue
        if snapshot_date and str(bet.get("date", "")) < snapshot_date:
            continue
        total += float(bet.get("profitDollars") or 0.0)
    return round(total, 2)


def pending_risk_for_book(ledger: dict, book: str) -> float:
    total = 0.0
    for bet in ledger.get("bets", []):
        if bet.get("book", "").lower() != book.lower():
            continue
        if bet.get("status") == "pending":
            total += float(bet.get("stakeDollars") or 0.0)
    return round(total, 2)


def running_bankroll(book_entry: dict, settled_profit: float) -> float | None:
    snap = book_entry.get("snapshotBankroll")
    if snap is None:
        return None
    return round(float(snap) + settled_profit, 2)


def summarize(ledger: dict) -> dict:
    books_out = []
    for entry in ledger.get("books", []):
        name = entry["book"]
        snap_date = entry.get("snapshotDate")
        settled = settled_profit_for_book(ledger, name, snap_date)
        pending = pending_risk_for_book(ledger, name)
        books_out.append(
            {
                "book": name,
                "status": entry.get("status"),
                "unitDollars": entry.get("unitDollars"),
                "openingBankroll": entry.get("openingBankroll"),
                "snapshotBankroll": entry.get("snapshotBankroll"),
                "snapshotDate": snap_date,
                "settledProfitSinceSnapshot": settled,
                "pendingRisk": pending,
                "runningBankroll": running_bankroll(entry, settled),
                "notes": entry.get("notes"),
            }
        )
    open_bets = [b for b in ledger.get("bets", []) if b.get("status") == "pending"]
    settled_bets = [b for b in ledger.get("bets", []) if b.get("status") != "pending"]
    return {
        "owner": ledger.get("owner"),
        "updatedAt": ledger.get("updatedAt"),
        "books": books_out,
        "openCount": len(open_bets),
        "settledCount": len(settled_bets),
        "open": open_bets,
        "settled": settled_bets,
    }


def next_id(ledger: dict, date: str) -> str:
    prefix = "pl-" + date.replace("-", "")
    taken = {b.get("id", "") for b in ledger.get("bets", [])}
    n = 1
    while f"{prefix}-{n:03d}" in taken:
        n += 1
    return f"{prefix}-{n:03d}"


def cmd_summary(args: argparse.Namespace) -> int:
    ledger = load_ledger(Path(args.ledger))
    summary = summarize(ledger)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    lines = [f"Personal ledger for {summary['owner']} (updated {summary['updatedAt']})", ""]
    for b in summary["books"]:
        run = b["runningBankroll"]
        run_txt = f"${run:.2f}" if run is not None else "n/a"
        unit = b["unitDollars"]
        unit_txt = f"${unit:.2f}/u" if unit else "unit n/a"
        lines.append(
            f"{b['book']} [{b['status']}] run {run_txt} "
            f"(snap {b['snapshotBankroll']}, settled {b['settledProfitSinceSnapshot']:+.2f}, "
            f"pending risk ${b['pendingRisk']:.2f}, {unit_txt})"
        )
    lines.append("")
    lines.append(f"Open ({summary['openCount']}):")
    for bet in summary["open"]:
        odds = bet.get("oddsAmerican")
        odds_txt = f"{odds:+d}" if isinstance(odds, int) else "odds TBD"
        lines.append(
            f"  {bet['id']} {bet['date']} {bet['book']} {bet['sport']} "
            f"{bet['selection']} ${bet['stakeDollars']:.2f} {odds_txt}"
        )
    lines.append(f"Settled ({summary['settledCount']}):")
    for bet in summary["settled"]:
        odds = bet.get("oddsAmerican")
        odds_txt = f"{odds:+d}" if isinstance(odds, int) else "odds TBD"
        lines.append(
            f"  {bet['id']} {bet['date']} {bet['book']} {bet['sport']} "
            f"{bet['selection']} ${bet['stakeDollars']:.2f} {odds_txt} "
            f"{bet['status']} {float(bet.get('profitDollars') or 0):+.2f}"
        )
    print("\n".join(lines))
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    ledger = load_ledger(path)
    books = book_map(ledger)
    key = args.book.lower()
    if key not in books:
        print(f"Unknown book '{args.book}'. Known: {sorted(books)}", file=sys.stderr)
        return 1
    entry = books[key]
    if entry.get("status") == "out":
        print(f"Book '{args.book}' is OUT. Ticket not added.", file=sys.stderr)
        return 1
    if entry.get("status") == "hold" and not args.allow_hold:
        print(
            f"Book '{args.book}' is on HOLD. Re-run with --allow-hold to record anyway.",
            file=sys.stderr,
        )
        return 1
    if args.odds_unknown and args.odds is not None:
        print("Pass either --odds or --odds-unknown, not both.", file=sys.stderr)
        return 1
    if not args.odds_unknown and args.odds is None:
        print("Odds required. Use --odds-unknown only when the ticket is truly unknown.", file=sys.stderr)
        return 1
    if args.stake <= 0:
        print("Stake must be positive.", file=sys.stderr)
        return 1
    unit = entry.get("unitDollars")
    units = round(args.stake / float(unit), 2) if unit else None
    bet_id = next_id(ledger, args.date)
    ledger.setdefault("bets", []).append(
        {
            "id": bet_id,
            "date": args.date,
            "book": entry["book"],
            "sport": args.sport,
            "selection": args.selection,
            "line": args.line,
            "oddsAmerican": args.odds,
            "oddsSource": None if args.odds_unknown else (args.odds_source or "harsh_ticket"),
            "stakeDollars": round(args.stake, 2),
            "units": units,
            "status": "pending",
            "profitDollars": 0.0,
            "legs": args.legs,
            "notes": args.notes or "",
        }
    )
    save_ledger(path, ledger)
    print(f"Added {bet_id} as pending on {entry['book']}.")
    return 0


def cmd_settle(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    ledger = load_ledger(path)
    target = None
    for bet in ledger.get("bets", []):
        if bet.get("id") == args.id:
            target = bet
            break
    if target is None:
        print(f"Bet '{args.id}' not found.", file=sys.stderr)
        return 1
    if args.result not in SETTLE_RESULTS:
        print(f"Result must be one of {SETTLE_RESULTS}.", file=sys.stderr)
        return 1
    odds = target.get("oddsAmerican")
    if args.odds is not None:
        odds = args.odds
        target["oddsAmerican"] = odds
        target["oddsSource"] = args.odds_source or "harsh_ticket"
    if odds is None and args.result in ("win", "loss"):
        print(
            "Cannot settle win/loss without odds. Provide --odds from the ticket.",
            file=sys.stderr,
        )
        return 1
    if args.result in ("win", "loss") and not isinstance(odds, int):
        print("Odds must be an American integer like -110 or +120.", file=sys.stderr)
        return 1
    stake = float(target.get("stakeDollars") or 0.0)
    profit = american_profit(stake, int(odds or 0), args.result) if args.result in ("win", "loss") else 0.0
    target["status"] = args.result
    target["profitDollars"] = profit
    if args.notes:
        prior = str(target.get("notes") or "").strip()
        target["notes"] = (prior + " " + args.notes).strip() if prior else args.notes
    save_ledger(path, ledger)
    print(f"Settled {args.id} as {args.result} ({profit:+.2f}).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harsh Dave personal betting ledger.")
    parser.add_argument("--ledger", default="data/personal_ledger.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_summary = sub.add_parser("summary", help="Print per-book bankroll plus open and settled tickets.")
    p_summary.add_argument("--json", action="store_true")
    p_summary.set_defaults(func=cmd_summary)

    p_add = sub.add_parser("add", help="Add a pending ticket. Blocked for OUT books, gated for HOLD.")
    p_add.add_argument("--date", required=True, help="YYYY-MM-DD slate date")
    p_add.add_argument("--book", required=True)
    p_add.add_argument("--sport", required=True)
    p_add.add_argument("--selection", required=True)
    p_add.add_argument("--line", default=None)
    p_add.add_argument("--odds", type=int, default=None)
    p_add.add_argument("--odds-unknown", action="store_true")
    p_add.add_argument("--odds-source", default=None)
    p_add.add_argument("--stake", type=float, required=True)
    p_add.add_argument("--legs", type=int, default=1)
    p_add.add_argument("--notes", default="")
    p_add.add_argument("--allow-hold", action="store_true")
    p_add.set_defaults(func=cmd_add)

    p_settle = sub.add_parser("settle", help="Settle a ticket. Wins and losses require confirmed odds.")
    p_settle.add_argument("--id", required=True)
    p_settle.add_argument("--result", required=True, choices=list(SETTLE_RESULTS))
    p_settle.add_argument("--odds", type=int, default=None)
    p_settle.add_argument("--odds-source", default=None)
    p_settle.add_argument("--notes", default="")
    p_settle.set_defaults(func=cmd_settle)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Scheduled bet brief generator for Harsh Dave (PickLedger climb filter).

Reads Profit Desk (never invents odds, edges, or P&L) and applies Harsh's
gatekeeping rules to produce max 1-2 BET THIS per day plus explicit PASS list.

Rules (see docs/BET_BRIEF.md):
  - NO tennis ever.
  - Singles only. Never parlays.
  - Skip juice past about -200 unless huge +EV (conservative EV above 0.05
    with probability of positive EV at least 0.80).
  - Profit Desk qualified picks (EDGE, VALUE) come first, max 2.
  - When Profit Desk qualifies zero (valid sit out), fall back to max 1 climb
    single: the top decision BET by sport priority then cleanest price.
    LEAN-only slates sit out rather than forcing action.
  - Novig first for sizing. ReBet is HOLD, Fliff is OUT.
  - Never auto-places bets. Harsh confirms the book price before betting.

Usage:
  python3 scripts/generate_bet_brief.py --slot am
  python3 scripts/generate_bet_brief.py --slot pm --date 2026-09-20
  python3 scripts/generate_bet_brief.py --slot manual --stdout
  python3 scripts/generate_bet_brief.py --slot am --as-of 2026-09-20T14:00:00Z
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

SPORT_PRIORITY = {
    "NFL": 0,
    "CFB": 1,
    "MLB": 2,
    "WNBA": 3,
    "MLS": 4,
    "NBA": 5,
    "NBA_SUMMER": 6,
    "FIFA_WORLD_CUP": 6,
    "FIFA WC": 6,
    "IPL": 6,
}

CT = ZoneInfo("America/Chicago")

JUICE_CAP = -200
HUGE_CONSERVATIVE_EV = 0.05
HUGE_PROB_POSITIVE_EV = 0.80
PROVEN_NEGATIVE_MIN_SAMPLES = 50
PROVEN_NEGATIVE_MAX_ROI = -0.05
MIN_PREGAME_MINUTES = 10


def american_break_even(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def fmt_odds(odds) -> str:
    if odds is None:
        return "odds TBD"
    try:
        o = int(odds)
    except (TypeError, ValueError):
        return "odds TBD"
    return f"{o:+d}"


def is_tennis(candidate: dict) -> bool:
    sport = str(candidate.get("sport") or "").lower()
    source = str(candidate.get("source") or "").lower()
    source_key = str(candidate.get("sourceKey") or "").lower()
    return "tennis" in sport or "tennis" in source or "tennis" in source_key


def price_info(candidate: dict) -> dict:
    price = candidate.get("price") or {}
    return {
        "source": price.get("source"),
        "timestamp": price.get("timestamp") or price.get("updatedAt"),
        "tier": price.get("tier"),
        "tierLabel": price.get("tierLabel"),
        "freshPregame": price.get("freshPregame"),
        "observedExecutable": price.get("observedExecutable"),
        "breakEven": price.get("breakEvenProbability"),
        "startTime": price.get("startTime"),
    }


def parse_iso(value) -> dt.datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def started_exclusion_reason(candidate: dict, as_of: dt.datetime) -> str | None:
    start = parse_iso((candidate.get("price") or {}).get("startTime"))
    if start is None:
        return None
    cutoff = start - dt.timedelta(minutes=MIN_PREGAME_MINUTES)
    if as_of >= cutoff:
        return f"Off the board: starts {start.strftime('%Y-%m-%d %H:%M UTC')} (needs {MIN_PREGAME_MINUTES} min pregame)."
    return None


def proven_negative_reason(candidate: dict, sources: dict) -> str | None:
    row = sources.get(str(candidate.get("source")))
    if not isinstance(row, dict):
        return None
    try:
        samples = int(row.get("samples") or 0)
        roi = float(row.get("flatRoi") or 0.0)
    except (TypeError, ValueError):
        return None
    if samples >= PROVEN_NEGATIVE_MIN_SAMPLES and roi < PROVEN_NEGATIVE_MAX_ROI:
        return (
            f"Source {candidate.get('source')} is {roi:+.1%} flat ROI over {samples} rows, "
            "proven negative, so no climb bet from this source."
        )
    return None


def estimate_numbers(candidate: dict) -> dict:
    est = candidate.get("estimate") or {}
    val = est.get("value") or {}
    return {
        "ev": est.get("expectedValue"),
        "conservativeEv": est.get("conservativeExpectedValue"),
        "probPositiveEv": est.get("probabilityPositiveEv"),
        "valueEv": val.get("expectedValue"),
        "valueConservativeEv": val.get("conservativeExpectedValue"),
        "valueProbPositiveEv": val.get("probabilityPositiveEv"),
    }


def is_huge_plus_ev(candidate: dict) -> bool:
    nums = estimate_numbers(candidate)
    pairs = [
        (nums["conservativeEv"], nums["probPositiveEv"]),
        (nums["valueConservativeEv"], nums["valueProbPositiveEv"]),
    ]
    for cev, prob in pairs:
        if isinstance(cev, (int, float)) and isinstance(prob, (int, float)):
            if cev > HUGE_CONSERVATIVE_EV and prob >= HUGE_PROB_POSITIVE_EV:
                return True
    return False


def juice_excluded(candidate: dict) -> bool:
    odds = candidate.get("oddsAmerican")
    if odds is None:
        return False
    try:
        o = int(odds)
    except (TypeError, ValueError):
        return False
    if o <= JUICE_CAP and not is_huge_plus_ev(candidate):
        return True
    return False


def hard_exclusion_reason(candidate: dict) -> str | None:
    if is_tennis(candidate):
        return "NO tennis ever (Hard exclusion)."
    tier = str(candidate.get("tier") or "").lower()
    if tier == "avoid":
        return "Tier AVOID in Profit Desk (structural or price block)."
    price = candidate.get("price") or {}
    if price.get("freshPregame") is False:
        return "Price is not fresh pregame, so it cannot be trusted for a ticket."
    if price.get("observedExecutable") is False:
        return "No observed executable price, so no bet."
    if candidate.get("oddsAmerican") is None:
        return "No posted odds, so no bet (odds are never guessed)."
    if juice_excluded(candidate):
        return f"Juice {fmt_odds(candidate.get('oddsAmerican'))} is past the -200 cap without huge +EV."
    return None


def sport_rank(candidate: dict) -> int:
    sport = str(candidate.get("sport") or "").upper().strip()
    if sport in SPORT_PRIORITY:
        return SPORT_PRIORITY[sport]
    key = sport.replace(" ", "_")
    return SPORT_PRIORITY.get(key, 6)


def price_cleanliness(candidate: dict) -> float:
    price = candidate.get("price") or {}
    be = price.get("breakEvenProbability")
    if not isinstance(be, (int, float)):
        odds = candidate.get("oddsAmerican")
        try:
            be = american_break_even(int(odds))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 1.0
    return abs(float(be) - american_break_even(-110))


def decision_rank(candidate: dict) -> int:
    d = str(candidate.get("decision") or "").upper()
    return {"BET": 0, "LEAN": 1}.get(d, 2)


def lane_rank(candidate: dict) -> int:
    lane = str(candidate.get("lane") or "").lower()
    return {"edge": 0, "value": 1}.get(lane, 2)


def sort_key(candidate: dict) -> tuple:
    return (
        lane_rank(candidate),
        decision_rank(candidate),
        sport_rank(candidate),
        price_cleanliness(candidate),
    )


def source_lookup(profit: dict) -> dict:
    out = {}
    for row in profit.get("sources", []) or []:
        out[str(row.get("source"))] = row
        key = row.get("sourceKey")
        if key:
            out.setdefault(str(key), row)
    return out


def _desk_context_suffix(candidate: dict, sources: dict) -> str:
    parts = []
    decision = str(candidate.get("decision") or "n/a")
    tier = str(candidate.get("tier") or "n/a")
    parts.append(f"Profit Desk decision {decision}, tier {tier} (not qualified).")
    src = sources.get(str(candidate.get("source")))
    if isinstance(src, dict):
        roi = src.get("flatRoi")
        samples = src.get("samples")
        dates = src.get("distinctDates")
        prob = src.get("probabilityPositiveEv")
        if roi is not None and samples is not None:
            try:
                parts.append(
                    f"Source {candidate.get('source')} is {float(roi):+.1%} flat ROI "
                    f"over {int(samples)} rows and {int(dates or 0)} dates "
                    f"(Prob+EV {float(prob or 0):.2f})."
                )
            except (TypeError, ValueError):
                pass
    blockers = candidate.get("blockers") or []
    names = []
    for b in blockers[:3]:
        if isinstance(b, dict):
            names.append(str(b.get("code") or b.get("name") or b))
        else:
            names.append(str(b))
    if names:
        parts.append("Top blockers: " + ", ".join(names) + ".")
    return " ".join(parts)


def pass_reason(candidate: dict, sources: dict, as_of: dt.datetime | None = None) -> str:
    hard = hard_exclusion_reason(candidate)
    if hard:
        return hard
    if as_of is not None:
        started = started_exclusion_reason(candidate, as_of)
        if started:
            return started + " " + _desk_context_suffix(candidate, sources)
    proven = proven_negative_reason(candidate, sources)
    if proven:
        return proven + " " + _desk_context_suffix(candidate, sources)
    return _desk_context_suffix(candidate, sources)


def build_bet_entry(candidate: dict, rank: int, qualified: bool) -> dict:
    p = price_info(candidate)
    nums = estimate_numbers(candidate)
    lane = candidate.get("lane")
    return {
        "rank": rank,
        "qualified": qualified,
        "lane": lane,
        "sport": candidate.get("sport"),
        "source": candidate.get("source"),
        "market": candidate.get("market"),
        "pick": candidate.get("pick"),
        "game": candidate.get("game"),
        "line": candidate.get("line"),
        "oddsAmerican": candidate.get("oddsAmerican"),
        "decision": candidate.get("decision"),
        "tier": candidate.get("tier"),
        "price": p,
        "estimate": {
            "expectedValue": nums["ev"],
            "conservativeExpectedValue": nums["conservativeEv"],
            "probabilityPositiveEv": nums["probPositiveEv"],
        },
        "stakeGuide": {
            "Novig": "1u (about $1). Confirm the Novig price matches before betting.",
            "Onyx": "1u (about $0.50) only if Harsh prefers Onyx for this ticket.",
            "ReBet": "HOLD, no bet.",
            "Fliff": "OUT, no bet.",
        },
        "why": "",
        "risks": "",
    }


def why_text(candidate: dict, qualified: bool, sources: dict) -> str:
    base = (
        f"{candidate.get('pick')} at {fmt_odds(candidate.get('oddsAmerican'))} "
        f"({candidate.get('source')}, {candidate.get('sport')}). "
        f"Profit Desk decision {candidate.get('decision')}, tier {candidate.get('tier')}."
    )
    if qualified:
        lane = str(candidate.get("lane") or "").upper()
        stake = candidate.get("stakeUnits")
        return base + f" Qualified {lane} at {stake}u in Profit Desk. Single only."
    src = sources.get(str(candidate.get("source")))
    extra = ""
    if isinstance(src, dict):
        try:
            extra = (
                f" Source context: {int(src.get('wins') or 0)}-{int(src.get('losses') or 0)} "
                f"all time in desk evidence, {float(src.get('flatRoi') or 0):+.1%} flat ROI."
            )
        except (TypeError, ValueError):
            extra = ""
    return (
        base
        + " Profit Desk qualified zero today, so this is a gatekept climb single, "
        + "not an EDGE or VALUE qualifier. Chosen as the top BET by sport priority "
        + "and cleanest price. Single only."
        + extra
    )


def risks_text(candidate: dict) -> str:
    p = price_info(candidate)
    nums = estimate_numbers(candidate)
    bits = []
    bits.append(
        f"Posted price is {fmt_odds(candidate.get('oddsAmerican'))} via {p['source'] or 'posted market'} "
        f"at {p['timestamp'] or 'unknown time'}. If Novig differs, pass or resize."
    )
    if isinstance(nums["conservativeEv"], (int, float)):
        bits.append(f"Desk conservative EV {float(nums['conservativeEv']):+.3f}.")
    blockers = candidate.get("blockers") or []
    if blockers:
        bits.append(f"Desk lists {len(blockers)} qualification blockers; see PASS detail for names.")
    bits.append("Climb sizing stays 1u. No parlay, no chase.")
    return " ".join(bits)


def load_ledger_snapshot(path: Path) -> dict:
    try:
        ledger = json.loads(path.read_text())
    except FileNotFoundError:
        return {"available": False, "note": "Ledger file not found."}
    except json.JSONDecodeError:
        return {"available": False, "note": "Ledger file is not valid JSON."}
    books_out = []
    for entry in ledger.get("books", []) or []:
        name = entry.get("book")
        snap_date = entry.get("snapshotDate")
        settled = 0.0
        pending = 0.0
        for bet in ledger.get("bets", []) or []:
            if str(bet.get("book", "")).lower() != str(name or "").lower():
                continue
            if bet.get("status") == "pending":
                pending += float(bet.get("stakeDollars") or 0.0)
            else:
                if snap_date and str(bet.get("date", "")) < str(snap_date):
                    continue
                settled += float(bet.get("profitDollars") or 0.0)
        settled = round(settled, 2)
        pending = round(pending, 2)
        snap = entry.get("snapshotBankroll")
        running = round(float(snap) + settled, 2) if snap is not None else None
        books_out.append(
            {
                "book": name,
                "status": entry.get("status"),
                "runningBankroll": running,
                "settledProfitSinceSnapshot": settled,
                "pendingRisk": pending,
                "unitDollars": entry.get("unitDollars"),
            }
        )
    open_bets = [b for b in ledger.get("bets", []) or [] if b.get("status") == "pending"]
    return {
        "available": True,
        "updatedAt": ledger.get("updatedAt"),
        "books": books_out,
        "open": [
            {
                "id": b.get("id"),
                "date": b.get("date"),
                "book": b.get("book"),
                "sport": b.get("sport"),
                "selection": b.get("selection"),
                "stakeDollars": b.get("stakeDollars"),
                "oddsAmerican": b.get("oddsAmerican"),
            }
            for b in open_bets
        ],
    }


def generate(profit: dict, ledger_path: Path, slot: str, as_of: dt.datetime) -> dict:
    date = str(profit.get("date") or dt.date.today().isoformat())
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    candidates = list(profit.get("candidates") or [])
    sources = source_lookup(profit)
    summary = profit.get("summary") or {}
    live_to_date = summary.get("liveRecordToDate") or {}

    qualified = [c for c in candidates if c.get("lane") in ("edge", "value")]
    qualified_sorted = sorted(qualified, key=sort_key)

    bet_this: list[dict] = []
    if qualified_sorted:
        for i, cand in enumerate(qualified_sorted[:2], start=1):
            entry = build_bet_entry(cand, i, True)
            entry["why"] = why_text(cand, True, sources)
            entry["risks"] = risks_text(cand)
            bet_this.append(entry)
    else:
        eligible = []
        for c in candidates:
            if hard_exclusion_reason(c) is not None:
                continue
            if started_exclusion_reason(c, as_of) is not None:
                continue
            if proven_negative_reason(c, sources) is not None:
                continue
            eligible.append(c)
        bets_only = [c for c in eligible if str(c.get("decision") or "").upper() == "BET"]
        bets_sorted = sorted(bets_only, key=sort_key)
        if bets_sorted:
            top = bets_sorted[0]
            entry = build_bet_entry(top, 1, False)
            entry["why"] = why_text(top, False, sources)
            entry["risks"] = risks_text(top)
            bet_this.append(entry)

    selected_ids = {c.get("id") for c in qualified_sorted[:2]} if qualified_sorted else ({bet_this[0]["pick"]} if bet_this else set())
    # Match PASS list by candidate id when qualified, else by pick text for the climb single.
    pass_list = []
    for cand in sorted(candidates, key=sort_key):
        if qualified_sorted:
            if cand.get("id") in selected_ids:
                continue
        else:
            if bet_this and cand.get("pick") == bet_this[0]["pick"] and cand.get("oddsAmerican") == bet_this[0]["oddsAmerican"]:
                continue
        pass_list.append(
            {
                "sport": cand.get("sport"),
                "source": cand.get("source"),
                "pick": cand.get("pick"),
                "game": cand.get("game"),
                "line": cand.get("line"),
                "oddsAmerican": cand.get("oddsAmerican"),
                "decision": cand.get("decision"),
                "tier": cand.get("tier"),
                "reason": pass_reason(cand, sources, as_of),
            }
        )

    return {
        "schemaVersion": 1,
        "date": date,
        "slot": slot,
        "generatedAt": generated_at,
        "asOf": as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profitDesk": {
            "date": profit.get("date"),
            "generatedAt": profit.get("generatedAt"),
            "engineVersion": profit.get("engineVersion"),
            "candidateCount": summary.get("candidateCount"),
            "edgeQualified": summary.get("edgeQualified"),
            "valueQualified": summary.get("valueQualified"),
            "liveQualified": summary.get("liveQualified"),
            "liveRecordToDate": live_to_date,
            "notices": profit.get("notices") or [],
        },
        "betThis": bet_this,
        "pass": pass_list,
        "ledger": load_ledger_snapshot(ledger_path),
        "rules": {
            "maxBetThis": 2,
            "singlesOnly": True,
            "noTennis": True,
            "juiceCap": JUICE_CAP,
            "juiceException": "conservative EV above 0.05 with Prob+EV at least 0.80",
            "bookPriority": "Novig first. ReBet HOLD, Fliff OUT.",
        },
        "disclaimers": [
            "Prices are quoted from Profit Desk evidence (book posted lines). Confirm the Novig ticket before betting; if the price moved, pass.",
            "Climb singles are not Profit Desk EDGE or VALUE qualifiers unless marked qualified.",
            "Nothing here places bets. Harsh places every ticket manually.",
        ],
    }


def brief_markdown(brief: dict) -> str:
    date = brief.get("date")
    slot = brief.get("slot")
    pd = brief.get("profitDesk") or {}
    live = pd.get("liveRecordToDate") or {}
    lines = [f"# Bet brief {date} ({slot})", ""]
    if brief.get("asOf"):
        lines.append(f"As of {brief.get('asOf')} (started games are off the board).")
        lines.append("")
    lines.append(
        f"Profit Desk {pd.get('date')}: {pd.get('candidateCount')} candidates, "
        f"{pd.get('liveQualified')} qualified (EDGE {pd.get('edgeQualified')}, VALUE {pd.get('valueQualified')})."
    )
    if live:
        lines.append(
            f"Desk live record to date: {live.get('wins')}-{live.get('losses')} "
            f"(pushes {live.get('pushes')}), {float(live.get('netUnits') or 0):+.2f}u, "
            f"ROI {float(live.get('roi') or 0):+.1%} over {live.get('settled')} settled."
        )
    lines.append("")
    bets = brief.get("betThis") or []
    if bets:
        lines.append(f"## BET THIS ({len(bets)})")
        for b in bets:
            tag = "QUALIFIED " + str(b.get("lane") or "").upper() if b.get("qualified") else "CLIMB SINGLE (not qualified)"
            lines.append(f"### {b.get('rank')}. {b.get('pick')} {fmt_odds(b.get('oddsAmerican'))} [{tag}]")
            lines.append(f"Sport {b.get('sport')} via {b.get('source')}. Desk {b.get('decision')}/{b.get('tier')}.")
            lines.append(f"Why: {b.get('why')}")
            lines.append(f"Risks: {b.get('risks')}")
            lines.append(f"Stake: Novig {b.get('stakeGuide', {}).get('Novig')}")
            lines.append("")
    else:
        lines.append("## BET THIS (0)")
        lines.append("Sit out. No Profit Desk qualifier and no gatekept BET survived the climb filter.")
        lines.append("")
    lines.append("## PASS")
    for p in brief.get("pass") or []:
        lines.append(f"- {p.get('pick')} {fmt_odds(p.get('oddsAmerican'))} ({p.get('sport')}, {p.get('source')}): {p.get('reason')}")
    lines.append("")
    ledger = brief.get("ledger") or {}
    if ledger.get("available"):
        lines.append("## Ledger snapshot")
        for b in ledger.get("books") or []:
            run = b.get("runningBankroll")
            run_txt = f"${run:.2f}" if run is not None else "n/a"
            lines.append(
                f"- {b.get('book')} [{b.get('status')}]: run {run_txt}, "
                f"settled {float(b.get('settledProfitSinceSnapshot') or 0):+.2f}, "
                f"pending risk ${float(b.get('pendingRisk') or 0):.2f}"
            )
        if ledger.get("open"):
            lines.append("Open tickets:")
            for t in ledger["open"]:
                lines.append(
                    f"  - {t.get('id')} {t.get('date')} {t.get('book')} {t.get('sport')} "
                    f"{t.get('selection')} ${float(t.get('stakeDollars') or 0):.2f} {fmt_odds(t.get('oddsAmerican'))}"
                )
        lines.append("")
    for d in brief.get("disclaimers") or []:
        lines.append(f"Note: {d}")
    lines.append("")
    return "\n".join(lines)


def auto_slot(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(CT)
    return "am" if now.hour < 12 else "pm"


def update_index(out_dir: Path) -> None:
    files = sorted(p.name for p in out_dir.glob("2*.json") if p.name != "index.json" and p.name != "latest.json")
    (out_dir / "index.json").write_text(json.dumps({"files": files}, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Harsh's gatekept bet brief.")
    parser.add_argument("--profit-desk", default="data/profit_desk/latest.json")
    parser.add_argument("--ledger", default="data/personal_ledger.json")
    parser.add_argument("--out-dir", default="data/bet_briefs")
    parser.add_argument("--date", default=None)
    parser.add_argument("--slot", default="auto", choices=["auto", "am", "pm", "manual"])
    parser.add_argument("--as-of", default=None, help="ISO UTC cutoff for started-game checks (default now).")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing files.")
    parser.add_argument("--markdown-stdout", action="store_true", help="Print Markdown to stdout instead of writing files.")
    args = parser.parse_args(argv)

    profit_path = Path(args.profit_desk)
    try:
        profit = json.loads(profit_path.read_text())
    except FileNotFoundError:
        print(f"Profit Desk file not found: {profit_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Profit Desk file is not valid JSON: {exc}", file=sys.stderr)
        return 1

    if args.date:
        profit = dict(profit)
        profit["date"] = args.date
    slot = auto_slot() if args.slot == "auto" else args.slot
    if args.as_of:
        as_of = parse_iso(args.as_of)
        if as_of is None:
            print(f"Invalid --as-of timestamp: {args.as_of}", file=sys.stderr)
            return 1
    else:
        as_of = dt.datetime.now(dt.timezone.utc)
    brief = generate(profit, Path(args.ledger), slot, as_of)

    if args.stdout:
        print(json.dumps(brief, indent=2))
        return 0
    if args.markdown_stdout:
        print(brief_markdown(brief))
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{brief['date']}-{slot}"
    (out_dir / f"{base}.json").write_text(json.dumps(brief, indent=2) + "\n")
    (out_dir / f"{base}.md").write_text(brief_markdown(brief))
    (out_dir / "latest.json").write_text(json.dumps(brief, indent=2) + "\n")
    (out_dir / "latest.md").write_text(brief_markdown(brief))
    update_index(out_dir)
    print(f"Wrote {base} plus latest to {out_dir}. BET THIS count: {len(brief['betThis'])}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Archive NBA DraftKings player-prop prices and grade them into daily shards.

Two append-safe, per-day outputs feed the NBA player-prop consensus trainer:

* ``data/nba_prop_prices/<date>.jsonl.gz`` -- pregame over/under quotes
  captured by the refresh workflow before tip-off (ESPN's DraftKings feed).
  Each key keeps its first-seen and latest pregame quote.
* ``data/player_props_training/nba_market_history/<date>.jsonl.gz`` -- one
  graded row per offered over/under market for completed games.  Quotes come
  from ESPN's archived DraftKings markets, falling back to our own pregame
  capture when ESPN has no archive for the event.

Only quotes last updated before the scheduled tip are kept: ESPN's archive
also stores in-game prices, which would leak the result into training.
One-sided milestone ladders ("25+ points") are archived as
``market_format="milestone"`` (over-only, line = threshold - 0.5); the
trainer uses two-sided totals only.  Preseason rows are stamped with
``season_type`` so the trainer can exclude them.  Every shard is a small,
deterministic gzip file, so no committed file approaches hosting limits.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from player_props.api import DirectApiClient  # noqa: E402
from player_props.basketball import BASKETBALL_MARKET_TYPES, event_season_type  # noqa: E402
from player_props.schema import american_implied_probability, safe_float  # noqa: E402


PRICE_DIR = REPO_ROOT / "data" / "nba_prop_prices"
HISTORY_DIR = REPO_ROOT / "data" / "player_props_training" / "nba_market_history"
LEAGUE = "nba"
DEFAULT_PROVIDER_ID = "100"
DEFAULT_PROVIDER_NAME = "DraftKings"
SLATE_TZ = ZoneInfo("America/Chicago")
# Defensive cap far below the repository's 50 MB per-file guard.
MAX_SHARD_BYTES = 20_000_000

# ESPN names NBA totals "Total Points and Rebounds", "Total 3-Point Field
# Goals", ...; the live parser's map predates those names, so the archive keeps
# its own aliases instead of changing live NBA/WNBA parsing.
NBA_STAT_ALIASES: dict[str, str] = {
    "totalpoints": "points",
    "totalrebounds": "totalRebounds",
    "totalassists": "assists",
    "total3pointfieldgoals": "three_pointers_made",
    "totalthreepointfieldgoals": "three_pointers_made",
    "total3pointfieldgoalsmade": "three_pointers_made",
    "totalpointsandrebounds": "points_rebounds",
    "totalpointsandassists": "points_assists",
    "totalpointsreboundsandassists": "points_rebounds_assists",
    "totalpointsassistsandrebounds": "points_rebounds_assists",
    "totalassistsandrebounds": "rebounds_assists",
    "totalreboundsandassists": "rebounds_assists",
    "totalsteals": "steals",
    "totalblocks": "blocks",
    "totalstealsandblocks": "steals_blocks",
}
NBA_MILESTONE_ALIASES: dict[str, str] = {
    "pointsmilestones": "points",
    "reboundsmilestones": "totalRebounds",
    "assistsmilestones": "assists",
    "3pointfieldgoalsmilestones": "three_pointers_made",
    "threepointfieldgoalsmilestones": "three_pointers_made",
    "pointsreboundsmilestones": "points_rebounds",
    "pointsassistsmilestones": "points_assists",
    "pointsassistsreboundsmilestones": "points_rebounds_assists",
    "pointsreboundsassistsmilestones": "points_rebounds_assists",
    "assistsreboundsmilestones": "rebounds_assists",
    "reboundsassistsmilestones": "rebounds_assists",
    "stealsmilestones": "steals",
    "blocksmilestones": "blocks",
}


def canonical(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def stat_key_for(type_name: str) -> str | None:
    """Stat for a full-game two-sided total; None for anything else."""
    name = canonical(type_name)
    if "milestone" in name or "quarter" in name or "half" in name:
        return None
    if name in NBA_STAT_ALIASES:
        return NBA_STAT_ALIASES[name]
    mapped = BASKETBALL_MARKET_TYPES.get(name)
    return mapped[0] if mapped else None


def milestone_stat_key_for(type_name: str) -> str | None:
    """Stat for a full-game one-sided milestone ladder; None otherwise."""
    name = canonical(type_name)
    if "quarter" in name or "half" in name:
        return None
    return NBA_MILESTONE_ALIASES.get(name)


def _american(value: Any) -> int | None:
    text = str(value or "").strip().upper()
    if text in {"EVEN", "EV"}:
        return 100
    try:
        number = int(float(text.replace("+", "")))
    except (TypeError, ValueError):
        return None
    if number == 0 or -100 < number < 100:
        return None
    return number


def _athlete_id(item: dict[str, Any]) -> str:
    ref = str(((item.get("athlete") or {}).get("$ref")) or "")
    if "/athletes/" not in ref:
        return str((item.get("athlete") or {}).get("id") or "")
    return ref.split("/athletes/", 1)[1].split("?", 1)[0].split("/", 1)[0]


def _line(item: dict[str, Any]) -> float | None:
    target = ((item.get("current") or {}).get("target") or {})
    raw = target.get("value")
    if raw is None:
        raw = ((item.get("odds") or {}).get("total") or {}).get("value")
    value = safe_float(str(raw).replace("+", ""), float("nan")) if raw is not None else float("nan")
    return value if math.isfinite(value) and value > 0 else None


def _open_line(item: dict[str, Any]) -> float | None:
    raw = ((item.get("open") or {}).get("target") or {}).get("value")
    value = safe_float(raw, float("nan")) if raw is not None else float("nan")
    return value if math.isfinite(value) and value > 0 else None


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def event_start(event: dict[str, Any]) -> str:
    return str(event.get("date") or ((event.get("competitions") or [{}])[0].get("date")) or "")


def event_state(event: dict[str, Any]) -> str:
    competition = (event.get("competitions") or [{}])[0]
    status = competition.get("status") or event.get("status") or {}
    return str((status.get("type") or {}).get("state") or "").lower()


def event_completed(event: dict[str, Any]) -> bool:
    competition = (event.get("competitions") or [{}])[0]
    status = competition.get("status") or event.get("status") or {}
    return (status.get("type") or {}).get("completed") is True


def event_season(event: dict[str, Any], slate_date: str) -> int:
    season = event.get("season") if isinstance(event.get("season"), dict) else {}
    try:
        return int(season.get("year"))
    except (TypeError, ValueError):
        return nba_season_for_date(slate_date)


def nba_season_for_date(date_iso: str) -> int:
    """ESPN labels an NBA season by the calendar year it ends in."""
    year, month = int(date_iso[:4]), int(date_iso[5:7])
    return year + 1 if month >= 8 else year


def _event_teams(event: dict[str, Any]) -> dict[str, str]:
    competition = (event.get("competitions") or [{}])[0]
    teams: dict[str, str] = {}
    for competitor in competition.get("competitors") or []:
        side = str(competitor.get("homeAway") or "")
        team_id = str((competitor.get("team") or {}).get("id") or competitor.get("id") or "")
        if side in {"home", "away"} and team_id:
            teams[f"{side}_team_id"] = team_id
    return teams


def parse_quotes(
    items: Iterable[dict[str, Any]],
    *,
    start_time: str,
    require_pregame: bool = True,
) -> list[dict[str, Any]]:
    """Pair ESPN over/under items into one quote per athlete, stat and line.

    ESPN lists the over before the under for each total.  Quotes updated at or
    after the scheduled tip are dropped when ``require_pregame`` is set.
    """
    tip = _parse_time(start_time)
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    milestones: list[tuple[dict[str, Any], str, str, float, str]] = []
    for item in items:
        type_name = str((item.get("type") or {}).get("name") or "")
        athlete_id = _athlete_id(item)
        line = _line(item)
        if not athlete_id or line is None:
            continue
        display = str(((item.get("current") or {}).get("target") or {}).get("displayValue") or "")
        milestone_key = milestone_stat_key_for(type_name)
        if milestone_key:
            milestones.append((item, athlete_id, milestone_key, line, type_name))
            continue
        stat_key = stat_key_for(type_name)
        if not stat_key or "+" in display:
            continue
        grouped[(athlete_id, stat_key, line, type_name)].append(item)
    quotes: list[dict[str, Any]] = []
    for item, athlete_id, stat_key, threshold, type_name in milestones:
        over_odds = _american(((item.get("odds") or {}).get("american") or {}).get("value"))
        updated = str(item.get("lastUpdated") or "")
        updated_at = _parse_time(updated)
        pregame = bool(tip and updated_at and updated_at < tip)
        over_implied = american_implied_probability(over_odds)
        if over_odds is None or over_implied is None or (require_pregame and not pregame):
            continue
        quotes.append({
            "athlete_id": athlete_id,
            "stat_key": stat_key,
            "market_type": type_name,
            "market_format": "milestone",
            "line": max(0.0, float(threshold) - 0.5),
            "over_odds": over_odds,
            "under_odds": None,
            "over_open_odds": _american(((item.get("odds") or {}).get("american") or {}).get("open")),
            "under_open_odds": None,
            "open_line": None,
            "over_implied": round(over_implied, 6),
            "under_implied": None,
            "no_vig_over": round(over_implied, 6),
            "market_updated_at": updated,
            "pregame": pregame,
        })
    for (athlete_id, stat_key, line, type_name), sides in grouped.items():
        if len(sides) != 2:
            continue
        over, under = sides
        over_odds = _american(((over.get("odds") or {}).get("american") or {}).get("value"))
        under_odds = _american(((under.get("odds") or {}).get("american") or {}).get("value"))
        if over_odds is None or under_odds is None:
            continue
        updated = max(str(over.get("lastUpdated") or ""), str(under.get("lastUpdated") or ""))
        updated_at = _parse_time(updated)
        pregame = bool(tip and updated_at and updated_at < tip)
        if require_pregame and not pregame:
            continue
        over_implied = american_implied_probability(over_odds)
        under_implied = american_implied_probability(under_odds)
        if over_implied is None or under_implied is None or over_implied + under_implied <= 0:
            continue
        quotes.append({
            "athlete_id": athlete_id,
            "stat_key": stat_key,
            "market_type": type_name,
            "market_format": "total",
            "line": float(line),
            "over_odds": over_odds,
            "under_odds": under_odds,
            "over_open_odds": _american(((over.get("odds") or {}).get("american") or {}).get("open")),
            "under_open_odds": _american(((under.get("odds") or {}).get("american") or {}).get("open")),
            "open_line": _open_line(over),
            "over_implied": round(over_implied, 6),
            "under_implied": round(under_implied, 6),
            "no_vig_over": round(over_implied / (over_implied + under_implied), 6),
            "market_updated_at": updated,
            "pregame": pregame,
        })
    return quotes


def basketball_actuals(summary: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Final box-score values for every NBA stat the archive can grade."""
    result: dict[tuple[str, str], float] = {}
    boxscore = summary.get("boxscore") or {}
    for team in boxscore.get("players") or []:
        for category in team.get("statistics") or []:
            keys = [str(key) for key in category.get("keys") or []]
            for row in category.get("athletes") or []:
                if row.get("didNotPlay") is True:
                    continue
                athlete_id = str((row.get("athlete") or {}).get("id") or "")
                stats = row.get("stats") or []
                if not athlete_id or not keys or len(stats) < len(keys):
                    continue
                values: dict[str, float | None] = {}
                for index, key in enumerate(keys):
                    text = str(stats[index] or "").strip()
                    if "-" in text and not text.startswith("-"):
                        text = text.split("-", 1)[0]
                    number = safe_float(text, float("nan")) if text not in {"", "--"} else float("nan")
                    values[key] = number if math.isfinite(number) else None
                minutes = values.get("minutes")
                if minutes is not None and minutes <= 0:
                    continue
                points = values.get("points")
                rebounds = values.get("rebounds")
                assists = values.get("assists")
                steals = values.get("steals")
                blocks = values.get("blocks")

                def total(*parts: float | None) -> float | None:
                    return float(sum(parts)) if all(part is not None for part in parts) else None

                aliases = {
                    "points": points,
                    "totalRebounds": rebounds,
                    "assists": assists,
                    "three_pointers_made": values.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"),
                    "steals": steals,
                    "blocks": blocks,
                    "points_rebounds": total(points, rebounds),
                    "points_assists": total(points, assists),
                    "rebounds_assists": total(rebounds, assists),
                    "points_rebounds_assists": total(points, rebounds, assists),
                    "steals_blocks": total(steals, blocks),
                }
                for stat_key, actual in aliases.items():
                    if actual is not None:
                        result[(athlete_id, stat_key)] = float(actual)
    return result


def _event_context(event: dict[str, Any], slate_date: str) -> dict[str, Any]:
    return {
        "sport": "NBA",
        "season": event_season(event, slate_date),
        "season_type": event_season_type(event),
        "date": slate_date,
        "start_time": event_start(event),
        "event_id": str(event.get("id") or ""),
        **_event_teams(event),
    }


def graded_rows(
    event: dict[str, Any],
    slate_date: str,
    quotes: list[dict[str, Any]],
    actuals: dict[tuple[str, str], float],
    *,
    provider: str,
    provenance: str,
) -> list[dict[str, Any]]:
    context = _event_context(event, slate_date)
    rows: list[dict[str, Any]] = []
    for quote in quotes:
        actual = actuals.get((str(quote.get("athlete_id")), str(quote.get("stat_key"))))
        line = safe_float(quote.get("line"), float("nan"))
        if actual is None or not math.isfinite(line) or actual == line:
            continue
        rows.append({
            **context,
            **{key: quote.get(key) for key in (
                "athlete_id", "stat_key", "market_type", "market_format", "line", "over_odds", "under_odds",
                "over_implied", "under_implied", "no_vig_over", "market_updated_at", "open_line",
            )},
            "actual": float(actual),
            "over_outcome": int(actual > line),
            "provider": provider,
            "provenance": provenance,
        })
    return rows


# ---------------------------------------------------------------- shard I/O

def shard_path(base: Path, date_iso: str) -> Path:
    return base / f"{date_iso}.jsonl.gz"


def read_shard(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_shards(base: Path, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for path in sorted(base.glob("*.jsonl.gz")):
        day = path.name.split(".", 1)[0]
        if (start and day < start) or (end and day > end):
            continue
        rows.extend(read_shard(path))
    return rows


def write_shard(path: Path, rows: list[dict[str, Any]]) -> bool:
    """Write a deterministic gzip shard; return False when content is unchanged."""
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("event_id") or ""), str(row.get("athlete_id") or ""), str(row.get("stat_key") or ""),
            str(row.get("market_format") or ""), safe_float(row.get("line")), str(row.get("market_updated_at") or ""),
        ),
    )
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as compressed:
        compressed.write(text.encode("utf-8"))
    payload = buffer.getvalue()
    if len(payload) > MAX_SHARD_BYTES:
        raise ValueError(f"{path} would be {len(payload)} bytes (> {MAX_SHARD_BYTES})")
    if path.is_file() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(payload)
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)
    return True


def merge_capture(existing: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first-seen and latest pregame quote for each market key."""
    merged: dict[tuple[str, str, str, float, str], dict[str, Any]] = {}
    for row in [*existing, *fresh]:
        key = (
            str(row.get("event_id") or ""), str(row.get("athlete_id") or ""),
            str(row.get("stat_key") or ""), safe_float(row.get("line")), str(row.get("market_format") or "total"),
        )
        previous = merged.get(key)
        if previous is None:
            merged[key] = dict(row)
            merged[key].setdefault("first_retrieved_at", row.get("retrieved_at"))
            continue
        first = min(
            str(previous.get("first_retrieved_at") or previous.get("retrieved_at") or ""),
            str(row.get("first_retrieved_at") or row.get("retrieved_at") or ""),
        )
        latest = row if str(row.get("retrieved_at") or "") >= str(previous.get("retrieved_at") or "") else previous
        merged[key] = {**latest, "first_retrieved_at": first}
    return list(merged.values())


def merge_history(existing: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, float, str], dict[str, Any]] = {}
    for row in [*existing, *fresh]:
        key = (
            str(row.get("event_id") or ""), str(row.get("athlete_id") or ""), str(row.get("stat_key") or ""),
            safe_float(row.get("line")), str(row.get("market_format") or "total"),
        )
        merged[key] = row
    return list(merged.values())


# ------------------------------------------------------------- network work

def _scoreboard(client: Any, date_iso: str) -> list[dict[str, Any]]:
    payload = client.basketball_scoreboard(LEAGUE, date_iso)
    return [event for event in payload.get("events") or [] if isinstance(event, dict)]


def _prop_items(client: Any, event_id: str) -> list[dict[str, Any]] | None:
    try:
        payload = client.basketball_espn_prop_bets(LEAGUE, event_id, DEFAULT_PROVIDER_ID)
    except RuntimeError as exc:
        if "404" in str(exc):
            return None
        raise
    items = payload.get("items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def capture_date(client: Any, date_iso: str, *, now: datetime | None = None, price_dir: Path = PRICE_DIR) -> dict[str, Any]:
    """Archive pregame quotes for every not-yet-started event on ``date_iso``."""
    moment = now or datetime.now(timezone.utc)
    events = _scoreboard(client, date_iso)
    fresh: list[dict[str, Any]] = []
    failures: list[str] = []
    captured_events = 0
    for event in events:
        start = event_start(event)
        tip = _parse_time(start)
        if event_state(event) not in {"", "pre"} or (tip and tip <= moment):
            continue
        event_id = str(event.get("id") or "")
        try:
            items = _prop_items(client, event_id)
        except Exception as exc:  # one bad event must not drop the slate
            failures.append(f"{event_id}: {exc}")
            continue
        if not items:
            continue
        quotes = parse_quotes(items, start_time=start)
        # A capture is pregame by construction; still enforce it per quote.
        quotes = [quote for quote in quotes if quote["pregame"]]
        if quotes:
            captured_events += 1
        context = _event_context(event, date_iso)
        for quote in quotes:
            fresh.append({**context, **quote, "provider": DEFAULT_PROVIDER_NAME, "retrieved_at": _iso(moment)})
    path = shard_path(price_dir, date_iso)
    changed = False
    if fresh:
        changed = write_shard(path, merge_capture(read_shard(path), fresh))
    return {
        "date": date_iso, "events": len(events), "captured_events": captured_events,
        "quotes": len(fresh), "changed": changed, "failures": failures, "path": str(path),
    }


def _grade_event(event: dict[str, Any], date_iso: str, capture_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    client = DirectApiClient(timeout=25.0, attempts=3)
    event_id = str(event.get("id") or "")
    items = _prop_items(client, event_id)
    quotes = parse_quotes(items or [], start_time=event_start(event))
    provenance = "espn_archived_pregame"
    if not quotes:
        quotes = [row for row in capture_rows if str(row.get("event_id")) == event_id and row.get("pregame")]
        provenance = "pickledger_pregame_capture"
    if not quotes:
        return []  # nothing priced pregame; skip the box-score fetch
    summary = client._get(  # noqa: SLF001 - archived ESPN box score endpoint
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/{LEAGUE}/summary", {"event": event_id}
    )
    actuals = basketball_actuals(summary)
    if not actuals:
        return []
    return graded_rows(event, date_iso, quotes, actuals, provider=DEFAULT_PROVIDER_NAME, provenance=provenance)


def grade_date(
    date_iso: str,
    *,
    client: Any | None = None,
    price_dir: Path = PRICE_DIR,
    history_dir: Path = HISTORY_DIR,
    max_workers: int = 6,
    regrade: bool = False,
) -> dict[str, Any]:
    client = client or DirectApiClient(timeout=25.0, attempts=3)
    events = [event for event in _scoreboard(client, date_iso) if event_completed(event)]
    path = shard_path(history_dir, date_iso)
    existing = [] if regrade else read_shard(path)
    done = {str(row.get("event_id")) for row in existing}
    pending = [event for event in events if str(event.get("id") or "") not in done]
    capture_rows = read_shard(shard_path(price_dir, date_iso))
    fresh: list[dict[str, Any]] = []
    failures: list[str] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pending)))) as executor:
            futures = {executor.submit(_grade_event, event, date_iso, capture_rows): event for event in pending}
            for future in as_completed(futures):
                event = futures[future]
                try:
                    fresh.extend(future.result())
                except Exception as exc:
                    failures.append(f"{event.get('id')}: {exc}")
    changed = False
    if fresh:
        changed = write_shard(path, merge_history(existing, fresh))
    return {
        "date": date_iso, "completed_events": len(events), "graded_events": len({r["event_id"] for r in fresh}),
        "rows": len(fresh), "changed": changed, "failures": failures, "path": str(path),
    }


def _dates(start: date, end: date) -> Iterable[str]:
    current = start
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)


def main() -> int:
    today = datetime.now(SLATE_TZ).date()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-date", default=today.isoformat(),
                        help="Slate date whose pregame quotes are captured ('' to skip).")
    parser.add_argument("--grade-start", default=(today - timedelta(days=4)).isoformat())
    parser.add_argument("--grade-end", default=(today - timedelta(days=1)).isoformat())
    parser.add_argument("--no-grade", action="store_true")
    parser.add_argument("--regrade", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    client = DirectApiClient(timeout=25.0, attempts=3)
    summary: dict[str, Any] = {"capture": None, "grades": []}
    problems: list[str] = []
    if args.capture_date:
        try:
            summary["capture"] = capture_date(client, args.capture_date)
            problems.extend(summary["capture"]["failures"])
        except Exception as exc:
            problems.append(f"capture {args.capture_date}: {exc}")
    if not args.no_grade:
        for day in _dates(date.fromisoformat(args.grade_start), date.fromisoformat(args.grade_end)):
            try:
                result = grade_date(day, client=client, max_workers=args.max_workers, regrade=args.regrade)
            except Exception as exc:
                problems.append(f"grade {day}: {exc}")
                continue
            summary["grades"].append(result)
            problems.extend(f"grade {day} {failure}" for failure in result["failures"])
            if result["completed_events"]:
                print(f"[nba-prop-archive] graded {day}: {result['graded_events']}/{result['completed_events']} "
                      f"events, {result['rows']} new rows", flush=True)
    summary["problems"] = problems
    summary["ok"] = not problems
    print(json.dumps(summary, indent=2, default=str))
    # Archiving is additive and optional for the refresh; problems are reported, not fatal.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

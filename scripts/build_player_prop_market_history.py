#!/usr/bin/env python3
"""Backfill immutable ESPN player-prop markets with final player outcomes.

The live player-prop cache is intentionally small and can change during the
day.  It is therefore unsuitable as a training corpus.  This script builds a
separate, append-safe market history from ESPN's archived DraftKings markets
and completed box scores.  Every output row represents one offered market,
not one model-selected pick.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from player_props.api import DirectApiClient  # noqa: E402
from player_props.basketball import (  # noqa: E402
    BASKETBALL_MARKET_TYPES,
    _american_odds as basketball_american_odds,
    _canonical_market_name as basketball_market_name,
    _is_milestone_market as basketball_is_milestone,
    _target_value as basketball_target_value,
)
from player_props.mlb import (  # noqa: E402
    MLB_MARKET_TYPES,
    _american_odds as mlb_american_odds,
    _athlete_ref_id,
    _canonical_market_name as mlb_market_name,
    _is_milestone_market as mlb_is_milestone,
    _market_display,
    _market_line,
)
from player_props.schema import american_implied_probability, safe_float  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data" / "player_props_training" / "market_history_2026.jsonl"
SPORT_CONFIG = {
    "MLB": {"segment": "baseball", "league": "mlb"},
    "WNBA": {"segment": "basketball", "league": "wnba"},
}


def _parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=f"{today.year}-03-20")
    parser.add_argument("--end", default=(today - timedelta(days=1)).isoformat())
    parser.add_argument("--sports", default="MLB,WNBA")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _number(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text or text in {"--", "-"}:
        return None
    if "-" in text and not text.startswith("-"):
        text = text.split("-", 1)[0]
    try:
        value_float = float(text)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def _mlb_outs(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "." not in text:
        whole = _number(text)
        return whole * 3.0 if whole is not None else None
    whole_text, partial_text = text.split(".", 1)
    whole = _number(whole_text)
    partial = _number(partial_text[:1])
    if whole is None or partial is None:
        return None
    return (whole * 3.0) + max(0.0, min(2.0, partial))


def _athlete_id(row: dict[str, Any]) -> str:
    return _athlete_ref_id(row)


def _event_start(event: dict[str, Any]) -> str:
    return str(event.get("date") or ((event.get("competitions") or [{}])[0].get("date")) or "")


def _event_provider(event: dict[str, Any]) -> tuple[str, str]:
    competition = (event.get("competitions") or [{}])[0]
    odds_rows = competition.get("odds") or []
    provider = (odds_rows[0].get("provider") or {}) if odds_rows else {}
    provider_id = str(provider.get("id") or "100")
    provider_name = str(provider.get("displayName") or provider.get("name") or "DraftKings")
    return provider_id, provider_name


def _completed(event: dict[str, Any]) -> bool:
    competition = (event.get("competitions") or [{}])[0]
    status = competition.get("status") or event.get("status") or {}
    status_type = status.get("type") or {}
    return bool(status_type.get("completed") is True)


def _summary_players(summary: dict[str, Any]) -> list[dict[str, Any]]:
    boxscore = summary.get("boxscore") or {}
    players = boxscore.get("players")
    return players if isinstance(players, list) else []


def _mlb_actuals(summary: dict[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for team in _summary_players(summary):
        for category in team.get("statistics") or []:
            keys = [str(key) for key in category.get("keys") or []]
            category_type = str(category.get("type") or "").lower()
            for row in category.get("athletes") or []:
                athlete_id = str((row.get("athlete") or {}).get("id") or "")
                stats = row.get("stats") or []
                if not athlete_id or len(stats) < len(keys):
                    continue
                values = {key: _number(stats[index]) for index, key in enumerate(keys)}
                if category_type == "batting":
                    hits = values.get("hits")
                    runs = values.get("runs")
                    rbis = values.get("RBIs")
                    aliases = {
                        "hits": hits,
                        "runs": runs,
                        "rbis": rbis,
                        "home_runs": values.get("homeRuns"),
                        "batter_walks": values.get("walks"),
                        "batter_strikeouts": values.get("strikeouts"),
                        "hits_runs_rbis": (
                            hits + runs + rbis
                            if hits is not None and runs is not None and rbis is not None
                            else None
                        ),
                    }
                elif category_type == "pitching":
                    innings_index = keys.index("fullInnings.partInnings") if "fullInnings.partInnings" in keys else -1
                    outs = _mlb_outs(stats[innings_index]) if innings_index >= 0 else None
                    aliases = {
                        "strikeouts": values.get("strikeouts"),
                        "pitcher_walks_allowed": values.get("walks"),
                        "pitcher_outs_recorded": outs,
                        "pitcher_hits_allowed": values.get("hits"),
                        "pitcher_earned_runs_allowed": values.get("earnedRuns"),
                    }
                else:
                    continue
                for stat_key, actual in aliases.items():
                    if actual is not None:
                        result[(athlete_id, stat_key)] = float(actual)
    return result


def _basketball_actuals(summary: dict[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for team in _summary_players(summary):
        for category in team.get("statistics") or []:
            keys = [str(key) for key in category.get("keys") or []]
            for row in category.get("athletes") or []:
                if row.get("didNotPlay") is True:
                    continue
                athlete_id = str((row.get("athlete") or {}).get("id") or "")
                stats = row.get("stats") or []
                if not athlete_id or len(stats) < len(keys):
                    continue
                values = {key: _number(stats[index]) for index, key in enumerate(keys)}
                points = values.get("points")
                rebounds = values.get("rebounds")
                assists = values.get("assists")
                threes_key = "threePointFieldGoalsMade-threePointFieldGoalsAttempted"
                aliases = {
                    "points": points,
                    "totalRebounds": rebounds,
                    "assists": assists,
                    "three_pointers_made": values.get(threes_key),
                    "steals": values.get("steals"),
                    "blocks": values.get("blocks"),
                    "points_rebounds": points + rebounds if points is not None and rebounds is not None else None,
                    "points_assists": points + assists if points is not None and assists is not None else None,
                    "points_rebounds_assists": (
                        points + rebounds + assists
                        if points is not None and rebounds is not None and assists is not None
                        else None
                    ),
                    "steals_blocks": (
                        values.get("steals") + values.get("blocks")
                        if values.get("steals") is not None and values.get("blocks") is not None
                        else None
                    ),
                }
                for stat_key, actual in aliases.items():
                    if actual is not None:
                        result[(athlete_id, stat_key)] = float(actual)
    return result


def _side_odds(row: dict[str, Any], parser: Any) -> int | None:
    return parser((((row.get("odds") or {}).get("american") or {}).get("value")))


def _market_rows(
    *,
    sport: str,
    event: dict[str, Any],
    items: list[dict[str, Any]],
    actuals: dict[tuple[str, str], float],
    provider_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    milestone_rows: list[tuple[dict[str, Any], str, str, float, str]] = []
    for item in items:
        type_name = str((item.get("type") or {}).get("name") or "")
        if sport == "MLB":
            normalized = mlb_market_name(type_name)
            market_type = MLB_MARKET_TYPES.get(normalized)
            athlete_id = _athlete_id(item)
            line = _market_line(item)
            display = _market_display(item)
            milestone = mlb_is_milestone(type_name, display)
            if not market_type:
                continue
            stat_key, _, _, grade_supported = market_type
            if not grade_supported:
                continue
        else:
            normalized = basketball_market_name(type_name)
            market_type = BASKETBALL_MARKET_TYPES.get(normalized)
            athlete_id = _athlete_id(item)
            line, display = basketball_target_value(item)
            milestone = basketball_is_milestone(type_name, display)
            if not market_type:
                continue
            stat_key, _ = market_type
        if not athlete_id or line <= 0 or (athlete_id, stat_key) not in actuals:
            continue
        if milestone:
            milestone_rows.append((item, athlete_id, stat_key, line, type_name))
        else:
            grouped[(athlete_id, stat_key, line, type_name)].append(item)

    event_id = str(event.get("id") or "")
    date_iso = _event_start(event)[:10]
    start_time = _event_start(event)
    output: list[dict[str, Any]] = []

    def build(
        athlete_id: str,
        stat_key: str,
        line: float,
        type_name: str,
        over_odds: int,
        under_odds: int | None,
        market_format: str,
        last_updated: str,
    ) -> dict[str, Any] | None:
        actual = actuals[(athlete_id, stat_key)]
        if actual == line:
            return None
        over_implied = american_implied_probability(over_odds)
        under_implied = american_implied_probability(under_odds)
        if over_implied is None:
            return None
        no_vig_over = (
            over_implied / (over_implied + under_implied)
            if under_implied is not None and over_implied + under_implied > 0
            else over_implied
        )
        return {
            "sport": sport,
            "season": int(date_iso[:4]),
            "date": date_iso,
            "start_time": start_time,
            "event_id": event_id,
            "athlete_id": athlete_id,
            "stat_key": stat_key,
            "market_type": type_name,
            "market_format": market_format,
            "line": float(line),
            "over_odds": over_odds,
            "under_odds": under_odds,
            "over_implied": round(over_implied, 6),
            "under_implied": round(under_implied, 6) if under_implied is not None else None,
            "no_vig_over": round(no_vig_over, 6),
            "actual": actual,
            "over_outcome": 1 if actual > line else 0,
            "market_updated_at": last_updated,
            "provider": provider_name,
        }

    for item, athlete_id, stat_key, threshold, type_name in milestone_rows:
        odds_parser = mlb_american_odds if sport == "MLB" else basketball_american_odds
        over_odds = _side_odds(item, odds_parser)
        if over_odds is None:
            continue
        row = build(
            athlete_id,
            stat_key,
            max(0.0, threshold - 0.5),
            type_name,
            over_odds,
            None,
            "milestone",
            str(item.get("lastUpdated") or ""),
        )
        if row:
            output.append(row)

    for (athlete_id, stat_key, line, type_name), sides in grouped.items():
        if len(sides) < 2:
            continue
        odds_parser = mlb_american_odds if sport == "MLB" else basketball_american_odds
        over_odds = _side_odds(sides[0], odds_parser)
        under_odds = _side_odds(sides[1], odds_parser)
        if over_odds is None or under_odds is None:
            continue
        row = build(
            athlete_id,
            stat_key,
            line,
            type_name,
            over_odds,
            under_odds,
            "total",
            str(sides[0].get("lastUpdated") or ""),
        )
        if row:
            output.append(row)
    return output


def _event_rows(sport: str, slate_date: str, event: dict[str, Any]) -> list[dict[str, Any]]:
    if not _completed(event):
        return []
    config = SPORT_CONFIG[sport]
    event_id = str(event.get("id") or "")
    if not event_id:
        return []
    provider_id, provider_name = _event_provider(event)
    client = DirectApiClient(timeout=25.0, attempts=3)
    summary = client._get(  # noqa: SLF001 - direct archived ESPN endpoint
        f"https://site.api.espn.com/apis/site/v2/sports/{config['segment']}/{config['league']}/summary",
        {"event": event_id},
    )
    try:
        if sport == "MLB":
            markets = client.mlb_espn_prop_bets(event_id, provider_id)
            actuals = _mlb_actuals(summary)
        else:
            markets = client.basketball_espn_prop_bets(config["league"], event_id, provider_id)
            actuals = _basketball_actuals(summary)
    except RuntimeError as exc:
        # ESPN returns 404 when a completed event never had provider props.
        if "404 Client Error" in str(exc):
            return []
        raise
    rows = _market_rows(
        sport=sport,
        event=event,
        items=list(markets.get("items") or []),
        actuals=actuals,
        provider_name=provider_name,
    )
    for row in rows:
        row["date"] = slate_date
    return rows


def _scoreboard(sport: str, date_iso: str) -> dict[str, Any]:
    client = DirectApiClient(timeout=25.0, attempts=3)
    config = SPORT_CONFIG[sport]
    if sport == "MLB":
        return client.mlb_espn_scoreboard(date_iso)
    return client.basketball_scoreboard(config["league"], date_iso)


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
            completed.add((str(row.get("sport") or ""), str(row.get("date") or "")))
    return rows, completed


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = {
        (
            row.get("sport"),
            row.get("event_id"),
            row.get("athlete_id"),
            row.get("stat_key"),
            row.get("line"),
            row.get("market_format"),
        ): row
        for row in rows
    }
    ordered = sorted(
        deduped.values(),
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("sport") or ""),
            str(row.get("event_id") or ""),
            str(row.get("athlete_id") or ""),
            str(row.get("stat_key") or ""),
            safe_float(row.get("line")),
        ),
    )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    sports = [item.strip().upper() for item in str(args.sports).split(",") if item.strip()]
    unknown = [sport for sport in sports if sport not in SPORT_CONFIG]
    if unknown:
        raise SystemExit(f"Unsupported sport(s): {', '.join(unknown)}")
    existing, completed_dates = _load_existing(args.output) if not args.no_resume else ([], set())
    rows = list(existing)
    failures: list[str] = []
    workers = max(1, int(args.max_workers))
    for target in _dates(start, end):
        date_iso = target.isoformat()
        for sport in sports:
            if (sport, date_iso) in completed_dates:
                continue
            try:
                events = list((_scoreboard(sport, date_iso).get("events") or []))
            except Exception as exc:
                failures.append(f"{sport} {date_iso} scoreboard: {exc}")
                continue
            day_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(workers, len(events) or 1)) as executor:
                futures = {
                    executor.submit(_event_rows, sport, date_iso, event): event
                    for event in events
                }
                for future in as_completed(futures):
                    event = futures[future]
                    try:
                        day_rows.extend(future.result())
                    except Exception as exc:
                        failures.append(f"{sport} {date_iso} {event.get('id')}: {exc}")
            rows.extend(day_rows)
            completed_dates.add((sport, date_iso))
            _write_rows(args.output, rows)
            print(f"[market-history] {sport} {date_iso}: {len(day_rows)} graded market(s)")
    summary = {
        "ok": not failures,
        "output": str(args.output),
        "rows": len(rows),
        "sports": sports,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "failures": failures,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

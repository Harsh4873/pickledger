"""MLS match history: football-data.co.uk archive plus ESPN roll-forward.

Training and runtime share one match representation keyed by ESPN team id, so
ratings fit on the historical archive apply directly to the live ESPN slate.
The football-data ``new/USA.csv`` workbook covers every MLS match since 2012
(including playoffs) with Pinnacle/average/best closing 1X2 prices and is
refreshed within a few days of each matchday; ESPN scoreboards fill the gap
between the workbook's last row and today.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MATCHES_PATH = ARTIFACT_DIR / "mls_matches.json.gz"
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/USA.csv"

# football-data.co.uk team name -> ESPN usa.1 team id. football-data uses each
# franchise's current name across all seasons, so the map is one entry per
# franchise. Chivas USA folded in 2014 and has no ESPN usa.1 entry; its
# matches still shape opponents' ratings, keyed by a synthetic id that no
# live slate can produce.
FOOTBALL_DATA_TO_ESPN = {
    "Atlanta Utd": "18418",
    "Austin FC": "20906",
    "CF Montreal": "9720",
    "Charlotte": "21300",
    "Chicago Fire": "182",
    "Chivas USA": "chivas-usa",
    "Colorado Rapids": "184",
    "Columbus Crew": "183",
    "DC United": "193",
    "FC Cincinnati": "18267",
    "FC Dallas": "185",
    "Houston Dynamo": "6077",
    "Inter Miami": "20232",
    "Los Angeles FC": "18966",
    "Los Angeles Galaxy": "187",
    "Minnesota United": "17362",
    "Nashville SC": "18986",
    "New England Revolution": "189",
    "New York City": "17606",
    "New York Red Bulls": "190",
    "Orlando City": "12011",
    "Philadelphia Union": "10739",
    "Portland Timbers": "9723",
    "Real Salt Lake": "4771",
    "San Diego FC": "22529",
    "San Jose Earthquakes": "191",
    "Seattle Sounders": "9726",
    "Sporting Kansas City": "186",
    "St. Louis City": "21812",
    "Toronto FC": "7318",
    "Vancouver Whitecaps": "9727",
}


@dataclass(frozen=True)
class MlsMatch:
    """One completed MLS match (90-minute result)."""

    date: date
    home: str
    away: str
    home_goals: int
    away_goals: int
    # Closing 1X2 decimal prices; None outside the training pipeline.
    close_home: float | None = None
    close_draw: float | None = None
    close_away: float | None = None
    close_source: str | None = None
    best_home: float | None = None
    best_draw: float | None = None
    best_away: float | None = None

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals


def _decimal(value: str | None) -> float | None:
    try:
        number = float(str(value or "").strip())
    except ValueError:
        return None
    return number if number > 1.0 else None


def parse_football_data_csv(text: str, *, strict: bool = True) -> list[MlsMatch]:
    """Parse the football-data ``new/USA.csv`` workbook into matches.

    Dates are dd/mm/yyyy at UTC kickoff. Closing-price preference is
    Pinnacle (PSC*) then the cross-book average (AvgC*) then bet365 (B365C*):
    Pinnacle is the sharp reference but is missing from some recent rows.
    """
    matches: list[MlsMatch] = []
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    for row in reader:
        home_name = (row.get("Home") or "").strip()
        away_name = (row.get("Away") or "").strip()
        if not home_name or not away_name:
            continue
        home = FOOTBALL_DATA_TO_ESPN.get(home_name)
        away = FOOTBALL_DATA_TO_ESPN.get(away_name)
        if home is None or away is None:
            if strict:
                unknown = home_name if home is None else away_name
                raise ValueError(f"unmapped football-data team name: {unknown!r}")
            continue
        try:
            when = datetime.strptime((row.get("Date") or "").strip(), "%d/%m/%Y").date()
            home_goals = int(float(row["HG"]))
            away_goals = int(float(row["AG"]))
        except (KeyError, TypeError, ValueError):
            continue
        close = None
        for source, keys in (
            ("pinnacle_close", ("PSCH", "PSCD", "PSCA")),
            ("average_close", ("AvgCH", "AvgCD", "AvgCA")),
            ("bet365_close", ("B365CH", "B365CD", "B365CA")),
        ):
            prices = tuple(_decimal(row.get(key)) for key in keys)
            if all(price is not None for price in prices):
                close = (source, prices)
                break
        best = tuple(_decimal(row.get(key)) for key in ("MaxCH", "MaxCD", "MaxCA"))
        matches.append(MlsMatch(
            date=when,
            home=home,
            away=away,
            home_goals=home_goals,
            away_goals=away_goals,
            close_home=close[1][0] if close else None,
            close_draw=close[1][1] if close else None,
            close_away=close[1][2] if close else None,
            close_source=close[0] if close else None,
            best_home=best[0] if all(price is not None for price in best) else None,
            best_draw=best[1] if all(price is not None for price in best) else None,
            best_away=best[2] if all(price is not None for price in best) else None,
        ))
    matches.sort(key=lambda match: (match.date, match.home, match.away))
    return matches


def match_key(match: MlsMatch) -> tuple[str, str, str]:
    return (match.date.isoformat(), match.home, match.away)


def merge_matches(base: Iterable[MlsMatch], extra: Iterable[MlsMatch]) -> list[MlsMatch]:
    """Union of two match lists; ``base`` wins on duplicates.

    football-data and ESPN both stamp UTC kickoff, but a rescheduled or
    double-listed fixture can land a day apart, so the same pairing within
    two days counts as one match.
    """
    merged: dict[tuple[str, str, str], MlsMatch] = {match_key(match): match for match in base}
    windows = {
        (match.home, match.away, match.date + timedelta(days=offset))
        for match in merged.values()
        for offset in (-2, -1, 0, 1, 2)
    }
    for match in extra:
        if (match.home, match.away, match.date) in windows:
            continue
        merged[match_key(match)] = match
        windows.update(
            (match.home, match.away, match.date + timedelta(days=offset))
            for offset in (-2, -1, 0, 1, 2)
        )
    return sorted(merged.values(), key=lambda match: (match.date, match.home, match.away))


def save_matches(matches: list[MlsMatch], path: Path = MATCHES_PATH) -> None:
    """Persist the compact score archive the runtime refits from."""
    rows = [
        [match.date.isoformat(), match.home, match.away, match.home_goals, match.away_goals]
        for match in matches
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"columns": ["date", "home", "away", "home_goals", "away_goals"], "rows": rows}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def load_matches(path: Path = MATCHES_PATH) -> list[MlsMatch]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        MlsMatch(
            date=date.fromisoformat(row[0]),
            home=str(row[1]),
            away=str(row[2]),
            home_goals=int(row[3]),
            away_goals=int(row[4]),
        )
        for row in payload.get("rows", [])
    ]


def fetch_football_data_matches(session: Any = None, timeout: int = 20, *, strict: bool = True) -> list[MlsMatch]:
    """Download the current workbook; raises on network/HTTP failure.

    Serving passes ``strict=False`` so a future franchise name missing from
    the map skips that row instead of rejecting the whole workbook; training
    stays strict so the map is always completed first.
    """
    import requests

    getter = session or requests
    response = getter.get(FOOTBALL_DATA_URL, timeout=timeout, headers={"User-Agent": "PickLedgerMLSModel/2.0"})
    response.raise_for_status()
    return parse_football_data_csv(response.content.decode("utf-8-sig"), strict=strict)


def espn_completed_matches(scoreboard: dict[str, Any]) -> list[MlsMatch]:
    """Completed 90-minute results from one ESPN usa.1 scoreboard payload."""
    matches: list[MlsMatch] = []
    for event in scoreboard.get("events") if isinstance(scoreboard.get("events"), list) else []:
        if not isinstance(event, dict):
            continue
        status = event.get("status") if isinstance(event.get("status"), dict) else {}
        status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
        if str(status_type.get("state") or "").lower() != "post" and not status_type.get("completed"):
            continue
        competitions = event.get("competitions") if isinstance(event.get("competitions"), list) else []
        competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
        competitors = competition.get("competitors") if isinstance(competition.get("competitors"), list) else []
        home = next((item for item in competitors if isinstance(item, dict) and item.get("homeAway") == "home"), None)
        away = next((item for item in competitors if isinstance(item, dict) and item.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            home_goals = int(float(home.get("score")))
            away_goals = int(float(away.get("score")))
        except (TypeError, ValueError):
            continue
        home_id = str((home.get("team") or {}).get("id") or "")
        away_id = str((away.get("team") or {}).get("id") or "")
        when = str(event.get("date") or "")[:10]
        if not home_id or not away_id or not when:
            continue
        try:
            match_date = date.fromisoformat(when)
        except ValueError:
            continue
        matches.append(MlsMatch(
            date=match_date,
            home=home_id,
            away=away_id,
            home_goals=home_goals,
            away_goals=away_goals,
        ))
    return matches

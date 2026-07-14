"""
wnba_injuries.py — WNBA injury impact scoring system.

Fetches injury reports from three independent sources (ESPN, RotoWire, and
BallDontLie), merges them into a single report keyed by player name, and
computes per-team "injury penalties" that quantify how much scoring output
the team is likely to lose on a given night.

Public API:
    get_injury_report(force_refresh=False)     -> dict
    get_team_injury_penalty(team_abbr, live_stats=None) -> float
    fetch_espn_injuries()                       -> list[dict]
    fetch_rotowire_injuries()                   -> list[dict]
    fetch_bdl_injuries()                        -> list[dict]
    merge_injury_reports(espn, rotowire, bdl)   -> dict
    compute_team_injury_penalty(...)            -> float

Cache: data/wnba_injuries.json (45-minute TTL).
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import requests
from bs4 import BeautifulSoup

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from .wnba_teams import WNBA_TEAM_MAP
except ImportError:
    from wnba_teams import WNBA_TEAM_MAP

# ---------------------------------------------------------------------------
# API key: try config.py first, then fall back to environment variable
# ---------------------------------------------------------------------------
try:
    from config import BDL_API_KEY
except ImportError:
    BDL_API_KEY = None

if not BDL_API_KEY:
    BDL_API_KEY = os.getenv("BDL_API_KEY", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESPN_INJURY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"
    "?team={espn_id}"
)
ROTOWIRE_INJURY_URL = (
    "https://www.rotowire.com/wnba/tables/injury-report.php?team=ALL&pos=ALL"
)
BDL_INJURY_URL = "https://api.balldontlie.io/wnba/v1/player_injuries"

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
)

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wnba")
)
os.makedirs(DATA_DIR, exist_ok=True)
INJURY_CACHE_PATH = os.path.join(DATA_DIR, "wnba_injuries.json")

CACHE_TTL_SECONDS = 45 * 60  # 45 minutes

# Abbreviation aliases — mirrors wnba_teams.py.  RotoWire uses different codes
# than ESPN/BDL; map them through here.
_ABBR_ALIASES: dict[str, str] = {
    "LVA":  "LV",
    "LAS":  "LV",   # Sometimes appears for Las Vegas
    "CONN": "CON",
    "NYL":  "NY",
    "WSH":  "WAS",
    "WAS":  "WAS",
    "PHO":  "PHX",
    "GS":   "GSV",
    "GSW":  "GSV",
    "LAS_SPARKS": "LA",
    "LAX":  "LA",
}

# ---------------------------------------------------------------------------
# Status severity
# ---------------------------------------------------------------------------
STATUS_WEIGHTS: dict[str, float] = {
    "Out":         1.0,
    "Doubtful":    0.65,
    "Questionable": 0.35,
    "Day-To-Day":  0.20,
}

# Ordered from most to least severe — used when merging sources.
_STATUS_SEVERITY = ["Out", "Doubtful", "Questionable", "Day-To-Day"]

# Aliases we normalize into the canonical status labels above.
_STATUS_ALIASES: dict[str, str] = {
    "out":           "Out",
    "o":             "Out",
    "doubtful":      "Doubtful",
    "d":             "Doubtful",
    "questionable":  "Questionable",
    "q":             "Questionable",
    "day-to-day":    "Day-To-Day",
    "day to day":    "Day-To-Day",
    "dtd":           "Day-To-Day",
    "gtd":           "Day-To-Day",   # "Game-Time Decision" ~= DTD
    "probable":      "Day-To-Day",
}

# ---------------------------------------------------------------------------
# Star ratings — hardcoded priors, overridden by live stats when available
# ---------------------------------------------------------------------------
WNBA_STAR_RATINGS: dict[str, dict] = {
    "caitlin clark":     {"team": "IND", "pts_share": 0.28, "ast_share": 0.40, "usage": 0.32},
    "a'ja wilson":       {"team": "LV",  "pts_share": 0.33, "ast_share": 0.15, "usage": 0.35},
    "breanna stewart":   {"team": "NY",  "pts_share": 0.27, "ast_share": 0.18, "usage": 0.30},
    "sabrina ionescu":   {"team": "NY",  "pts_share": 0.24, "ast_share": 0.30, "usage": 0.28},
    "napheesa collier":  {"team": "MIN", "pts_share": 0.26, "ast_share": 0.15, "usage": 0.29},
    "alyssa thomas":     {"team": "CON", "pts_share": 0.18, "ast_share": 0.32, "usage": 0.26},
    "kelsey plum":       {"team": "LV",  "pts_share": 0.22, "ast_share": 0.22, "usage": 0.27},
    "diana taurasi":     {"team": "PHX", "pts_share": 0.20, "ast_share": 0.20, "usage": 0.25},
    "nneka ogwumike":    {"team": "SEA", "pts_share": 0.22, "ast_share": 0.12, "usage": 0.25},
    "jewell loyd":       {"team": "SEA", "pts_share": 0.25, "ast_share": 0.18, "usage": 0.27},
}

# Position-based default pts_share used when no other info is available.
_POSITION_DEFAULT_PTS_SHARE: dict[str, float] = {
    "G":  0.12,
    "F":  0.10,
    "C":  0.09,
    "":   0.08,
}

# Module-level sink for star warnings — populated by compute_team_injury_penalty
# so the CLI can replay them at the end.
_STAR_WARNINGS: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_abbr(raw: str) -> str:
    """Return the canonical team abbreviation for a raw API/scraper value."""
    if not raw:
        return ""
    raw = raw.strip().upper()
    canonical = _ABBR_ALIASES.get(raw, raw)
    return canonical


def _normalize_name(name: str) -> str:
    """Lowercase + strip whitespace for name keying."""
    return (name or "").strip().lower()


def _normalize_status(raw: str) -> str:
    """
    Map a raw status string to one of the four canonical labels.
    Unknown values fall back to 'Day-To-Day' (least severe) so the player
    is still represented — we'd rather slightly over-count than drop them.
    """
    if not raw:
        return "Day-To-Day"
    key = raw.strip().lower()
    # Try exact alias match first
    if key in _STATUS_ALIASES:
        return _STATUS_ALIASES[key]
    # Substring heuristic for messy free-text statuses
    for alias, canonical in _STATUS_ALIASES.items():
        if alias in key:
            return canonical
    return "Day-To-Day"


def _status_severity(status: str) -> int:
    """Return a numeric severity index (lower = more severe)."""
    try:
        return _STATUS_SEVERITY.index(status)
    except ValueError:
        return len(_STATUS_SEVERITY)


def _more_severe(a: str, b: str) -> str:
    """Return the more severe of two status strings."""
    return a if _status_severity(a) <= _status_severity(b) else b


# ---------------------------------------------------------------------------
# Section 1 — Data fetchers
# ---------------------------------------------------------------------------

def fetch_espn_injuries() -> list[dict]:
    """
    Fetch injuries from ESPN for every team in WNBA_TEAM_MAP.

    Returns a flat list of dicts:
        {player_name, team_abbr, status, source: "espn", comment}
    HTTP errors per-team are swallowed — one failing team must never
    affect the others.
    """
    results: list[dict] = []

    for team_abbr, team_info in WNBA_TEAM_MAP.items():
        espn_id = team_info.get("espn_id")
        if not espn_id:
            continue

        url = ESPN_INJURY_URL.format(espn_id=espn_id)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"[WNBA] ESPN injury fetch failed for {team_abbr}: {exc}")
            time.sleep(0.25)
            continue

        # ESPN sometimes nests injuries per team under a wrapper, sometimes
        # returns them flat.  Handle the common shapes.
        injuries_list: list[dict] = []
        raw_injuries = data.get("injuries", []) or []
        if raw_injuries and isinstance(raw_injuries[0], dict) and "injuries" in raw_injuries[0]:
            for wrapper in raw_injuries:
                injuries_list.extend(wrapper.get("injuries", []) or [])
        else:
            injuries_list = raw_injuries

        for injury in injuries_list:
            try:
                athlete = injury.get("athlete") or {}
                player_name = (
                    athlete.get("displayName")
                    or athlete.get("fullName")
                    or ""
                ).strip()
                if not player_name:
                    continue

                status = _normalize_status(injury.get("status", ""))
                comment = injury.get("shortComment") or ""

                results.append({
                    "player_name": player_name,
                    "team_abbr": team_abbr,
                    "status": status,
                    "source": "espn",
                    "comment": comment,
                })
            except (KeyError, TypeError, AttributeError):
                continue

        time.sleep(0.25)

    return results


def fetch_rotowire_injuries() -> list[dict]:
    """
    Scrape the RotoWire WNBA injury report.

    Returns:
        list of dicts: {player_name, team_abbr, position, status,
                        source: "rotowire", comment}
    On any HTTP or parse error: prints a warning and returns [].
    """
    try:
        resp = requests.get(
            ROTOWIRE_INJURY_URL,
            headers={"User-Agent": BROWSER_UA},
            timeout=20,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:  # noqa: BLE001 — broad on purpose, never crash
        print(f"[WNBA] RotoWire scrape failed: {exc}")
        return []

    try:
        # Find the injury table — look for a <table> whose header row has
        # Player, Team, and Status columns.
        target_table = None
        for table in soup.find_all("table"):
            header_cells = table.find_all("th")
            headers_text = [
                (c.get_text(" ", strip=True) or "").lower()
                for c in header_cells
            ]
            has_player = any("player" in h for h in headers_text)
            has_team = any("team" == h or "team " in h or h.startswith("team") for h in headers_text)
            has_status = any("status" in h for h in headers_text)
            if has_player and has_team and has_status:
                target_table = table
                break

        if target_table is None:
            print("[WNBA] RotoWire scrape failed: injury table not found")
            return []

        # Index the header row so we know which column is which.
        header_cells = target_table.find_all("th")
        header_index: dict[str, int] = {}
        for idx, cell in enumerate(header_cells):
            label = (cell.get_text(" ", strip=True) or "").lower()
            if "player" in label and "player" not in header_index:
                header_index["player"] = idx
            elif label.startswith("team") and "team" not in header_index:
                header_index["team"] = idx
            elif ("pos" in label) and "position" not in header_index:
                header_index["position"] = idx
            elif "status" in label and "status" not in header_index:
                header_index["status"] = idx
            elif (
                "injury" in label
                or "description" in label
                or "comment" in label
                or "notes" in label
            ) and "comment" not in header_index:
                header_index["comment"] = idx

        results: list[dict] = []
        # Iterate body rows — prefer <tbody> rows if present, else every <tr>
        # except the header.
        body = target_table.find("tbody")
        rows = body.find_all("tr") if body else target_table.find_all("tr")[1:]

        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            def _cell(key: str) -> str:
                idx = header_index.get(key)
                if idx is None or idx >= len(cells):
                    return ""
                return cells[idx].get_text(" ", strip=True) or ""

            player_name = _cell("player").strip()
            if not player_name:
                continue

            raw_team = _cell("team")
            team_abbr = _normalize_abbr(raw_team)
            position = _cell("position")
            status = _normalize_status(_cell("status"))
            comment = _cell("comment")

            results.append({
                "player_name": player_name,
                "team_abbr": team_abbr,
                "position": position,
                "status": status,
                "source": "rotowire",
                "comment": comment,
            })

        return results
    except Exception as exc:  # noqa: BLE001
        print(f"[WNBA] RotoWire scrape failed: {exc}")
        return []


def fetch_bdl_injuries() -> list[dict]:
    """
    Fetch WNBA player injuries from BallDontLie.

    Returns:
        list of dicts: {player_name, team_abbr, status, source: "bdl",
                        return_date, comment}
    Silently returns [] if BDL_API_KEY is missing, or on error.
    """
    if not BDL_API_KEY:
        return []

    results: list[dict] = []
    cursor: str | None = None

    while True:
        try:
            params: dict[str, str | int] = {"per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor

            resp = requests.get(
                BDL_INJURY_URL,
                headers={"Authorization": BDL_API_KEY},
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        for entry in data.get("data", []) or []:
            try:
                player = entry.get("player") or {}
                first = (player.get("first_name") or "").strip()
                last = (player.get("last_name") or "").strip()
                player_name = f"{first} {last}".strip()
                if not player_name:
                    # Some payloads put the name at the top level
                    player_name = (entry.get("player_name") or "").strip()
                if not player_name:
                    continue

                team_info = player.get("team") or entry.get("team") or {}
                team_abbr = _normalize_abbr(team_info.get("abbreviation", ""))

                status = _normalize_status(entry.get("status", ""))
                return_date = entry.get("return_date") or ""
                comment = entry.get("description") or entry.get("comment") or ""

                results.append({
                    "player_name": player_name,
                    "team_abbr": team_abbr,
                    "status": status,
                    "source": "bdl",
                    "return_date": return_date,
                    "comment": comment,
                })
            except (KeyError, TypeError, AttributeError):
                continue

        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)

    return results


# ---------------------------------------------------------------------------
# Section 2 — Merge
# ---------------------------------------------------------------------------

def merge_injury_reports(
    espn: list[dict],
    rotowire: list[dict],
    bdl: list[dict],
) -> dict:
    """
    Merge three per-source injury lists into a single dict keyed by
    lowercase player name.

    When a player appears in multiple sources we keep the **most severe**
    status and collect every source label in `sources`.  The first non-empty
    team_abbr / comment we see is preferred.

    Returns:
        {player_name_lower: {player_name, team_abbr, status,
                              sources: [list], comment}}
    """
    merged: dict[str, dict] = {}

    for source_list in (espn or [], rotowire or [], bdl or []):
        for entry in source_list:
            name = entry.get("player_name", "")
            key = _normalize_name(name)
            if not key:
                continue

            status = entry.get("status") or "Day-To-Day"
            source = entry.get("source", "")
            team_abbr = entry.get("team_abbr", "") or ""
            comment = entry.get("comment", "") or ""

            if key not in merged:
                merged[key] = {
                    "player_name": name.strip(),
                    "team_abbr": team_abbr,
                    "status": status,
                    "sources": [source] if source else [],
                    "comment": comment,
                }
                continue

            existing = merged[key]

            # Keep the most severe status
            existing["status"] = _more_severe(existing["status"], status)

            # Fill in missing team_abbr / comment
            if not existing.get("team_abbr") and team_abbr:
                existing["team_abbr"] = team_abbr
            if not existing.get("comment") and comment:
                existing["comment"] = comment

            # Track sources (unique, in order of appearance)
            if source and source not in existing["sources"]:
                existing["sources"].append(source)

    return merged


# ---------------------------------------------------------------------------
# Section 3 — Impact scoring
# ---------------------------------------------------------------------------

def compute_team_injury_penalty(
    team_abbr: str,
    injury_report: dict,
    live_stats: dict | None = None,
) -> float:
    """
    Compute an injury penalty in [0.0, 0.45] for a team.

    For each injured player on the team:
        pts_share * STATUS_WEIGHTS[status]
    Summed and capped at 0.45 (no team loses more than 45% of scoring).

    pts_share priority:
        1. live_stats[player_key]["pts_share"]  (canonical)
        2. WNBA_STAR_RATINGS[player_key]["pts_share"]  (hardcoded prior)
        3. position default (G/F/C/"")
    """
    abbr = _normalize_abbr(team_abbr)
    if not abbr or not injury_report:
        return 0.0

    live_stats = live_stats or {}
    total_penalty = 0.0

    for key, info in injury_report.items():
        if _normalize_abbr(info.get("team_abbr", "")) != abbr:
            continue

        status = info.get("status") or "Day-To-Day"
        weight = STATUS_WEIGHTS.get(status, 0.0)
        if weight <= 0.0:
            continue

        # 1) live_stats lookup
        pts_share: float | None = None
        live_entry = live_stats.get(key) if isinstance(live_stats, dict) else None
        if isinstance(live_entry, dict) and "pts_share" in live_entry:
            try:
                pts_share = float(live_entry["pts_share"])
            except (TypeError, ValueError):
                pts_share = None

        # 2) hardcoded star rating
        if pts_share is None and key in WNBA_STAR_RATINGS:
            pts_share = float(WNBA_STAR_RATINGS[key]["pts_share"])

        # 3) position-based default
        if pts_share is None:
            position = (info.get("position") or "").strip().upper()
            # Normalize compound positions like "G-F" to the primary letter
            primary = position[0] if position else ""
            pts_share = _POSITION_DEFAULT_PTS_SHARE.get(
                primary,
                _POSITION_DEFAULT_PTS_SHARE[""],
            )

        player_penalty = pts_share * weight
        total_penalty += player_penalty

        # Warn on any star who is Out or Doubtful
        if key in WNBA_STAR_RATINGS and status in ("Out", "Doubtful"):
            player_name = info.get("player_name") or key.title()
            warning = (
                f"[WNBA] ⚠️ {player_name} ({abbr}) is {status} — "
                f"high impact penalty: {player_penalty:.2f}"
            )
            print(warning)
            _STAR_WARNINGS.append(warning)

    # Cap at 0.45 — prevent absurd outputs when a whole team is banged up
    if total_penalty > 0.45:
        total_penalty = 0.45
    if total_penalty < 0.0:
        total_penalty = 0.0

    return total_penalty


# ---------------------------------------------------------------------------
# Section 4 — Public API
# ---------------------------------------------------------------------------

def _load_cache() -> dict | None:
    """Return the cached merged report dict if it is < 45 min old."""
    if not os.path.exists(INJURY_CACHE_PATH):
        return None

    try:
        with open(INJURY_CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[WNBA] Could not read injury cache: {exc}")
        return None

    last_updated = payload.get("last_updated")
    if not last_updated:
        return None

    try:
        updated_dt = datetime.datetime.fromisoformat(last_updated)
    except ValueError:
        return None

    now = (
        datetime.datetime.now(updated_dt.tzinfo)
        if updated_dt.tzinfo
        else datetime.datetime.now()
    )
    age = (now - updated_dt).total_seconds()
    if age < 0 or age > CACHE_TTL_SECONDS:
        return None

    report = payload.get("report")
    if not isinstance(report, dict):
        return None
    return report


def _save_cache(report: dict) -> None:
    """Persist the merged report with a `last_updated` ISO timestamp."""
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "last_updated": datetime.datetime.now().isoformat(),
        "report": report,
    }
    try:
        with open(INJURY_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"[WNBA] Could not write injury cache: {exc}")


def get_injury_report(force_refresh: bool = False) -> dict:
    """
    Return the merged injury report dict.

    Uses the 45-minute cache unless `force_refresh=True`.
    On refresh, all three fetchers are called independently — one failing
    must never prevent the others from contributing.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            return cached

    # Each fetcher is wrapped so a crash in one never affects the others.
    try:
        espn = fetch_espn_injuries()
    except Exception as exc:  # noqa: BLE001
        print(f"[WNBA] ESPN injuries fetcher crashed: {exc}")
        espn = []

    try:
        rotowire = fetch_rotowire_injuries()
    except Exception as exc:  # noqa: BLE001
        print(f"[WNBA] RotoWire injuries fetcher crashed: {exc}")
        rotowire = []

    try:
        bdl = fetch_bdl_injuries()
    except Exception as exc:  # noqa: BLE001
        print(f"[WNBA] BDL injuries fetcher crashed: {exc}")
        bdl = []

    merged = merge_injury_reports(espn, rotowire, bdl)
    _save_cache(merged)
    return merged


def get_team_injury_penalty(
    team_abbr: str,
    live_stats: dict | None = None,
) -> float:
    """Convenience wrapper: load the cached report and score one team."""
    report = get_injury_report()
    return compute_team_injury_penalty(team_abbr, report, live_stats=live_stats)


# ---------------------------------------------------------------------------
# Section 5 — CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = get_injury_report(force_refresh=True)
    print(f"\n[WNBA] Total injured players found: {len(report)}")

    # Reset warnings sink so only this run's warnings are printed at the end.
    _STAR_WARNINGS.clear()

    for team in ["IND", "LV", "MIN", "NY", "SEA"]:
        penalty = get_team_injury_penalty(team)
        print(f"[{team}] Injury penalty: {penalty:.3f}")

    if _STAR_WARNINGS:
        print("\n[WNBA] Star warnings that fired this run:")
        for warning in _STAR_WARNINGS:
            print(f"  {warning}")
    else:
        print("\n[WNBA] No star warnings fired this run.")

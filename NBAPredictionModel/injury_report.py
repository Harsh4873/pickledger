"""
NBA Injury Report Fetcher.
Uses scraper-based injuries by default (no JVM), with optional nbainjuries fallback.
"""
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup

_HAS_NBAINJURIES = False
_NBAINJURIES_IMPORT_ERROR = ""

# Team full name -> short name mapping
TEAM_SHORT = {
    "Atlanta Hawks": "Hawks", "Boston Celtics": "Celtics", "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets", "Chicago Bulls": "Bulls", "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks", "Denver Nuggets": "Nuggets", "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors", "Houston Rockets": "Rockets", "Indiana Pacers": "Pacers",
    "Los Angeles Clippers": "Clippers", "Los Angeles Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies", "Miami Heat": "Heat", "Milwaukee Bucks": "Bucks",
    "Minnesota Timberwolves": "Timberwolves", "New Orleans Pelicans": "Pelicans",
    "New York Knicks": "Knicks", "Oklahoma City Thunder": "Thunder", "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers", "Phoenix Suns": "Suns", "Portland Trail Blazers": "Trail Blazers",
    "Sacramento Kings": "Kings", "San Antonio Spurs": "Spurs", "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz", "Washington Wizards": "Wizards",
}

TEAM_ALIASES = {full.lower(): short for full, short in TEAM_SHORT.items()}
for short in TEAM_SHORT.values():
    TEAM_ALIASES[short.lower()] = short

EXPECTED_ABSENCE_PROBABILITIES = {
    "Out": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.50,
    "Probable": 0.25,
}


def _normalize_team_name(team_name: str) -> str:
    if not team_name:
        return team_name
    cleaned = " ".join(str(team_name).split()).strip()
    direct = TEAM_ALIASES.get(cleaned.lower())
    if direct:
        return direct
    for full, short in TEAM_SHORT.items():
        if cleaned.lower() in full.lower() or short.lower() in cleaned.lower():
            return short
    return cleaned


def _normalize_status(status: str) -> str:
    s = str(status or "").strip().lower()
    if "out" in s:
        return "Out"
    if "doubt" in s:
        return "Doubtful"
    if "question" in s or "gtd" in s or "game-time" in s or "game time" in s or "day-to-day" in s:
        return "Questionable"
    if "probable" in s:
        return "Probable"
    return str(status or "").strip().title() or "Unknown"


def get_expected_absence_probability(status: str) -> float:
    """Map a raw injury status to the expected probability the player sits."""
    normalized_status = _normalize_status(status)
    return EXPECTED_ABSENCE_PROBABILITIES.get(normalized_status, 0.0)


def _fetch_scraped_injuries() -> dict:
    scraped = _scrape_cbs_injuries()
    if not scraped:
        scraped = _scrape_rotowire_injuries()
    if not scraped:
        return {}
    injuries = {}
    for team_raw, players in scraped.items():
        team = _normalize_team_name(team_raw)
        if team not in injuries:
            injuries[team] = []
        for p in players or []:
            name = str(p.get("name", "")).strip()
            if not name:
                continue
            status = _normalize_status(p.get("status", ""))
            reason = str(p.get("injury", "")).strip() or str(p.get("reason", "")).strip()
            injuries[team].append({
                "name": name,
                "status": status,
                "reason": reason,
            })
    return injuries


def _scrape_cbs_injuries() -> dict:
    url = "https://www.cbssports.com/nba/injuries/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        print(f"    [injury_report] WARNING: CBS scrape failed: {exc}")
        return {}

    injuries = {}
    for wrapper in soup.select(".TableBaseWrapper"):
        team_lockup = wrapper.select_one(".TeamLogoNameLockup-name, .TeamName")
        if not team_lockup:
            continue
        team = team_lockup.get_text(" ", strip=True)
        if not team:
            continue
        team_key = _normalize_team_name(team)
        injuries.setdefault(team_key, [])

        table = wrapper.select_one("table.TableBase-table")
        if not table:
            continue
        for row in table.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 5:
                continue
            raw_player = cells[0].strip()
            player_name = raw_player
            first_dot = raw_player.find(". ")
            if first_dot != -1 and first_dot + 2 < len(raw_player):
                # CBS often formats as "N. Last First Last"; keep the full-name tail.
                player_name = raw_player[first_dot + 2 :].strip()
            status = cells[4]
            injury_type = cells[3]
            injuries[team_key].append({
                "name": player_name.strip(),
                "status": status.strip(),
                "injury": injury_type.strip(),
            })

    return {k: v for k, v in injuries.items() if v}


def _scrape_rotowire_injuries() -> dict:
    url = "https://www.rotowire.com/basketball/injury-report.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        print(f"    [injury_report] WARNING: Rotowire scrape failed: {exc}")
        return {}

    injuries = {}
    current_team = None
    for el in soup.find_all(["div", "tr"]):
        team_header = el.find(class_=lambda x: x and "team" in str(x).lower())
        if team_header:
            team_text = team_header.get_text(strip=True)
            if team_text and len(team_text) > 2:
                current_team = team_text
                injuries.setdefault(current_team, [])

        cells = el.find_all("td")
        if not cells or len(cells) < 2 or not current_team:
            continue
        player_name = cells[0].get_text(strip=True)
        status = cells[1].get_text(strip=True)
        injury_detail = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        if player_name and any(s in status.lower() for s in ["out", "doubtful", "questionable", "probable", "day-to-day", "gtd"]):
            injuries[current_team].append({
                "name": player_name,
                "status": status,
                "injury": injury_detail,
            })
    return injuries


def fetch_injuries(dt: datetime = None) -> dict:
    """
    Fetch official NBA injury report.
    
    Args:
        dt: datetime for the report snapshot (default: most recent available snapshot)
    
    Returns:
        Dict keyed by short team name -> list of injured players.
    """
    import datetime as dt_module

    # Default path: scraper injuries (safe on Render and local; no JVM/JPype).
    scraped = _fetch_scraped_injuries()
    if scraped:
        print("    [injury_report] Using scraper-based injury report.")
        return scraped

    # Optional fallback: nbainjuries for environments where JVM integration is known-safe.
    if os.getenv("NBA_ENABLE_NBAINJURIES", "0") != "1":
        print("    [injury_report] WARNING: scraper returned no injuries and nbainjuries is disabled.")
        return {}

    global _HAS_NBAINJURIES, _NBAINJURIES_IMPORT_ERROR
    if not _HAS_NBAINJURIES:
        try:
            from nbainjuries import injury as _injury
            globals()["injury"] = _injury
            _HAS_NBAINJURIES = True
            _NBAINJURIES_IMPORT_ERROR = ""
        except Exception as exc:
            _NBAINJURIES_IMPORT_ERROR = str(exc)
            print("    [injury_report] WARNING: nbainjuries unavailable; running with no injury adjustments.")
            if _NBAINJURIES_IMPORT_ERROR:
                print(f"    [injury_report] Detail: {_NBAINJURIES_IMPORT_ERROR}")
            return {}
    
    df = None
    if dt is None:
        now = datetime.now()
        # Search backward hour-by-hour for the latest available report snapshot
        print("    [injury_report] Finding freshest injury report snapshot...")
        for i in range(24):
            try_dt = now - dt_module.timedelta(hours=i)
            # Try half-past the hour and top of the hour
            for minute in [30, 0]:
                dt_attempt = datetime(year=try_dt.year, month=try_dt.month, day=try_dt.day, hour=try_dt.hour, minute=minute)
                try:
                    df = injury.get_reportdata(dt_attempt, return_df=True)
                    if df is not None and not df.empty:
                        print(f"    [injury_report] Success: Pulled {dt_attempt.strftime('%I:%M %p')} report.")
                        break
                except Exception:
                    pass
            if df is not None and not df.empty:
                break
                
        # Fallback just in case
        if df is None or df.empty:
            dt = datetime(year=now.year, month=now.month, day=now.day, hour=17, minute=30)
            try:
                df = injury.get_reportdata(dt, return_df=True)
            except Exception as exc:
                print(f"    [injury_report] WARNING: fallback report fetch failed: {exc}")
                return {}
    else:
        try:
            df = injury.get_reportdata(dt, return_df=True)
        except Exception as exc:
            print(f"    [injury_report] WARNING: injury report fetch failed: {exc}")
            return {}

    if df is None or df.empty:
        return {}
    
    # Drop rows where Player Name is NaN (NOT YET SUBMITTED entries)
    df = df.dropna(subset=['Player Name'])
    
    injuries = {}
    for _, row in df.iterrows():
        full_team = row['Team']
        short = TEAM_SHORT.get(full_team, full_team)
        status = row.get('Current Status', '')
        
        # Skip "Available" players and G-League assignments
        reason = str(row.get('Reason', ''))
        if status == 'Available' or 'G League' in reason:
            continue
        
        if short not in injuries:
            injuries[short] = []
        
        # Reformat "Last, First" to "First Last"
        raw_name = row.get('Player Name', '')
        parts = raw_name.split(', ')
        name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else raw_name
        
        injuries[short].append({
            "name": name,
            "status": status,
            "reason": reason.replace("Injury/Illness - ", ""),
        })
    
    return injuries


def get_team_out_players(injuries: dict, team_name: str) -> list:
    """Get list of player names with status 'Out' for a team."""
    team_key = _normalize_team_name(team_name)
    return [
        p["name"] for p in injuries.get(team_key, [])
        if p["status"] == "Out"
    ]


def get_expected_injury_impact(injuries: dict, team_name: str) -> list[dict]:
    """
    Return all injured players for a team with an expected absence probability.

    Players with statuses outside the modeled probabilities are ignored so the
    NBANEW injury layer only consumes statuses with a defined expected value.
    """
    team_key = _normalize_team_name(team_name)
    expected_players = []
    for player in injuries.get(team_key, []):
        status = _normalize_status(player.get("status", ""))
        absence_probability = get_expected_absence_probability(status)
        if absence_probability <= 0.0:
            continue
        expected_players.append({
            "name": str(player.get("name", "")).strip(),
            "status": status,
            "reason": str(player.get("reason", "")).strip() or str(player.get("injury", "")).strip(),
            "absence_probability": absence_probability,
        })
    return expected_players


def print_injury_report(injuries: dict, teams: list = None):
    """Pretty print the injury report."""
    print(f"\n{'='*70}")
    print(f"📋 OFFICIAL NBA INJURY REPORT (via nbainjuries)")
    print(f"{'='*70}")
    
    teams_to_show = teams if teams else sorted(injuries.keys())
    for team in teams_to_show:
        if team in injuries and injuries[team]:
            print(f"\n  {team}:")
            for p in injuries[team]:
                emoji = "❌" if p["status"] == "Out" else "⚠️" if p["status"] == "Questionable" else "🟡"
                print(f"    {emoji} {p['name']} — {p['status']} ({p['reason']})")
        elif team in (teams or []):
            print(f"\n  {team}: ✅ Fully Healthy")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    injuries = fetch_injuries()
    if injuries:
        print_injury_report(injuries, [
            "Mavericks", "Grizzlies", "Nuggets", "Spurs",
            "Celtics", "Thunder", "Lakers", "Bulls"
        ])

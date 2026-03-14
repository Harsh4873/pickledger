"""
NBA Injury Report Fetcher — Uses the nbainjuries PyPI package.
Pulls directly from official NBA injury reports (no API key needed).
"""
from nbainjuries import injury
from datetime import datetime

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

def fetch_injuries(dt: datetime = None) -> dict:
    """
    Fetch official NBA injury report.
    
    Args:
        dt: datetime for the report snapshot (default: most recent available snapshot)
    
    Returns:
        Dict keyed by short team name -> list of injured players.
    """
    import datetime as dt_module
    
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
            df = injury.get_reportdata(dt, return_df=True)
    else:
        df = injury.get_reportdata(dt, return_df=True)
    
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
    return [
        p["name"] for p in injuries.get(team_name, [])
        if p["status"] == "Out"
    ]


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

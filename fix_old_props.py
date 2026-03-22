import json
import sqlite3
import argparse
from nba_api.stats.static import teams
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print changes without saving to DB')
    args = parser.parse_args()

    # Get team abbreviation to nickname mapping
    nba_teams = teams.get_teams()
    abbr_to_nickname = {t['abbreviation']: t['nickname'] for t in nba_teams}

    conn = sqlite3.connect('pickledger.db')
    # The user's DB has 'state_key' = 'primary'
    row = conn.execute("SELECT state_json FROM ledger_state WHERE state_key = 'primary'").fetchone()
    if not row:
        print("No 'primary' ledger state found. Checking latest...")
        row = conn.execute('SELECT state_json, state_key FROM ledger_state ORDER BY updated_at DESC LIMIT 1').fetchone()
        if not row:
            print("No ledger state found at all.")
            return
        state_json = row[0]
        state_key = row[1]
    else:
        state_json = row[0]
        state_key = 'primary'

    data = json.loads(state_json)
    
    # Pre-calculate Matchups for a date
    # Format: {'Mar 19': [{'home_abbr': 'WAS', 'away_abbr': 'DET', 'matchup': '(Pistons @ Wizards)'}, ...]}
    date_to_matchups = {}

    for pick in data.get('addedPicks', []):
        if pick.get('sport') == 'NBA' and pick.get('source') == 'NBA Model':
            date = pick.get('date')
            pick_text = pick.get('pick', '')
            if '(' in pick_text:
                suffix = pick_text[pick_text.find('('):]
                if '@' in suffix:
                    # e.g., "(Pistons @ Wizards)"
                    clean_suffix = suffix.strip('()') # Pistons @ Wizards
                    parts = clean_suffix.split(' @ ')
                    if len(parts) == 2:
                        away_nick, home_nick = parts
                        # Find abbreviations
                        away_abbr = next((abbr for abbr, nick in abbr_to_nickname.items() if nick == away_nick), None)
                        home_abbr = next((abbr for abbr, nick in abbr_to_nickname.items() if nick == home_nick), None)
                        
                        if away_abbr and home_abbr:
                            if date not in date_to_matchups:
                                date_to_matchups[date] = []
                            matchup_dict = {'home_abbr': home_abbr, 'away_abbr': away_abbr, 'matchup': suffix}
                            if matchup_dict not in date_to_matchups[date]:
                                date_to_matchups[date].append(matchup_dict)

    changes_made = 0
    props_to_update = []

    for pick in data.get('addedPicks', []):
        if pick.get('source') == 'NBA Props Model' and '(' not in pick.get('pick', ''):
            pick_text = pick.get('pick', '')
            parts = pick_text.split(' vs ')
            if len(parts) == 2:
                opponent_abbr = parts[1].strip()
                date = pick.get('date')
                
                # Find matching game in date_to_matchups
                matchup_suffix = None
                if date in date_to_matchups:
                    for game in date_to_matchups[date]:
                        if game['home_abbr'] == opponent_abbr or game['away_abbr'] == opponent_abbr:
                            matchup_suffix = game['matchup']
                            break
                            
                if not matchup_suffix:
                    # Fallback to just appending the opponent if no match found
                    print(f"Warning: No matchup found for {pick_text} on {date}")
                    matchup_suffix = f"(vs {abbr_to_nickname.get(opponent_abbr, opponent_abbr)})"
                
                # Apply changes
                old_pick = pick_text
                pick['pick'] = f"{pick_text} {matchup_suffix}"
                original_prob = pick.get('probability')
                if pick.get('probability') is None:
                    pick['probability'] = 0.55
                
                if args.dry_run and len(props_to_update) < 10:
                    print(f"Old Pick: {old_pick}")
                    print(f"New Pick: {pick['pick']} | Prob: {pick['probability']}")
                    print("-" * 50)
                elif args.dry_run and len(props_to_update) == 10:
                    print("... skipping remainder of dry-run output ...")
                
                props_to_update.append(pick)
                changes_made += 1

    print(f"Total props to update/updated: {changes_made}")

    if not args.dry_run and changes_made > 0:
        new_state_json = json.dumps(data)
        conn.execute('UPDATE ledger_state SET state_json = ? WHERE state_key = ?', (new_state_json, state_key))
        conn.commit()
        print(f"Successfully updated ledger in DB with {changes_made} changes.")

if __name__ == '__main__':
    main()

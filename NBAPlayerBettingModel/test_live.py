import sys
import live_data

print("Has injury report module?", live_data._HAS_INJURY_REPORT)
if live_data._HAS_INJURY_REPORT:
    inj = live_data.fetch_injuries()
    print("Injuries fetched directly:", len(inj), "teams")
    print("Is Cade out?", any('cade' in p['name'].lower() for t in inj.values() for p in t))

games, players, opp, df, bases, meta, season = live_data.load_props_slate(game_ids={'0022501009'})
print("Players returned:", len(players))
print("Is Cade in players?", any('cade' in p.player_name.lower() for p in players))

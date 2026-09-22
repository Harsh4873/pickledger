"""Official completed singles results when the workbook archive is unavailable."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

from .tennis_data import Match, player_key, RAW_DIR


def completed_matches(payload, tour, after, before, resolve_meta, round_order):
    matches = {}
    for event in payload.get("events", []):
        for group in event.get("groupings", []):
            if "singles" not in str(group.get("grouping", {}).get("slug", "")):
                continue
            for comp in group.get("competitions", []):
                day = str(comp.get("date", ""))[:10]
                status = comp.get("status", {}).get("type", {})
                if not after < day < before or status.get("name") != "STATUS_FINAL":
                    continue
                rnd = str(comp.get("round", {}).get("displayName", ""))
                if "qualif" in rnd.lower():
                    continue
                players = comp.get("competitors", [])
                if len(players) != 2 or any(not p.get("athlete", {}).get("displayName") for p in players):
                    continue
                wins = [p for p in players if p.get("winner") is True]
                if len(wins) != 1:
                    continue
                winner = wins[0]; loser = next(p for p in players if p is not winner)
                ws, ls = winner.get("linescores", []), loser.get("linescores", [])
                if len(ws) < 2 or len(ws) != len(ls):
                    continue
                meta = resolve_meta(tour, event.get("name", ""), date.fromisoformat(day), comp.get("venue", {}).get("fullName", ""))
                if meta.get("assumed"):
                    continue  # do not update surface ratings with guessed courts
                wn, ln = winner["athlete"]["displayName"], loser["athlete"]["displayName"]
                match = Match(day, tour, int(day[:4]), event.get("name", ""), "", meta.get("series", ""),
                    int(meta.get("tier", 2)), meta.get("court", "Outdoor"), meta["surface"], rnd, round_order(rnd),
                    int(meta.get("best_of", 3)), wn, ln, player_key(wn), player_key(ln), None, None, None, None,
                    sum(int(s["value"]) for s in ws), sum(int(s["value"]) for s in ls),
                    sum(s.get("winner") is True for s in ws), sum(s.get("winner") is True for s in ls), "completed")
                matches[(tour, comp.get("id"))] = match
    return matches


def fetch_completed_matches(after, before, resolve_meta, round_order, *, fetch=None, cache_dir=None):
    cache = Path(cache_dir) if cache_dir is not None else RAW_DIR / "espn_results"
    days = []
    day = date.fromisoformat(after) + timedelta(days=1)
    while day < date.fromisoformat(before):
        days.extend((tour, day.isoformat()) for tour in ("ATP", "WTA"))
        day += timedelta(days=1)
    def load(item):
        tour, day = item
        path = cache / f"{tour}-{day}.json"
        try:
            if path.exists():
                payload = json.loads(path.read_text())
            else:
                url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour.lower()}/scoreboard?dates={day.replace('-', '')}"
                if fetch:
                    payload = fetch(url)
                else:
                    response = requests.get(url, timeout=25); response.raise_for_status(); payload = response.json()
                if not isinstance(payload.get("events"), list):
                    raise ValueError("invalid scoreboard")
                # Empty answers may be a transient source gap; do not cache them.
                if payload["events"]:
                    cache.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload))
            return completed_matches(payload, tour, after, before, resolve_meta, round_order), None
        except Exception as exc:
            return {}, f"{tour} {day}: {exc}"
    rows, errors = {}, []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for matches, error in pool.map(load, days):
            rows.update(matches)
            if error:
                errors.append(error)
    return sorted(rows.values(), key=Match.sort_key), errors

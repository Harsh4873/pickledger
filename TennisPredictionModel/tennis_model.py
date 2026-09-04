"""Tennis serving model — daily match-winner picks for the cache.

The slate comes from the same ESPN ATP/WTA scoreboards the tennis scrapers
already use, so every pick this emits grades through the existing
``grade_tennis_picks`` path against ESPN's competitor-level ``winner`` flag.

Ratings are not recomputed from scratch at 6am. A pruned Elo/WElo snapshot is
committed with the model; serving reloads it and replays only the matches that
finished since — one small current-season workbook per tour — so a daily run
costs two HTTP requests instead of twenty-seven years of history.

What is published is the *market-free* model probability. Tennis moneylines are
not carried by the shared market-odds attachment (its scoreboard parser is
built for team sports, where competitors hang off ``event.competitions``
instead of tennis's ``groupings``), so these rows are unpriced by design and
say so. Confidence thresholds from the held-out backtest are recorded as
``source_decision`` / ``source_units``, but an unpriced row publishes as PASS
at 0u — a 0.98 model probability is not a bet until there is a price.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .tennis_core import (
    ARTIFACT_DIR,
    RATINGS_PATH,
    RatingEngine,
    to_vector,
)
from .tennis_data import (
    OOXML_FIRST_SEASON,
    SURFACES,
    Match,
    _strip_accents,
    ensure_season,
    parse_workbook,
    player_key,
    workbook_path,
)
from .tennis_infer import load_model

TOURNAMENT_INDEX_PATH = ARTIFACT_DIR / "tennis_tournaments.json"

# Confidence gates, read off the held-out threshold table in
# artifacts/backtest.json (2022-2026, 22,912 matches). Cumulative hit rate for
# the model's own pick:
#
#     >= 0.58   67% of the slate, 71.6% hit rate
#     >= 0.70   31% of the slate, 79.5% hit rate
#     >= 0.75   22% of the slate, 82.7% hit rate
#
# The gates select on hit rate rather than on expected value, because ROI is
# negative at every threshold and every book (best case -0.9% at the best price
# available anywhere) — see the README. Publishing these as unpriced,
# graded-on-result picks is the only claim the evidence supports.
BET_PROBABILITY = 0.70
LEAN_PROBABILITY = 0.58

# ESPN writes round names in prose; the ratings engine wants the draw position.
# Order matters and the specific cases must come first: "semifinals" and
# "quarterfinals" both contain "final", so a naive final-first scan promotes
# every semi to a final.
ROUND_HINTS = (
    # Qualifying first: "Qualifying 1st Round" also contains "1st round".
    ("qualif", 0),
    # Then the compound finals, because both contain "final".
    ("quarterfinal", 5),
    ("semifinal", 6),
    ("round robin", 1),
    ("round of 16", 4),
    ("4th round", 4),
    ("fourth round", 4),
    ("round of 32", 3),
    ("3rd round", 3),
    ("third round", 3),
    ("round of 64", 2),
    ("2nd round", 2),
    ("second round", 2),
    ("round of 128", 1),
    ("1st round", 1),
    ("first round", 1),
    ("final", 7),
)

# Last-resort surface guess by ISO week, for an event with no history anywhere.
# These are the modal surfaces measured over 2015+ in the archive, not a
# recollection of the calendar — the tour swings do not line up with month
# boundaries, and the two weeks *after* Wimbledon are back on clay (Hamburg,
# Palermo, Kitzbühel, Umag), which a "July means grass" rule gets wrong. The
# committed tournament index carries a freshly measured version of this table;
# this constant only covers the case where the index is missing entirely.
_DEFAULT_WEEK_SURFACE = tuple(
    ["Hard"] * 13 + ["Clay"] * 9 + ["Grass"] * 5 + ["Clay"] * 3 + ["Hard"] * 23
)


def _calendar_surface(target: date, index: dict[str, Any] | None = None) -> str:
    week = target.isocalendar()[1]
    measured = (index or {}).get("week_surface") or {}
    surface = measured.get(str(week))
    if surface in SURFACES:
        return str(surface)
    return _DEFAULT_WEEK_SURFACE[min(week, len(_DEFAULT_WEEK_SURFACE)) - 1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def load_tournament_index(path: Path = TOURNAMENT_INDEX_PATH) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        return {"tournaments": {}, "locations": {}, "week_surface": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"tournaments": {}, "locations": {}}
    return {
        "tournaments": payload.get("tournaments") or {},
        "locations": payload.get("locations") or {},
        "week_surface": payload.get("week_surface") or {},
    }


def build_tournament_index(matches: list[Match]) -> dict[str, Any]:
    """Most recent surface/tier/court/best-of per tournament, per tour.

    Built at train time and committed, so the daily job can classify a court it
    has not yet seen a result from this week without parsing history. Indexed by
    host city as well as by name: sponsor names churn every year (ESPN's
    "Palermo Ladies Open" is the archive's "Internazionali Femminili di
    Palermo"), and getting the surface wrong is not cosmetic — it swings the
    surface-Elo features and moved one test prediction from 0.69 to 0.55.

    "Most recent wins" is deliberate: events do move surface (Prague ran on clay
    in 2024 and hard in 2025), so the latest observation is the right guess.
    """
    index: dict[str, dict[str, Any]] = {}
    by_location: dict[str, dict[str, Any]] = {}
    for match in matches:
        entry = {
            "date": match.date,
            "surface": match.surface,
            "tier": match.tier,
            "series": match.series,
            "court": match.court,
            "best_of": match.best_of,
            "tour": match.tour,
            "tournament": match.tournament,
            "location": match.location,
        }
        key = f"{match.tour}|{_normalise(match.tournament)}"
        if key not in index or match.date > index[key]["date"]:
            index[key] = entry
        city = _normalise(match.location)
        if city:
            location_key = f"{match.tour}|{city}"
            if location_key not in by_location or match.date > by_location[location_key]["date"]:
                by_location[location_key] = entry
    return {
        "schema": 3,
        "built_at": _now_iso(),
        "tournaments": index,
        "locations": by_location,
        "week_surface": _measure_week_surfaces(matches),
    }


def _measure_week_surfaces(matches: list[Match], since_season: int = 2015) -> dict[str, str]:
    """Modal surface per ISO week, measured rather than assumed."""
    counts: dict[int, dict[str, int]] = {}
    for match in matches:
        if match.season < since_season:
            continue
        try:
            week = date.fromisoformat(match.date).isocalendar()[1]
        except ValueError:
            continue
        bucket = counts.setdefault(week, {})
        bucket[match.surface] = bucket.get(match.surface, 0) + 1
    return {
        str(week): max(bucket.items(), key=lambda item: item[1])[0]
        for week, bucket in counts.items()
        if bucket
    }


def _venue_city(venue: str) -> str:
    """"Kitzbühel, Austria" -> "kitzbuhel"; the archive stores bare cities."""
    head = str(venue or "").split(",")[0]
    return _normalise(_strip_accents(head))


def _tournament_meta(
    index: dict[str, dict[str, dict[str, Any]]],
    tour: str,
    tournament: str,
    target: date,
    venue: str = "",
) -> dict[str, Any]:
    """Resolve surface/tier/best-of for a live event, most reliable route first."""
    tournaments = index.get("tournaments") or {}
    locations = index.get("locations") or {}
    other = "WTA" if tour == "ATP" else "ATP"
    needle = _normalise(tournament)
    city = _venue_city(venue)

    def cross_tour(entry: dict[str, Any]) -> dict[str, Any]:
        # Best-of is a tour property (men play five sets at the Slams), so it
        # must not be inherited across tours; the surface can be.
        return {**entry, "tour": tour, "best_of": 3 if tour == "WTA" else entry.get("best_of", 3)}

    if needle:
        entry = tournaments.get(f"{tour}|{needle}")
        if entry:
            return entry
    if city:
        entry = locations.get(f"{tour}|{city}")
        if entry:
            return entry
    if needle:
        entry = tournaments.get(f"{other}|{needle}")
        if entry:
            return cross_tour(entry)
    if city:
        entry = locations.get(f"{other}|{city}")
        if entry:
            return cross_tour(entry)
    if needle:
        # Loose containment ("Wimbledon" vs "The Championships, Wimbledon").
        for candidate_key, candidate in tournaments.items():
            candidate_tour, _, candidate_name = candidate_key.partition("|")
            if candidate_tour != tour or not candidate_name:
                continue
            if candidate_name in needle or needle in candidate_name:
                return candidate
    return {
        "surface": _calendar_surface(target, index),
        "tier": 2,
        "series": "",
        "court": "Outdoor",
        "best_of": 3,
        "tour": tour,
        "tournament": tournament,
        "assumed": True,
    }


def _round_order(round_name: str) -> int:
    text = _normalise(round_name)
    for needle, order in ROUND_HINTS:
        if needle in text:
            return order
    return 2


def catch_up_ratings(engine: RatingEngine, through_date: str, *, download: bool = True) -> dict[str, Any]:
    """Replay results that landed after the snapshot was taken.

    Only the current (and, across a new year boundary, the previous) season is
    fetched, and only the OOXML era is reachable — which is all that matters,
    because a snapshot is never older than the season it shipped in.
    """
    if not through_date:
        return {"applied": 0, "through": engine.last_date}
    try:
        snapshot_date = date.fromisoformat(through_date)
    except ValueError:
        return {"applied": 0, "through": engine.last_date}
    seasons = sorted({snapshot_date.year, date.today().year})
    fresh: list[Match] = []
    for season in seasons:
        if season < OOXML_FIRST_SEASON:
            continue
        for tour in ("ATP", "WTA"):
            # `download=False` means "don't hit the network", not "ignore the
            # cache" — an already-downloaded workbook is still good history.
            path = ensure_season(tour, season) if download else workbook_path(tour, season)
            if path is None or not path.exists():
                continue
            try:
                fresh.extend(parse_workbook(path, tour, season))
            except Exception as exc:  # a stale snapshot beats a crashed slate
                print(f"[tennis] catch-up parse failed for {tour} {season}: {exc}")
    pending = sorted(
        (match for match in fresh if match.date > through_date),
        key=Match.sort_key,
    )
    for match in pending:
        engine.update(match)
    return {"applied": len(pending), "through": engine.last_date}


def _decision(probability: float) -> str:
    if probability >= BET_PROBABILITY:
        return "BET"
    if probability >= LEAN_PROBABILITY:
        return "LEAN"
    return "PASS"


def _confidence_units(probability: float) -> float:
    return 1.0 if probability >= BET_PROBABILITY else 0.5


def generate_tennis_picks(
    date_iso: str | None = None,
    *,
    fetch_json: Callable[[str], Any] | None = None,
    download: bool = True,
) -> dict[str, Any]:
    """Build the in-house tennis slate for ``date_iso``."""
    target_iso = str(date_iso or date.today().isoformat())
    try:
        target = date.fromisoformat(target_iso)
    except ValueError:
        return {"ok": False, "error": f"invalid date {target_iso!r}"}

    model = load_model()
    engine = RatingEngine.load(RATINGS_PATH)
    if model is None or engine is None:
        return {
            "ok": True,
            "date": target_iso,
            "model": "TennisElo",
            "picks": [],
            "matches": [],
            "note": "Tennis artifacts not trained yet; emitting an empty slate.",
        }

    # The model was trained on features produced under one set of rating
    # dynamics. Training writes the model and the ratings snapshot together, so
    # a mismatch means one was regenerated without the other — the predictions
    # would be subtly off rather than obviously broken, which is exactly the
    # kind of drift worth surfacing rather than swallowing. Not fatal: stale
    # dynamics still beat no slate at all.
    trained_config = (model.metadata.get("elo_config") or {}) if model.metadata else {}
    live_config = engine.config.to_dict()
    config_matches = not trained_config or all(
        abs(float(value) - float(live_config.get(key, value))) < 1e-9
        for key, value in trained_config.items()
    )
    if not config_matches:
        print("[tennis] ratings snapshot Elo config differs from the trained config; retrain to realign")

    snapshot_through = engine.last_date
    catch_up = catch_up_ratings(engine, snapshot_through, download=download)
    index = load_tournament_index()

    # Imported lazily: the scraper pulls in requests/bs4, which the rating and
    # training layers deliberately do not need.
    from scripts.scrapers.tennis_scraper import espn_tennis_matches  # noqa: PLC0415

    try:
        slate, resolved = espn_tennis_matches(target_iso, fetch_json=fetch_json)
    except Exception as exc:
        return {"ok": False, "date": target_iso, "error": f"ESPN slate fetch failed: {exc}"}
    if not resolved:
        return {"ok": False, "date": target_iso, "error": "ESPN tennis scoreboards did not resolve"}

    ordinal = target.toordinal()
    picks: list[dict[str, Any]] = []
    matches_out: list[dict[str, Any]] = []
    unknown_players = 0

    for entry in slate:
        away = str(entry.get("away") or "")
        home = str(entry.get("home") or "")
        if not away or not home:
            continue
        away_key = player_key(away)
        home_key = player_key(home)
        if not away_key or not home_key or away_key == home_key:
            continue
        tour = str(entry.get("tour") or "ATP").upper()
        tournament = str(entry.get("tournament") or "")
        meta = _tournament_meta(index, tour, tournament, target, str(entry.get("venue") or ""))
        round_order = _round_order(str(entry.get("round_display") or ""))

        p1_key, p2_key = sorted((away_key, home_key))
        p1_name = away if p1_key == away_key else home
        p2_name = home if p1_key == away_key else away

        away_state = engine.players.get(away_key)
        home_state = engine.players.get(home_key)
        if away_state is None or home_state is None:
            unknown_players += 1

        features = engine.features(
            p1_key,
            p2_key,
            surface=str(meta.get("surface") or "Hard"),
            best_of=int(meta.get("best_of") or 3),
            tier=int(meta.get("tier") or 2),
            round_order=round_order,
            indoor=str(meta.get("court") or "").lower() == "indoor",
            tour=tour,
            ordinal=ordinal,
            # Must be the *archive* event key, not ESPN's. The replay stamps
            # players with "season:tour:<archive tournament>", so keying off
            # ESPN's sponsor name ("Palermo Ladies Open" vs "Internazionali
            # Femminili di Palermo") looks like a different event and silently
            # zeroes `event_games_diff` for the whole slate — a feature that
            # carried real values throughout training.
            event_key=f"{target.year}:{tour}:{meta.get('tournament') or tournament}",
        )
        p1_probability = model.predict(to_vector(features))

        pick_p1 = p1_probability >= 0.5
        selected = p1_name if pick_p1 else p2_name
        probability = p1_probability if pick_p1 else 1.0 - p1_probability
        confidence_decision = _decision(probability)
        confidence_units = _confidence_units(probability)
        matchup = f"{away} vs {home}"

        picks.append({
            "source": "Tennis Model",
            "pick": f"{selected} ML ({matchup})",
            "tip": f"{selected} to win",
            "sport": "Tennis",
            "league": tour,
            "espn_league": entry.get("league"),
            "date": target_iso,
            "matchup": matchup,
            "game": matchup,
            "away_team": away,
            "home_team": home,
            "selected_player": selected,
            "opponent": p2_name if pick_p1 else p1_name,
            "start_time": entry.get("start_time"),
            "round": entry.get("round_display") or entry.get("round"),
            "tournament": tournament or None,
            "surface": meta.get("surface"),
            "best_of": int(meta.get("best_of") or 3),
            "odds": None,
            "units": 0,
            "probability": round(probability, 4),
            "model_probability": round(probability, 4),
            "edge": None,
            "decision": "PASS",
            "source_decision": confidence_decision,
            "source_units": confidence_units,
            "market_type": "tennis_moneyline",
            "grade_supported": True,
            # Tennis moneylines are not covered by the shared odds attachment,
            # so there is no price to devig and no edge to claim. Confidence
            # is preserved on source_* so the row stays auditable.
            "pricing_type": "unpriced",
            "market_priced": False,
            # No tennis calibration model exists yet; these rows carry a real
            # probability, so wiring them in is a follow-up rather than a gap.
            "calibration_excluded": True,
            "model_version": model.version,
            "ratings_through": engine.last_date,
        })

        matches_out.append({
            "matchup": matchup,
            "tour": tour,
            "tournament": tournament,
            "surface": meta.get("surface"),
            "round": entry.get("round"),
            "p1": p1_name,
            "p2": p2_name,
            "round_display": entry.get("round_display"),
            "p1_probability": round(p1_probability, 4),
            "elo_diff": round(features["elo_diff"], 1),
            "surface_elo_diff": round(features["surface_elo_diff"], 1),
            "welo_diff": round(features["welo_diff"], 1),
            "assumed_tournament_meta": bool(meta.get("assumed")),
        })

    decided = sum(1 for pick in picks if pick["decision"] in {"BET", "LEAN"})
    return {
        "ok": True,
        "date": target_iso,
        "model": "TennisElo",
        "model_version": model.version,
        "generatedAt": _now_iso(),
        "picks": picks,
        "matches": matches_out,
        "meta": {
            "officialMatchups": len(slate),
            "rated": len(picks),
            "decided": decided,
            "unknownPlayers": unknown_players,
            "ratingsSnapshotThrough": snapshot_through,
            "ratingsThrough": engine.last_date,
            "catchUpMatches": catch_up["applied"],
            "betThreshold": BET_PROBABILITY,
            "leanThreshold": LEAN_PROBABILITY,
            "ratingConfigMatchesModel": config_matches,
        },
        "note": (
            f"Tennis model: {len(picks)} rated match(es), {decided} actionable; "
            f"ratings through {engine.last_date}."
        ),
    }


if __name__ == "__main__":
    import sys

    result = generate_tennis_picks(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({key: value for key, value in result.items() if key != "picks"}, indent=2)[:4000])
    for row in result.get("picks", [])[:10]:
        print(f"  {row['decision']:5s} {row['probability']:.3f}  {row['pick']}")

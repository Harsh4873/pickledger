"""Market-priced MLB hits, H+R+RBI, and pitcher strikeout props."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .schema import (
    american_implied_probability,
    build_pick,
    normal_probability,
    normalize_name,
    safe_float,
    safe_int,
)
from .ml import apply_ml_to_pick, ev_sort_key, market_family_for_stat, select_top_props


PARK_FACTORS = {
    19: 1.18,
    2: 1.07,
    1: 1.06,
    2602: 1.05,
    3: 1.04,
    17: 1.04,
    7: 1.03,
    22: 0.98,
    32: 0.97,
    4: 0.96,
    2680: 0.95,
    12: 0.94,
    2395: 0.92,
    2889: 0.91,
}

PITCH_LABELS = {
    "FF": "four-seamers",
    "SI": "sinkers",
    "FC": "cutters",
    "SL": "sliders",
    "ST": "sweepers",
    "CU": "curves",
    "KC": "knuckle curves",
    "CH": "changeups",
    "FS": "splitters",
    "SV": "slurves",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {
    "foul",
    "foul_bunt",
    "foul_pitchout",
    "hit_into_play",
    "missed_bunt",
}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
OUT_EVENTS = {
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "fielders_choice_out",
    "sac_fly",
    "sac_bunt",
    "strikeout",
    "strikeout_double_play",
}
LEAGUE_WHIFF_PER_SWING = 0.245
LEAGUE_K_PA_RATE = 0.225
LEAGUE_HIT_PA_RATE = 0.235
MAX_PITCHER_STRIKEOUT_LINE = 12.5

MLB_MARKET_TYPES = {
    "totalhits": ("hits", "Hits", "batter", True),
    "hits": ("hits", "Hits", "batter", True),
    "hitmilestones": ("hits", "Hits", "batter", True),
    "hitsmilestones": ("hits", "Hits", "batter", True),
    "totalhitsrunsrbis": ("hits_runs_rbis", "Hits + Runs + RBIs", "batter", True),
    "hitsrunsrbis": ("hits_runs_rbis", "Hits + Runs + RBIs", "batter", True),
    "hitsrunsrbismilestones": ("hits_runs_rbis", "Hits + Runs + RBIs", "batter", True),
    "hrr": ("hits_runs_rbis", "Hits + Runs + RBIs", "batter", True),
    "totalruns": ("runs", "Runs", "batter", True),
    "runs": ("runs", "Runs", "batter", True),
    "runsmilestones": ("runs", "Runs", "batter", True),
    "totalrbis": ("rbis", "RBIs", "batter", True),
    "rbis": ("rbis", "RBIs", "batter", True),
    "rbimilestones": ("rbis", "RBIs", "batter", True),
    "rbismilestones": ("rbis", "RBIs", "batter", True),
    "totalwalksbatter": ("batter_walks", "Walks", "batter", True),
    "batterwalks": ("batter_walks", "Walks", "batter", True),
    "walksbatter": ("batter_walks", "Walks", "batter", True),
    "walksbattermilestones": ("batter_walks", "Walks", "batter", True),
    "totalstrikeoutsbatter": ("batter_strikeouts", "Batter Strikeouts", "batter", True),
    "batterstrikeouts": ("batter_strikeouts", "Batter Strikeouts", "batter", True),
    "strikeoutsbatter": ("batter_strikeouts", "Batter Strikeouts", "batter", True),
    "strikeoutsbattermilestones": ("batter_strikeouts", "Batter Strikeouts", "batter", True),
    "totaltotalbases": ("total_bases", "Total Bases", "batter", True),
    "totalbases": ("total_bases", "Total Bases", "batter", True),
    "totalbasesmilestones": ("total_bases", "Total Bases", "batter", True),
    "singles": ("singles", "Singles", "batter", True),
    "totalsingles": ("singles", "Singles", "batter", True),
    "totalsingleshit": ("singles", "Singles", "batter", True),
    "singlesmilestones": ("singles", "Singles", "batter", True),
    "doubles": ("doubles", "Doubles", "batter", True),
    "totaldoubles": ("doubles", "Doubles", "batter", True),
    "totaldoubleshit": ("doubles", "Doubles", "batter", True),
    "doublesmilestones": ("doubles", "Doubles", "batter", True),
    "triples": ("triples", "Triples", "batter", True),
    "totaltriples": ("triples", "Triples", "batter", True),
    "triplesmilestones": ("triples", "Triples", "batter", True),
    "homeruns": ("home_runs", "Home Runs", "batter", True),
    "totalhomeruns": ("home_runs", "Home Runs", "batter", True),
    "homerunmilestones": ("home_runs", "Home Runs", "batter", True),
    "homerunsmilestones": ("home_runs", "Home Runs", "batter", True),
    "stolenbases": ("stolen_bases", "Stolen Bases", "batter", True),
    "totalstolenbases": ("stolen_bases", "Stolen Bases", "batter", True),
    "stolenbasesmilestones": ("stolen_bases", "Stolen Bases", "batter", True),
    "totalstrikeouts": ("strikeouts", "Strikeouts", "pitcher", True),
    "pitcherstrikeouts": ("strikeouts", "Strikeouts", "pitcher", True),
    "strikeouts": ("strikeouts", "Strikeouts", "pitcher", True),
    "strikeoutsthrownmilestones": ("strikeouts", "Strikeouts", "pitcher", True),
    "totalwalksallowed": ("pitcher_walks_allowed", "Walks Allowed", "pitcher", True),
    "pitcherwalks": ("pitcher_walks_allowed", "Walks Allowed", "pitcher", True),
    "pitcherwalksallowed": ("pitcher_walks_allowed", "Walks Allowed", "pitcher", True),
    "totaloutsrecorded": ("pitcher_outs_recorded", "Outs Recorded", "pitcher", True),
    "pitcherouts": ("pitcher_outs_recorded", "Outs Recorded", "pitcher", True),
    "pitcheroutsrecorded": ("pitcher_outs_recorded", "Outs Recorded", "pitcher", True),
    "totalhitsallowed": ("pitcher_hits_allowed", "Hits Allowed", "pitcher", True),
    "pitcherhitsallowed": ("pitcher_hits_allowed", "Hits Allowed", "pitcher", True),
    "totalearnedrunsallowed": ("pitcher_earned_runs_allowed", "Earned Runs Allowed", "pitcher", True),
    "earnedrunsallowed": ("pitcher_earned_runs_allowed", "Earned Runs Allowed", "pitcher", True),
    "pitcherearnedruns": ("pitcher_earned_runs_allowed", "Earned Runs Allowed", "pitcher", True),
    "pitcherearnedrunsallowed": ("pitcher_earned_runs_allowed", "Earned Runs Allowed", "pitcher", True),
    "pitchertorecordwin": ("pitcher_win", "Pitcher To Record Win", "pitcher", False),
}

_MLB_TEAM_NAME_ALIASES = {
    "alallstars": "americanleagueallstars",
    "americanallstars": "americanleagueallstars",
    "americanleagueallstars": "americanleagueallstars",
    "nlallstars": "nationalleagueallstars",
    "nationalallstars": "nationalleagueallstars",
    "nationalleagueallstars": "nationalleagueallstars",
}

MLB_TEAM_ABBREVIATIONS = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pitch_label(pitch_type: str) -> str:
    return PITCH_LABELS.get(str(pitch_type or "").strip().upper(), str(pitch_type or "unknown pitch").upper())


def _blank_pitch_profile() -> dict[str, Any]:
    return {
        "sample_pitches": 0,
        "sample_pa": 0,
        "sample_swings": 0,
        "mix": {},
        "by_pitch": {},
        "overall_k_rate": None,
        "overall_whiff_rate": None,
    }


def _summarize_pitch_rows(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    profile = _blank_pitch_profile()
    by_pitch: dict[str, dict[str, int]] = {}
    total_pitches = 0
    total_pa = 0
    total_k = 0
    total_swings = 0
    total_whiffs = 0

    for row in rows or []:
        pitch_type = str(row.get("pitch_type") or "").strip().upper()
        if not pitch_type:
            continue
        total_pitches += 1
        bucket = by_pitch.setdefault(
            pitch_type,
            {"pitches": 0, "swings": 0, "whiffs": 0, "pa": 0, "strikeouts": 0, "hits": 0, "outs": 0},
        )
        bucket["pitches"] += 1
        description = str(row.get("description") or "").strip().lower()
        event = str(row.get("events") or "").strip().lower()
        if description in SWING_DESCRIPTIONS:
            bucket["swings"] += 1
            total_swings += 1
        if description in WHIFF_DESCRIPTIONS:
            bucket["whiffs"] += 1
            total_whiffs += 1
        if event:
            bucket["pa"] += 1
            total_pa += 1
        if event in STRIKEOUT_EVENTS:
            bucket["strikeouts"] += 1
            total_k += 1
        if event in HIT_EVENTS:
            bucket["hits"] += 1
        if event in OUT_EVENTS:
            bucket["outs"] += 1

    profile["sample_pitches"] = total_pitches
    profile["sample_pa"] = total_pa
    profile["sample_swings"] = total_swings
    profile["mix"] = {
        pitch_type: bucket["pitches"] / total_pitches
        for pitch_type, bucket in by_pitch.items()
        if total_pitches
    }
    profile["by_pitch"] = by_pitch
    profile["overall_k_rate"] = (total_k / total_pa) if total_pa else None
    profile["overall_whiff_rate"] = (total_whiffs / total_swings) if total_swings else None
    return profile


def _profile_from_player_statcast(
    client: Any,
    player_id: int,
    player_type: str,
    date_iso: str,
    days: int = 45,
) -> dict[str, Any]:
    method = getattr(client, "mlb_statcast_player_pitches", None)
    if not callable(method) or not player_id:
        return _blank_pitch_profile()
    try:
        return _summarize_pitch_rows(method(player_id, player_type, date_iso, days=days))
    except Exception:
        return _blank_pitch_profile()


def _team_statcast_rows(client: Any, team_abbr: str, date_iso: str, days: int = 30) -> list[dict[str, Any]]:
    method = getattr(client, "mlb_statcast_team_pitches", None)
    if not callable(method) or not team_abbr:
        return []
    try:
        return method(team_abbr, date_iso, days=days)
    except Exception:
        return []


def _profile_for_batter_from_team_rows(rows: list[dict[str, Any]], batter_id: int) -> dict[str, Any]:
    if not batter_id:
        return _blank_pitch_profile()
    return _summarize_pitch_rows([
        row for row in rows
        if safe_int(row.get("batter")) == batter_id
    ])


def _team_abbreviation(feed: dict[str, Any], side: str, fallback_name: str) -> str:
    team = (((feed.get("gameData") or {}).get("teams") or {}).get(side) or {})
    abbreviation = str(team.get("abbreviation") or "").strip().upper()
    if abbreviation:
        return abbreviation
    return MLB_TEAM_ABBREVIATIONS.get(str(fallback_name or "").strip(), "")


def _top_pitch_mix(profile: dict[str, Any], limit: int = 3) -> str:
    mix = profile.get("mix") if isinstance(profile, dict) else {}
    if not isinstance(mix, dict) or not mix:
        return "pitch mix unavailable"
    parts = [
        f"{_pitch_label(pitch_type)} {share:.0%}"
        for pitch_type, share in sorted(mix.items(), key=lambda item: safe_float(item[1]), reverse=True)[:limit]
    ]
    return ", ".join(parts)


def _pitcher_arsenal_signal(profile: dict[str, Any]) -> tuple[float, str]:
    swings = safe_int(profile.get("sample_swings") if isinstance(profile, dict) else 0)
    whiff_rate = profile.get("overall_whiff_rate") if isinstance(profile, dict) else None
    if swings < 35 or whiff_rate is None:
        return 1.0, "Pitcher pitch-mix whiff sample unavailable"
    whiff = safe_float(whiff_rate)
    factor = 1.0 + _clamp(((whiff - LEAGUE_WHIFF_PER_SWING) / 0.12) * 0.055, -0.05, 0.07)
    return factor, f"Pitcher recent arsenal {_top_pitch_mix(profile)} with {whiff:.1%} whiffs/swing"


def _pitch_type_k_signal(
    pitcher_profile: dict[str, Any],
    target_profile: dict[str, Any],
    target_label: str,
    min_pitch_sample: int = 12,
) -> tuple[float, str]:
    mix = pitcher_profile.get("mix") if isinstance(pitcher_profile, dict) else {}
    by_pitch = target_profile.get("by_pitch") if isinstance(target_profile, dict) else {}
    if not isinstance(mix, dict) or not isinstance(by_pitch, dict) or not mix or not by_pitch:
        return 1.0, f"{target_label} pitch-type strikeout sample unavailable"

    score = 0.0
    weight = 0.0
    sample = 0
    details: list[str] = []
    for pitch_type, share_raw in sorted(mix.items(), key=lambda item: safe_float(item[1]), reverse=True)[:4]:
        share = safe_float(share_raw)
        if share < 0.08:
            continue
        bucket = by_pitch.get(pitch_type)
        if not bucket or safe_int(bucket.get("pitches")) < min_pitch_sample:
            continue
        swings = safe_int(bucket.get("swings"))
        whiffs = safe_int(bucket.get("whiffs"))
        pa = safe_int(bucket.get("pa"))
        strikeouts = safe_int(bucket.get("strikeouts"))
        whiff_rate = whiffs / swings if swings else LEAGUE_WHIFF_PER_SWING
        k_rate = strikeouts / pa if pa else LEAGUE_K_PA_RATE
        vulnerability = (
            0.65 * ((whiff_rate - LEAGUE_WHIFF_PER_SWING) / 0.12)
            + 0.35 * ((k_rate - LEAGUE_K_PA_RATE) / 0.14)
        )
        score += share * _clamp(vulnerability, -2.0, 2.0)
        weight += share
        sample += safe_int(bucket.get("pitches"))
        details.append(f"{_pitch_label(pitch_type)} {share:.0%} mix: {whiff_rate:.0%} whiff, {k_rate:.0%} K-ending PA")

    if not weight:
        return 1.0, f"{target_label} has no matched pitch-type sample against this arsenal"
    normalized = score / weight
    factor = 1.0 + _clamp(normalized * 0.115, -0.12, 0.14)
    direction = "vulnerable" if factor > 1.025 else "resistant" if factor < 0.975 else "neutral"
    return factor, f"{target_label} pitch-type K matchup {direction} ({sample} pitches): " + "; ".join(details[:2])


def _pitch_type_hit_signal(
    pitcher_profile: dict[str, Any],
    batter_profile: dict[str, Any],
    min_pitch_sample: int = 10,
) -> tuple[float, str]:
    mix = pitcher_profile.get("mix") if isinstance(pitcher_profile, dict) else {}
    by_pitch = batter_profile.get("by_pitch") if isinstance(batter_profile, dict) else {}
    if not isinstance(mix, dict) or not isinstance(by_pitch, dict) or not mix or not by_pitch:
        return 1.0, "Batter pitch-type hit sample unavailable"

    score = 0.0
    weight = 0.0
    sample = 0
    details: list[str] = []
    for pitch_type, share_raw in sorted(mix.items(), key=lambda item: safe_float(item[1]), reverse=True)[:4]:
        share = safe_float(share_raw)
        if share < 0.08:
            continue
        bucket = by_pitch.get(pitch_type)
        if not bucket or safe_int(bucket.get("pitches")) < min_pitch_sample:
            continue
        pa = safe_int(bucket.get("pa"))
        swings = safe_int(bucket.get("swings"))
        hits = safe_int(bucket.get("hits"))
        outs = safe_int(bucket.get("outs"))
        whiffs = safe_int(bucket.get("whiffs"))
        hit_rate = hits / pa if pa else LEAGUE_HIT_PA_RATE
        out_rate = outs / pa if pa else 0.66
        whiff_rate = whiffs / swings if swings else LEAGUE_WHIFF_PER_SWING
        contact_score = (
            0.55 * ((hit_rate - LEAGUE_HIT_PA_RATE) / 0.12)
            - 0.30 * ((whiff_rate - LEAGUE_WHIFF_PER_SWING) / 0.12)
            - 0.15 * ((out_rate - 0.66) / 0.16)
        )
        score += share * _clamp(contact_score, -2.0, 2.0)
        weight += share
        sample += safe_int(bucket.get("pitches"))
        details.append(f"{_pitch_label(pitch_type)} {share:.0%} mix: {hit_rate:.0%} hit-ending PA, {whiff_rate:.0%} whiff")

    if not weight:
        return 1.0, "Batter has no matched pitch-type hit sample against this arsenal"
    normalized = score / weight
    factor = 1.0 + _clamp(normalized * 0.085, -0.10, 0.10)
    direction = "handles arsenal" if factor > 1.025 else "vulnerable to arsenal" if factor < 0.975 else "neutral"
    return factor, f"Batter pitch-type hit matchup {direction} ({sample} pitches): " + "; ".join(details[:2])


def _binomial_over_probability(per_trial: float, trials: float, line: float) -> float:
    p = _clamp(per_trial, 0.01, 0.65)
    n = max(1, int(round(trials)))
    threshold = max(1, int(math.floor(line)) + 1)
    probability = 0.0
    for successes in range(threshold, n + 1):
        probability += math.comb(n, successes) * (p ** successes) * ((1.0 - p) ** (n - successes))
    return _clamp(probability, 0.01, 0.99)


def _american_odds(value: Any) -> int | None:
    text = str(value or "").strip().replace("+", "")
    if not text:
        return None
    try:
        odds = int(float(text))
    except (TypeError, ValueError):
        return None
    return odds if odds else None


def _canonical_market_name(value: str) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _canonical_team_name(value: str) -> str:
    normalized = normalize_name(value)
    return _MLB_TEAM_NAME_ALIASES.get(normalized, normalized)


def _is_milestone_market(type_name: str, display: str) -> bool:
    return "milestone" in str(type_name or "").lower() or "+" in str(display or "")


def _espn_event_market(
    scoreboard: dict[str, Any],
    game: dict[str, Any],
) -> tuple[str, str, str] | None:
    away_target = _canonical_team_name(game["away_team"])
    home_target = _canonical_team_name(game["home_team"])
    for event in scoreboard.get("events") or []:
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors") or []
        away = next((row for row in competitors if row.get("homeAway") == "away"), {})
        home = next((row for row in competitors if row.get("homeAway") == "home"), {})
        if _canonical_team_name((away.get("team") or {}).get("displayName")) != away_target:
            continue
        if _canonical_team_name((home.get("team") or {}).get("displayName")) != home_target:
            continue
        odds_rows = competition.get("odds") or []
        odds = odds_rows[0] if odds_rows else {}
        provider = odds.get("provider") or {}
        provider_id = str(provider.get("id") or "100")
        provider_name = str(provider.get("name") or "DraftKings").strip()
        return str(event.get("id") or ""), provider_id, f"{provider_name} via ESPN"
    return None


def _summary_athlete_names(summary: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for team in summary.get("rosters") or []:
        for row in team.get("roster") or []:
            athlete = row.get("athlete") or {}
            athlete_id = str(athlete.get("id") or "")
            name = str(athlete.get("displayName") or athlete.get("fullName") or "")
            if athlete_id and name:
                names[athlete_id] = name
    return names


def _athlete_ref_id(row: dict[str, Any]) -> str:
    ref = str((row.get("athlete") or {}).get("$ref") or "").split("?", 1)[0].rstrip("/")
    return ref.rsplit("/", 1)[-1] if "/athletes/" in ref else ""


def _athlete_name(payload: dict[str, Any]) -> str:
    athlete = payload.get("athlete") if isinstance(payload.get("athlete"), dict) else payload
    return str(athlete.get("displayName") or athlete.get("fullName") or "").strip()


def _market_line(row: dict[str, Any]) -> float:
    current = row.get("current") or {}
    target = current.get("target") or {}
    odds = row.get("odds") or {}
    total = odds.get("total") or {}
    return safe_float(target.get("value") if target.get("value") is not None else total.get("value"))


def _market_display(row: dict[str, Any]) -> str:
    current = row.get("current") or {}
    target = current.get("target") or {}
    odds = row.get("odds") or {}
    total = odds.get("total") or {}
    return str(target.get("displayValue") or total.get("value") or target.get("value") or "").strip()


def _market_index(
    items: list[dict[str, Any]],
    athlete_names: dict[str, str],
    source: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        type_name = str((row.get("type") or {}).get("name") or "")
        market_type = MLB_MARKET_TYPES.get(_canonical_market_name(type_name))
        athlete_id = _athlete_ref_id(row)
        line = _market_line(row)
        if not market_type or not athlete_id or line <= 0:
            continue
        display = _market_display(row)
        stat_key, stat_label, role, grade_supported = market_type
        if _is_milestone_market(type_name, display):
            name = athlete_names.get(athlete_id)
            over_odds = _american_odds((((row.get("odds") or {}).get("american") or {}).get("value")))
            if not name or over_odds is None:
                continue
            threshold = line
            grouped_key = (athlete_id, f"{type_name}__milestone", threshold, type_name)
            grouped[grouped_key].append(
                {
                    **row,
                    "_parsed_market": {
                        "stat_key": stat_key,
                        "stat_label": stat_label,
                        "line": max(0.0, threshold - 0.5),
                        "over_odds": over_odds,
                        "market_role": role,
                        "grade_supported": grade_supported,
                        "market_format": "milestone",
                        "market_threshold": display or f"{threshold:g}+",
                    },
                }
            )
            continue
        grouped[(athlete_id, stat_key, line, type_name)].append(row)

    markets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (athlete_id, stat_key_or_marker, line, type_name), sides in grouped.items():
        name = athlete_names.get(athlete_id)
        if not name:
            continue
        parsed = sides[0].get("_parsed_market") if isinstance(sides[0], dict) else None
        if isinstance(parsed, dict):
            markets[normalize_name(name)].append(
                {
                    **parsed,
                    "market_athlete_id": athlete_id,
                    "market_type": type_name,
                    "market_source": source,
                    "market_updated_at": str(sides[0].get("lastUpdated") or ""),
                }
            )
            continue
        if len(sides) < 2:
            continue
        market_type = MLB_MARKET_TYPES.get(_canonical_market_name(type_name))
        if not market_type:
            continue
        stat_key, stat_label, role, grade_supported = market_type
        # ESPN's two-sided total collections consistently return over, then under.
        over_odds = _american_odds((((sides[0].get("odds") or {}).get("american") or {}).get("value")))
        under_odds = _american_odds((((sides[1].get("odds") or {}).get("american") or {}).get("value")))
        if over_odds is None or under_odds is None:
            continue
        markets[normalize_name(name)].append(
            {
                "stat_key": stat_key,
                "stat_label": stat_label,
                "market_athlete_id": athlete_id,
                "line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "market_role": role,
                "grade_supported": grade_supported,
                "market_format": "total",
                "market_type": type_name,
                "market_source": source,
                "market_updated_at": str(sides[0].get("lastUpdated") or ""),
            }
        )
    return dict(markets)


def _game_market_index(
    client: Any,
    scoreboard: dict[str, Any],
    game: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    event_market = _espn_event_market(scoreboard, game)
    if not event_market:
        return {}
    event_id, provider_id, source = event_market
    try:
        summary = client.mlb_espn_summary(event_id)
        payload = client.mlb_espn_prop_bets(event_id, provider_id)
    except Exception:
        return {}
    items = payload.get("items") or []
    athlete_names = _summary_athlete_names(summary)
    relevant_athlete_ids = {
        _athlete_ref_id(row)
        for row in items
        if _canonical_market_name(str((row.get("type") or {}).get("name") or "")) in MLB_MARKET_TYPES
    }
    for athlete_id in sorted(relevant_athlete_ids - athlete_names.keys()):
        if not athlete_id:
            continue
        try:
            name = _athlete_name(client.mlb_espn_athlete(athlete_id))
        except Exception:
            continue
        if name:
            athlete_names[athlete_id] = name
    return _market_index(items, athlete_names, source)


def _best_market_side(
    market: dict[str, Any],
    over_probability: float,
) -> tuple[str, float, int]:
    over_odds = int(market["over_odds"])
    choices = [
        ("Over", over_probability, over_odds),
    ]
    if market.get("under_odds") is not None:
        choices.append(("Under", 1.0 - over_probability, int(market["under_odds"])))
    return max(
        choices,
        key=lambda row: row[1] - safe_float(american_implied_probability(row[2])),
    )


def _candidate_sort_key(prop: dict[str, Any]) -> tuple[int, float, float, str]:
    ml_key = ev_sort_key(prop)
    return (ml_key[0], ml_key[1], ml_key[3], ml_key[4])


def _first_stat(payload: dict[str, Any]) -> dict[str, Any]:
    for group in payload.get("stats") or []:
        splits = group.get("splits") or []
        if splits:
            return splits[0].get("stat") or {}
    return {}


def _baseball_innings(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if "." not in text:
        return safe_float(text)
    whole, partial = text.split(".", 1)
    outs = safe_int(partial[:1])
    return safe_float(whole) + (min(2, max(0, outs)) / 3.0)


def _h2h(payload: dict[str, Any]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for group in payload.get("stats") or []:
        if str((group.get("type") or {}).get("displayName") or "") == "vsPlayerTotal":
            splits = group.get("splits") or []
            if splits:
                total = splits[0].get("stat") or {}
                break
    at_bats = safe_int(total.get("atBats"))
    hits = safe_int(total.get("hits"))
    return {
        "available": at_bats > 0,
        "at_bats": at_bats,
        "hits": hits,
        "average": round(hits / at_bats, 3) if at_bats else None,
    }


def _weather_factor(weather: dict[str, Any]) -> tuple[float, list[str]]:
    condition = str(weather.get("condition") or "Unknown")
    temperature = safe_float(weather.get("temp"), 72.0)
    wind = str(weather.get("wind") or "Unknown")
    lower_wind = wind.lower()
    factor = 1.0 + max(-0.025, min(0.025, (temperature - 72.0) / 600.0))
    if "out to" in lower_wind or "outward" in lower_wind:
        factor += 0.025
    elif "in from" in lower_wind or "inward" in lower_wind:
        factor -= 0.02
    return factor, [f"Weather {condition}, {temperature:.0f}F", f"Wind {wind}"]


def _game_parts(game: dict[str, Any]) -> dict[str, Any]:
    away = ((game.get("teams") or {}).get("away") or {})
    home = ((game.get("teams") or {}).get("home") or {})
    return {
        "game_pk": safe_int(game.get("gamePk")),
        "start_time": str(game.get("gameDate") or ""),
        "away_id": safe_int((away.get("team") or {}).get("id")),
        "home_id": safe_int((home.get("team") or {}).get("id")),
        "away_team": str((away.get("team") or {}).get("name") or ""),
        "home_team": str((home.get("team") or {}).get("name") or ""),
        "away_pitcher": away.get("probablePitcher") or {},
        "home_pitcher": home.get("probablePitcher") or {},
        "venue": game.get("venue") or {},
    }


def _roster_hitters(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hitters: list[dict[str, Any]] = []
    for row in payload.get("roster") or []:
        if str((row.get("position") or {}).get("abbreviation") or "") == "P":
            continue
        person = row.get("person") or {}
        stats = _first_stat(person)
        at_bats = safe_int(stats.get("atBats"))
        if not person.get("id") or at_bats < 20:
            continue
        hitters.append(
            {
                "id": safe_int(person.get("id")),
                "name": str(person.get("fullName") or ""),
                "stats": stats,
            }
        )
    return hitters


def _live_lineup(feed: dict[str, Any], side: str) -> list[dict[str, Any]]:
    team_box = ((((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {})
    players = team_box.get("players") or {}
    hitters: list[dict[str, Any]] = []
    for player_id in (team_box.get("battingOrder") or [])[:9]:
        player = players.get(f"ID{player_id}") or {}
        person = player.get("person") or {}
        stats = (player.get("seasonStats") or {}).get("batting") or {}
        if person.get("id") and safe_int(stats.get("atBats")) >= 20:
            hitters.append(
                {
                    "id": safe_int(person.get("id")),
                    "name": str(person.get("fullName") or ""),
                    "stats": stats,
                }
            )
    return hitters


def _team_strikeout_rate(hitters: list[dict[str, Any]]) -> float:
    strikeouts = sum(safe_float(player["stats"].get("strikeOuts")) for player in hitters)
    plate_appearances = sum(safe_float(player["stats"].get("plateAppearances")) for player in hitters)
    return strikeouts / plate_appearances if plate_appearances else 0.225


def _pitcher_props(
    *,
    sport: str,
    date_iso: str,
    game: dict[str, Any],
    pitcher: dict[str, Any],
    pitcher_stats: dict[str, Any],
    team: str,
    opponent: str,
    opponent_hitters: list[dict[str, Any]],
    team_id: int,
    opponent_id: int,
    pitcher_profile: dict[str, Any],
    opponent_pitch_profile: dict[str, Any],
    markets: list[dict[str, Any]],
    park_factor: float,
    weather_factor: float,
    environment_factors: list[str],
    apply_precision: bool = True,
) -> list[dict[str, Any]]:
    pitcher_id = safe_int(pitcher.get("id"))
    name = str(pitcher.get("fullName") or "")
    starts = safe_int(pitcher_stats.get("gamesStarted"))
    appearances = safe_int(pitcher_stats.get("gamesPitched") or pitcher_stats.get("gamesPlayed"))
    innings = _baseball_innings(pitcher_stats.get("inningsPitched"))
    strikeouts = safe_float(pitcher_stats.get("strikeOuts"))
    if (
        not pitcher_id
        or not name
        or starts < 2
        or appearances <= 0
        or starts / appearances < 0.25
        or innings <= 0
    ):
        return []
    innings_per_appearance = innings / appearances
    expected_innings = _clamp(innings_per_appearance, 3.0, 7.5)
    k_per_9 = strikeouts * 9.0 / innings
    workload_projection = k_per_9 * expected_innings / 9.0
    opponent_k_rate = _team_strikeout_rate(opponent_hitters)
    opponent_adjustment = max(0.82, min(1.18, opponent_k_rate / 0.225))
    environment_adjustment = max(0.94, min(1.06, 2.0 - (park_factor * weather_factor)))
    pitch_type_factor, pitch_type_reason = _pitch_type_k_signal(
        pitcher_profile,
        opponent_pitch_profile,
        f"{opponent} lineup",
    )
    arsenal_factor, arsenal_reason = _pitcher_arsenal_signal(pitcher_profile)
    base_factors = [
        f"Season workload {k_per_9:.2f} K/9 over {appearances} appearances ({starts} starts)",
        f"Expected workload {expected_innings:.2f} innings",
        f"Opponent lineup strikeout rate {opponent_k_rate:.1%}",
        pitch_type_reason,
        arsenal_reason,
        *environment_factors,
    ]
    props: list[dict[str, Any]] = []
    for market in markets:
        if market.get("grade_supported") is False:
            continue
        stat_key = str(market.get("stat_key") or "")
        line = safe_float(market.get("line"))
        if line <= 0:
            continue
        stat_label = str(market.get("stat_label") or stat_key)
        if stat_key == "strikeouts":
            if line > MAX_PITCHER_STRIKEOUT_LINE:
                continue
            projection = workload_projection * opponent_adjustment * environment_adjustment * pitch_type_factor * arsenal_factor
            sigma = max(1.35, math.sqrt(max(0.5, projection)) * 0.9)
            detail = f"{name} projects for {projection:.2f} strikeouts against a {opponent_k_rate:.1%} K lineup."
        elif stat_key == "pitcher_walks_allowed":
            walks = safe_float(pitcher_stats.get("baseOnBalls") or pitcher_stats.get("walks"))
            bb_per_9 = walks * 9.0 / innings if innings else 3.2
            projection = bb_per_9 * expected_innings / 9.0
            sigma = max(0.85, math.sqrt(max(0.4, projection)) * 0.95)
            detail = f"{name} projects for {projection:.2f} walks allowed from a {bb_per_9:.2f} BB/9 baseline."
        elif stat_key == "pitcher_outs_recorded":
            projection = expected_innings * 3.0
            sigma = max(2.2, math.sqrt(max(1.0, projection)) * 0.75)
            detail = f"{name} projects for {projection:.1f} outs from starter workload."
        elif stat_key == "pitcher_hits_allowed":
            hits_per_9 = safe_float(pitcher_stats.get("hitsPer9Inn"), 8.4)
            projection = hits_per_9 * expected_innings / 9.0 * park_factor * weather_factor
            sigma = max(1.25, math.sqrt(max(0.5, projection)) * 0.9)
            detail = f"{name} projects for {projection:.2f} hits allowed from a {hits_per_9:.2f} H/9 baseline."
        elif stat_key == "pitcher_earned_runs_allowed":
            era = safe_float(pitcher_stats.get("era"), 4.25)
            projection = era * expected_innings / 9.0 * park_factor * weather_factor
            sigma = max(1.05, math.sqrt(max(0.5, projection)) * 1.05)
            detail = f"{name} projects for {projection:.2f} earned runs allowed from a {era:.2f} ERA baseline."
        else:
            continue

        over_probability = normal_probability(projection, line, sigma, "Over")
        selection, probability, odds = _best_market_side(market, over_probability)
        factors = [
            f"Posted {line:.1f} {stat_label.lower()} line at {odds:+d}",
            *base_factors,
        ]
        pick = build_pick(
            sport=sport,
            date_iso=date_iso,
            game_id=str(game["game_pk"]),
            away_team=game["away_team"],
            home_team=game["home_team"],
            start_time=game["start_time"],
            player_id=str(pitcher_id),
            player_name=name,
            team=team,
            opponent=opponent,
            stat_key=stat_key,
            stat_label=stat_label,
            selection=selection,
            line=line,
            projection=projection,
            probability=probability,
            odds=odds,
            reason=f"{detail} The posted market is {selection.lower()} {line:.1f} {stat_label.lower()}.",
            key_factors=factors,
            extra={
                "game_id": str(game["game_pk"]),
                "player_id": str(pitcher_id),
                "team_id": str(team_id),
                "opponent_id": str(opponent_id),
                "prop_role": "pitcher",
                "pitch_type_factor": round(pitch_type_factor * arsenal_factor, 4),
                "pitch_mix": _top_pitch_mix(pitcher_profile),
                "opponent_lineup_strikeout_rate": round(opponent_k_rate, 6),
                "opponent_adjustment": round(opponent_adjustment, 4),
                "arsenal_factor": round(arsenal_factor, 4),
                "market_source": market.get("market_source"),
                "market_athlete_id": market.get("market_athlete_id"),
                "market_over_odds": market.get("over_odds"),
                "market_under_odds": market.get("under_odds"),
                "market_type": market.get("market_type"),
                "market_format": market.get("market_format"),
                "market_updated_at": market.get("market_updated_at"),
                "pricing_type": "market",
                "line_source": "posted_market",
                "odds_source": "posted_market",
                "market_priced": True,
                "actionability": "market_priced",
            },
        )
        apply_ml_to_pick(
            pick,
            baseline_probability=probability,
            baseline_projection=projection,
            market_family=market_family_for_stat(stat_key),
            apply_precision=apply_precision,
        )
        props.append(pick)
    return props


def _hitter_props(
    *,
    client: Any,
    sport: str,
    date_iso: str,
    game: dict[str, Any],
    hitter: dict[str, Any],
    pitcher: dict[str, Any],
    pitcher_stats: dict[str, Any],
    pitcher_profile: dict[str, Any],
    hitter_pitch_profile: dict[str, Any],
    markets: list[dict[str, Any]],
    team: str,
    opponent: str,
    team_id: int,
    opponent_id: int,
    is_home: bool,
    park_factor: float,
    weather_factor: float,
    environment_factors: list[str],
    apply_precision: bool = True,
) -> list[dict[str, Any]]:
    stats = hitter["stats"]
    at_bats = safe_int(stats.get("atBats"))
    hits = safe_float(stats.get("hits"))
    relevant_markets = [
        market for market in markets
        if market.get("market_role") == "batter" and market.get("grade_supported") is not False
    ]
    if at_bats < 20 or not hitter.get("id") or not hitter.get("name") or not relevant_markets:
        return []
    batting_average = hits / at_bats if at_bats else safe_float(stats.get("avg"), 0.245)
    pitcher_hits_per_9 = safe_float(pitcher_stats.get("hitsPer9Inn"), 8.4)
    pitcher_adjustment = max(0.82, min(1.20, pitcher_hits_per_9 / 8.4))
    expected_at_bats = 4.0 if is_home else 4.2
    games_played = safe_float(stats.get("gamesPlayed")) or max(1.0, safe_float(stats.get("plateAppearances")) / 4.1)
    plate_appearances = safe_float(stats.get("plateAppearances")) or max(float(at_bats), games_played * 4.1)
    expected_plate_appearances = _clamp(plate_appearances / max(1.0, games_played), 3.2, 5.1)
    h2h = {"available": False, "at_bats": 0, "hits": 0, "average": None}
    try:
        h2h = _h2h(client.mlb_h2h(safe_int(hitter["id"]), safe_int(pitcher.get("id"))))
    except Exception:
        pass
    h2h_adjustment = 1.0
    if h2h["available"] and h2h["at_bats"] >= 3:
        h2h_adjustment = max(
            0.85,
            min(1.15, 1.0 + ((safe_float(h2h["average"], batting_average) - batting_average) * 0.25)),
        )
    pitch_type_factor, pitch_type_reason = _pitch_type_hit_signal(pitcher_profile, hitter_pitch_profile)
    per_at_bat = max(
        0.08,
        min(
            0.45,
            batting_average
            * pitcher_adjustment
            * park_factor
            * weather_factor
            * h2h_adjustment
            * pitch_type_factor,
        ),
    )
    h2h_factor = (
        f"H2H available: {h2h['hits']}-for-{h2h['at_bats']} ({safe_float(h2h['average']):.3f})"
        if h2h["available"]
        else "H2H unavailable from MLB StatsAPI"
    )
    factors = [
        f"Season batting average {batting_average:.3f} over {at_bats} at-bats",
        f"Opposing pitcher allows {pitcher_hits_per_9:.2f} hits/9",
        pitch_type_reason,
        h2h_factor,
        *environment_factors,
    ]
    props: list[dict[str, Any]] = []
    run_creation_adjustment = _clamp(
        pitcher_adjustment * park_factor * weather_factor,
        0.82,
        1.20,
    )
    doubles = safe_float(stats.get("doubles"))
    triples = safe_float(stats.get("triples"))
    home_runs = safe_float(stats.get("homeRuns"))
    singles = max(0.0, hits - doubles - triples - home_runs)
    total_bases = safe_float(stats.get("totalBases")) or (singles + (2 * doubles) + (3 * triples) + (4 * home_runs))
    stat_per_game = {
        "hits_runs_rbis": hits + safe_float(stats.get("runs")) + safe_float(stats.get("rbi") or stats.get("rbis")),
        "runs": safe_float(stats.get("runs")),
        "rbis": safe_float(stats.get("rbi") or stats.get("rbis")),
        "total_bases": total_bases,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "home_runs": home_runs,
        "stolen_bases": safe_float(stats.get("stolenBases")),
    }
    pitcher_k_per_9 = safe_float(pitcher_stats.get("strikeoutsPer9Inn"), 8.4)
    pitcher_walks_per_9 = safe_float(pitcher_stats.get("walksPer9Inn") or pitcher_stats.get("baseOnBallsPer9"), 3.1)
    for market in relevant_markets:
        stat_key = str(market.get("stat_key") or "")
        line = safe_float(market.get("line"))
        if line <= 0:
            continue
        if stat_key == "hits":
            projection = per_at_bat * expected_at_bats
            over_probability = _binomial_over_probability(per_at_bat, expected_at_bats, line)
            market_factors = [
                f"Posted {line:.1f} hits line at {int(market['over_odds']):+d} over",
                *factors,
            ]
            reason = (
                f"{hitter['name']} projects for {projection:.2f} hits versus the posted {line:.1f} line "
                "after pitcher, pitch-type, park, weather, wind, and H2H adjustments."
            )
            prop_role = "batter"
        elif stat_key == "batter_walks":
            walks = safe_float(stats.get("baseOnBalls") or stats.get("walks"))
            walk_rate = walks / plate_appearances if plate_appearances else 0.08
            pitcher_walk_factor = _clamp(pitcher_walks_per_9 / 3.1, 0.82, 1.20)
            per_pa = _clamp(walk_rate * pitcher_walk_factor, 0.025, 0.24)
            projection = per_pa * expected_plate_appearances
            over_probability = _binomial_over_probability(per_pa, expected_plate_appearances, line)
            market_factors = [
                f"Season walk rate {walk_rate:.1%} over {int(plate_appearances)} PA",
                f"Posted {line:.1f} walks line at {int(market['over_odds']):+d} over",
                f"Opposing pitcher walks profile {pitcher_walks_per_9:.2f} BB/9",
                *environment_factors,
            ]
            reason = (
                f"{hitter['name']} projects for {projection:.2f} walks versus the posted {line:.1f} line "
                "after plate-appearance and opposing-pitcher walk-rate adjustments."
            )
            prop_role = "batter_walks"
        elif stat_key == "batter_strikeouts":
            strikeout_rate = safe_float(stats.get("strikeOuts")) / plate_appearances if plate_appearances else 0.22
            pitcher_k_factor = _clamp(pitcher_k_per_9 / 8.4, 0.82, 1.22)
            per_pa = _clamp(strikeout_rate * pitcher_k_factor, 0.06, 0.45)
            projection = per_pa * expected_plate_appearances
            over_probability = _binomial_over_probability(per_pa, expected_plate_appearances, line)
            market_factors = [
                f"Season strikeout rate {strikeout_rate:.1%} over {int(plate_appearances)} PA",
                f"Posted {line:.1f} batter strikeouts line at {int(market['over_odds']):+d} over",
                f"Opposing pitcher strikeout profile {pitcher_k_per_9:.2f} K/9",
                pitch_type_reason,
                *environment_factors,
            ]
            reason = (
                f"{hitter['name']} projects for {projection:.2f} strikeouts versus the posted {line:.1f} line "
                "after plate-appearance, pitcher, and pitch-type adjustments."
            )
            prop_role = "batter_strikeouts"
        else:
            raw_total = stat_per_game.get(stat_key)
            if games_played < 10 or raw_total is None:
                continue
            per_game = raw_total / games_played
            if per_game <= 0:
                continue
            projection = per_game * run_creation_adjustment
            over_probability = normal_probability(
                projection,
                line,
                max(1.20, math.sqrt(max(0.5, projection)) * 0.95),
                "Over",
            )
            stat_label = str(market.get("stat_label") or stat_key)
            market_factors = [
                f"Season {stat_label.lower()} per game {per_game:.2f} over {int(games_played)} games",
                f"Posted {line:.1f} {stat_label.lower()} line at {int(market['over_odds']):+d} over",
                f"Opposing pitcher allows {pitcher_hits_per_9:.2f} hits/9",
                *environment_factors,
            ]
            reason = (
                f"{hitter['name']} projects for {projection:.2f} {stat_label.lower()} versus the posted "
                f"{line:.1f} line after pitcher, park, and run-environment adjustments."
            )
            prop_role = "batter_hrr" if stat_key == "hits_runs_rbis" else "batter"

        selection, probability, odds = _best_market_side(market, over_probability)
        pick = build_pick(
            sport=sport,
            date_iso=date_iso,
            game_id=str(game["game_pk"]),
            away_team=game["away_team"],
            home_team=game["home_team"],
            start_time=game["start_time"],
            player_id=str(hitter["id"]),
            player_name=hitter["name"],
            team=team,
            opponent=opponent,
            stat_key=stat_key,
            stat_label=str(market.get("stat_label") or stat_key),
            selection=selection,
            line=line,
            projection=projection,
            probability=probability,
            odds=odds,
            reason=reason,
            key_factors=market_factors,
            extra={
                "game_id": str(game["game_pk"]),
                "player_id": str(hitter["id"]),
                "team_id": str(team_id),
                "opponent_id": str(opponent_id),
                "prop_role": prop_role,
                "h2h": h2h,
                "h2h_adjustment": round(h2h_adjustment, 4),
                "pitcher_adjustment": round(pitcher_adjustment, 4),
                "park_factor": round(park_factor, 4),
                "weather_factor": round(weather_factor, 4),
                "pitch_type_factor": round(pitch_type_factor, 4),
                "pitch_mix": _top_pitch_mix(pitcher_profile),
                "market_source": market.get("market_source"),
                "market_athlete_id": market.get("market_athlete_id"),
                "market_over_odds": market.get("over_odds"),
                "market_under_odds": market.get("under_odds"),
                "market_type": market.get("market_type"),
                "market_format": market.get("market_format"),
                "market_updated_at": market.get("market_updated_at"),
                "pricing_type": "market",
                "line_source": "posted_market",
                "odds_source": "posted_market",
                "market_priced": True,
                "actionability": "market_priced",
            },
        )
        apply_ml_to_pick(
            pick,
            baseline_probability=probability,
            baseline_projection=projection,
            market_family=market_family_for_stat(stat_key),
            apply_precision=apply_precision,
        )
        props.append(pick)
    return props


def _game_props(
    client: Any,
    date_iso: str,
    raw_game: dict[str, Any],
    season: int,
    market_scoreboard: dict[str, Any],
    diagnostics: dict[str, int] | None = None,
    select: bool = True,
    apply_precision: bool = True,
) -> list[dict[str, Any]]:
    game = _game_parts(raw_game)
    market_index = _game_market_index(client, market_scoreboard, game)
    feed = client.mlb_live_feed(game["game_pk"])
    game_data = feed.get("gameData") or {}
    venue = game_data.get("venue") or game["venue"]
    venue_id = safe_int(venue.get("id"))
    venue_name = str(venue.get("name") or game["venue"].get("name") or "Unknown venue")
    park_factor = PARK_FACTORS.get(venue_id, 1.0)
    weather_factor, weather_factors = _weather_factor(game_data.get("weather") or {})
    environment_factors = [f"Venue {venue_name}, park factor {park_factor:.2f}", *weather_factors]
    away_abbr = _team_abbreviation(feed, "away", game["away_team"])
    home_abbr = _team_abbreviation(feed, "home", game["home_team"])

    hitters: dict[str, list[dict[str, Any]]] = {}
    for side, team_id in (("away", game["away_id"]), ("home", game["home_id"])):
        lineup = _live_lineup(feed, side)
        if not lineup:
            lineup = _roster_hitters(client.mlb_roster(team_id, date_iso, season))
        hitters[side] = lineup

    pitcher_stats: dict[str, dict[str, Any]] = {}
    pitcher_profiles: dict[str, dict[str, Any]] = {}
    for side in ("away", "home"):
        pitcher = game[f"{side}_pitcher"]
        pitcher_stats[side] = _first_stat(
            client.mlb_player_stats(safe_int(pitcher.get("id")), "pitching", season)
        ) if pitcher.get("id") else {}
        pitcher_profiles[side] = _profile_from_player_statcast(
            client,
            safe_int(pitcher.get("id")),
            "pitcher",
            date_iso,
            days=45,
        )

    team_pitch_rows = {
        "away": _team_statcast_rows(client, away_abbr, date_iso, days=30),
        "home": _team_statcast_rows(client, home_abbr, date_iso, days=30),
    }
    team_pitch_profiles = {
        side: _summarize_pitch_rows(rows)
        for side, rows in team_pitch_rows.items()
    }

    candidates: list[dict[str, Any]] = []
    away_pitcher_markets = [
        market for market in market_index.get(normalize_name(str(game["away_pitcher"].get("fullName") or "")), [])
        if market.get("market_role") == "pitcher"
    ]
    home_pitcher_markets = [
        market for market in market_index.get(normalize_name(str(game["home_pitcher"].get("fullName") or "")), [])
        if market.get("market_role") == "pitcher"
    ]
    candidates.extend(
        _pitcher_props(
            sport="MLB",
            date_iso=date_iso,
            game=game,
            pitcher=game["away_pitcher"],
            pitcher_stats=pitcher_stats["away"],
            team=game["away_team"],
            opponent=game["home_team"],
            opponent_hitters=hitters["home"],
            team_id=game["away_id"],
            opponent_id=game["home_id"],
            pitcher_profile=pitcher_profiles["away"],
            opponent_pitch_profile=team_pitch_profiles["home"],
            markets=away_pitcher_markets,
            park_factor=park_factor,
            weather_factor=weather_factor,
            environment_factors=environment_factors,
            apply_precision=apply_precision,
        )
    )
    candidates.extend(
        _pitcher_props(
            sport="MLB",
            date_iso=date_iso,
            game=game,
            pitcher=game["home_pitcher"],
            pitcher_stats=pitcher_stats["home"],
            team=game["home_team"],
            opponent=game["away_team"],
            opponent_hitters=hitters["away"],
            team_id=game["home_id"],
            opponent_id=game["away_id"],
            pitcher_profile=pitcher_profiles["home"],
            opponent_pitch_profile=team_pitch_profiles["away"],
            markets=home_pitcher_markets,
            park_factor=park_factor,
            weather_factor=weather_factor,
            environment_factors=environment_factors,
            apply_precision=apply_precision,
        )
    )

    hitter_profile_cache: dict[int, dict[str, Any]] = {}
    for side, opposing_side in (("away", "home"), ("home", "away")):
        ordered = sorted(
            hitters[side],
            key=lambda player: (
                safe_float(player["stats"].get("avg")),
                safe_float(player["stats"].get("ops")),
                safe_int(player["stats"].get("atBats")),
            ),
            reverse=True,
        )
        ordered_with_markets = [
            hitter for hitter in ordered
            if market_index.get(normalize_name(str(hitter.get("name") or "")))
        ]
        for hitter in ordered_with_markets:
            hitter_id = safe_int(hitter.get("id"))
            if hitter_id not in hitter_profile_cache:
                hitter_profile = _profile_for_batter_from_team_rows(team_pitch_rows[side], hitter_id)
                if safe_int(hitter_profile.get("sample_pitches")) < 20:
                    hitter_profile = _profile_from_player_statcast(client, hitter_id, "batter", date_iso, days=45)
                hitter_profile_cache[hitter_id] = hitter_profile
            hitter_profile = hitter_profile_cache[hitter_id]
            hitter_props = _hitter_props(
                client=client,
                sport="MLB",
                date_iso=date_iso,
                game=game,
                hitter=hitter,
                pitcher=game[f"{opposing_side}_pitcher"],
                pitcher_stats=pitcher_stats[opposing_side],
                pitcher_profile=pitcher_profiles[opposing_side],
                hitter_pitch_profile=hitter_profile,
                markets=market_index.get(normalize_name(str(hitter.get("name") or "")), []),
                team=game[f"{side}_team"],
                opponent=game[f"{opposing_side}_team"],
                team_id=game[f"{side}_id"],
                opponent_id=game[f"{opposing_side}_id"],
                is_home=side == "home",
                park_factor=park_factor,
                weather_factor=weather_factor,
                environment_factors=environment_factors,
                apply_precision=apply_precision,
            )
            candidates.extend(hitter_props)

    if diagnostics is not None:
        diagnostics["market_candidates"] = diagnostics.get("market_candidates", 0) + sum(
            candidate.get("market_priced") is True for candidate in candidates
        )
    if not select:
        return candidates
    return select_top_props(candidates)


def generate_mlb_model(client: Any, date_iso: str) -> dict[str, Any]:
    """Generate up to four validated, market-priced MLB props per game."""
    try:
        schedule = client.mlb_schedule(date_iso)
    except Exception as exc:
        return {"ok": False, "sport": "MLB", "date": date_iso, "games": 0, "picks": [], "errors": [str(exc)]}
    games = [
        game
        for date_group in schedule.get("dates") or []
        for game in date_group.get("games") or []
    ]
    if not games:
        return {
            "ok": True,
            "sport": "MLB",
            "date": date_iso,
            "games": 0,
            "picks": [],
            "errors": [],
            "note": "No MLB games scheduled; empty slate is healthy.",
        }

    picks: list[dict[str, Any]] = []
    errors: list[str] = []
    diagnostics: dict[str, int] = {"market_candidates": 0}
    season = int(date_iso[:4])
    try:
        market_scoreboard = client.mlb_espn_scoreboard(date_iso)
    except Exception as exc:
        market_scoreboard = {}
        errors.append(f"ESPN MLB market scoreboard failed: {exc}")
    for game in games:
        try:
            picks.extend(_game_props(client, date_iso, game, season, market_scoreboard, diagnostics))
        except Exception as exc:
            errors.append(f"{game.get('gamePk')}: {exc}")
    if not picks:
        from .precision import precision_model_required

        if precision_model_required() and diagnostics["market_candidates"] > 0:
            return {
                "ok": True,
                "sport": "MLB",
                "date": date_iso,
                "games": len(games),
                "picks": [],
                "errors": errors,
                "abstained": True,
                "note": "No posted MLB prop cleared the active 70% season precision gate.",
                "method": "Season-trained precision model with chronological validation and abstention",
            }
        error = (
            f"No MLB player props generated for {len(games)} scheduled game(s); "
            "posted sportsbook markets were unavailable or could not be parsed."
        )
        errors.append(error)
        return {
            "ok": False,
            "sport": "MLB",
            "date": date_iso,
            "games": len(games),
            "picks": [],
            "error": error,
            "errors": errors,
        }
    return {
        "ok": True,
        "sport": "MLB",
        "date": date_iso,
        "games": len(games),
        "picks": picks,
        "errors": errors,
        "method": "Season-trained precision model over MLB StatsAPI, Statcast, and posted DraftKings markets via ESPN",
    }


def generate_mlb_candidate_model(client: Any, date_iso: str) -> dict[str, Any]:
    """Generate the full market-priced MLB candidate pool for model variants."""
    try:
        schedule = client.mlb_schedule(date_iso)
    except Exception as exc:
        return {"ok": False, "sport": "MLB", "date": date_iso, "games": 0, "picks": [], "errors": [str(exc)]}
    games = [
        game
        for date_group in schedule.get("dates") or []
        for game in date_group.get("games") or []
    ]
    if not games:
        return {
            "ok": True,
            "sport": "MLB",
            "date": date_iso,
            "games": 0,
            "picks": [],
            "errors": [],
            "note": "No MLB games scheduled; empty slate is healthy.",
        }

    picks: list[dict[str, Any]] = []
    errors: list[str] = []
    diagnostics: dict[str, int] = {"market_candidates": 0}
    season = int(date_iso[:4])
    try:
        market_scoreboard = client.mlb_espn_scoreboard(date_iso)
    except Exception as exc:
        market_scoreboard = {}
        errors.append(f"ESPN MLB market scoreboard failed: {exc}")
    for game in games:
        try:
            picks.extend(
                _game_props(
                    client,
                    date_iso,
                    game,
                    season,
                    market_scoreboard,
                    diagnostics,
                    select=False,
                    apply_precision=False,
                )
            )
        except Exception as exc:
            errors.append(f"{game.get('gamePk')}: {exc}")
    if not picks and diagnostics["market_candidates"] == 0 and games:
        errors.append(
            f"No MLB player-prop candidates generated for {len(games)} scheduled game(s); "
            "posted sportsbook markets were unavailable or could not be parsed."
        )
    return {
        "ok": True,
        "sport": "MLB",
        "date": date_iso,
        "games": len(games),
        "picks": picks,
        "errors": errors,
        "method": "MLB candidate pool with season, all-time, hot L10, and H2H matchup inputs",
    }

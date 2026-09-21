"""Daily NHL publisher.

Every pregame game gets one research moneyline. Spread and total rows are
published only when the caller supplies an observed line. Odds stay empty
unless that same observed quote includes an American price. Decisions are
PASS at 0 units: the goal model has no validated staking segment.
"""
from __future__ import annotations

import math
from typing import Any

try:
    from nhl_core import canonical_abbrev, load_ratings, over_probability, project_game
    from nhl_data import is_pregame, load_slate
except ImportError:
    from .nhl_core import canonical_abbrev, load_ratings, over_probability, project_game
    from .nhl_data import is_pregame, load_slate


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _american(value: Any) -> int | None:
    number = _num(value)
    if number is None or -100.0 < number < 100.0:
        return None
    return int(round(number))


def _price_fields(odds: int | None) -> dict[str, Any]:
    if odds is None:
        return {"odds": None, "pricing_type": "unpriced", "odds_source": None, "market_priced": False}
    return {"odds": odds, "pricing_type": "market", "odds_source": "observed_quote", "market_priced": True}


def _round_features(projection: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in projection.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return rounded


def _reason(game: dict[str, Any], evidence_ok: bool) -> str:
    if not evidence_ok:
        return "research_only:team_ratings_unavailable"
    if str(game.get("season_type") or "") == "PRE":
        return "research_only:preseason"
    return "research_only:no_validated_segment"


def generate_nhl_picks(
    date_iso: str,
    *,
    games: list[dict[str, Any]] | None = None,
    ratings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = ratings if ratings is not None else load_ratings()
    if not loaded:
        return {
            "ok": False,
            "date": date_iso,
            "model": "NHLModel",
            "picks": [],
            "games": [],
            "error": "NHL ratings artifact is missing or unreadable; forecasts could not run.",
        }
    if games is None:
        try:
            slate, slate_source = load_slate(date_iso)
        except Exception as exc:
            return {
                "ok": False,
                "date": date_iso,
                "model": "NHLModel",
                "picks": [],
                "games": [],
                "error": str(exc),
            }
    else:
        slate, slate_source = games, "injected"
    model_version = str(loaded.get("model_version") or "nhl_poisson_v1")
    picks: list[dict[str, Any]] = []
    games_out: list[dict[str, Any]] = []
    started = 0
    for game in slate:
        if not isinstance(game, dict):
            continue
        if not is_pregame(game):
            started += 1
            continue
        home = str(game.get("home_team") or "")
        away = str(game.get("away_team") or "")
        if not home or not away:
            continue
        projection = project_game(
            loaded,
            str(game.get("home_abbrev") or ""),
            str(game.get("away_abbrev") or ""),
        )
        home_probability = float(projection["moneyline_home_probability"])
        pick_home = home_probability >= 0.5
        team = home if pick_home else away
        side_probability = home_probability if pick_home else 1.0 - home_probability
        matchup = f"{away} @ {home}"
        start = str(game.get("start_time") or "")
        reason = _reason(game, bool(projection["evidence_ok"]))
        features = _round_features(projection)
        base = {
            "source": "NHL Model",
            "sport": "NHL",
            "league": "NHL",
            "date": date_iso,
            "game_id": str(game.get("game_id") or ""),
            "game": matchup,
            "matchup": matchup,
            "home_team": home,
            "away_team": away,
            "home_abbrev": canonical_abbrev(game.get("home_abbrev")),
            "away_abbrev": canonical_abbrev(game.get("away_abbrev")),
            "start_time": start,
            "game_start_time": start,
            "season_type": str(game.get("season_type") or ""),
            "model_version": model_version,
            "shadow_mode": False,
            "actionability": "research",
            "calibration_excluded": True,
            "grade_supported": True,
            "decision": "PASS",
            "source_decision": "PASS",
            "decision_reason": reason,
            "units": 0,
            "features": features,
        }
        side_odds = _american(game.get("home_moneyline") if pick_home else game.get("away_moneyline"))
        opposite_odds = _american(game.get("away_moneyline") if pick_home else game.get("home_moneyline"))
        picks.append({
            **base,
            "pick": f"{team} ML ({matchup})",
            "market": "h2h",
            "market_type": "h2h",
            "team": team,
            "selection": team,
            "side": "home" if pick_home else "away",
            **_price_fields(side_odds),
            "opposite_odds": opposite_odds,
            "probability": round(side_probability, 4),
            "raw_probability": round(side_probability, 4),
            "calibrated_probability": round(side_probability, 4),
            "model_home_win_probability": round(home_probability, 4),
        })
        spread_line = _num(game.get("spread_line"))
        # Cover math is the chance of winning by 2+, which is the -1.5 puck line.
        # Any other observed number is left off the card instead of relabeled.
        if spread_line is not None and abs(spread_line + 1.5) < 1e-9:
            cover = float(projection["puckline_home_cover_probability"])
            pick_home_spread = cover >= 0.5
            spread_team = home if pick_home_spread else away
            team_line = -1.5 if pick_home_spread else 1.5
            spread_probability = cover if pick_home_spread else 1.0 - cover
            spread_odds = _american(game.get("home_spread_odds") if pick_home_spread else game.get("away_spread_odds"))
            picks.append({
                **base,
                "pick": f"{spread_team} {team_line:+g} ({matchup})",
                "market": "spread",
                "market_type": "spread",
                "team": spread_team,
                "selection": spread_team,
                "side": "home" if pick_home_spread else "away",
                "line": team_line,
                "market_line": -1.5,
                **_price_fields(spread_odds),
                "probability": round(spread_probability, 4),
                "raw_probability": round(spread_probability, 4),
                "calibrated_probability": round(spread_probability, 4),
            })
        total_line = _num(game.get("total_line"))
        if total_line is not None:
            total_probability = over_probability(
                float(projection["lambda_home"]),
                float(projection["lambda_away"]),
                total_line,
            )
            over = total_probability >= 0.5
            direction = "over" if over else "under"
            published_probability = total_probability if over else 1.0 - total_probability
            total_odds = _american(game.get("over_odds") if over else game.get("under_odds"))
            picks.append({
                **base,
                "pick": f"{direction.title()} {total_line:g} ({matchup})",
                "market": "totals",
                "market_type": "totals",
                "direction": direction,
                "selection": direction.title(),
                "line": total_line,
                "market_line": total_line,
                **_price_fields(total_odds),
                "probability": round(published_probability, 4),
                "raw_probability": round(published_probability, 4),
                "calibrated_probability": round(published_probability, 4),
                "model_expected_total": round(float(projection["expected_total_with_ot_goal"]), 3),
            })
        games_out.append({
            "game_id": base["game_id"],
            "matchup": matchup,
            "start_time": start,
            "season_type": base["season_type"],
            "home_win_probability": round(home_probability, 4),
            "expected_total": round(float(projection["expected_total_with_ot_goal"]), 3),
            "ratings_found": bool(projection["evidence_ok"]),
        })
    season_types = sorted({str(game.get("season_type") or "") for game in games_out if game.get("season_type")})
    if not slate:
        note = f"NHL active slate: 0 game(s), 0 row(s). No NHL games scheduled for {date_iso}."
    else:
        note = (
            f"NHL active slate: {len(games_out)} pregame game(s), {len(picks)} research row(s). "
            "No validated staking segment; rows are PASS at 0 units."
        )
        if started:
            note += f" {started} started game(s) skipped."
        if "PRE" in season_types:
            note += " Preseason games use the previous regular season's goal rates and are not staked."
        if any(not game["ratings_found"] for game in games_out):
            note += " One or more teams had no prior-season rating; those rows use league-average rates."
    return {
        "ok": True,
        "date": date_iso,
        "model": "NHLModel",
        "model_version": model_version,
        "shadow_mode": False,
        "actionability": "research",
        "slate_source": slate_source,
        "prior_season": loaded.get("prior_season"),
        "standings_date": loaded.get("standings_date"),
        "picks": picks,
        "games": games_out,
        "note": note,
        "coverage": {
            "official_games": len(slate),
            "pregame_games": len(games_out),
            "started_games": started,
            "staked_rows": 0,
        },
    }

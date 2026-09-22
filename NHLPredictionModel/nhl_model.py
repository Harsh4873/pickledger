"""Daily NHL publisher.

Moneyline, puck line, game total, team total, and player props are published
only when that market has an observed American price. A missing price is
skipped. Player props also need an observed scoring mean; this model does not
invent one. Decisions stay PASS at 0 units: the goal model has no validated
staking segment.
"""
from __future__ import annotations

import math
from typing import Any

try:
    from nhl_core import (
        canonical_abbrev,
        counting_over_probability,
        load_ratings,
        over_probability,
        project_game,
        team_total_over_probability,
    )
    from nhl_data import is_pregame, load_slate
    from nhl_markets import fetch_pregame_quotes
except ImportError:
    from .nhl_core import (
        canonical_abbrev,
        counting_over_probability,
        load_ratings,
        over_probability,
        project_game,
        team_total_over_probability,
    )
    from .nhl_data import is_pregame, load_slate
    from .nhl_markets import fetch_pregame_quotes


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _american(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or -100.0 < number < 100.0:
        return None
    return int(round(number))


def _price_fields(odds: int | None, odds_source: str | None) -> dict[str, Any]:
    if odds is None:
        return {"odds": None, "pricing_type": "unpriced", "odds_source": None, "market_priced": False}
    return {
        "odds": odds,
        "pricing_type": "market",
        "odds_source": odds_source or "observed_quote",
        "market_priced": True,
    }


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


def _row_shell(
    *,
    base: dict[str, Any],
    pick: str,
    market: str,
    probability: float,
    odds: int | None,
    odds_source: str | None,
    reason: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base,
        "pick": pick,
        "market": market,
        "market_type": market,
        **extra,
        **_price_fields(odds, odds_source),
        "probability": round(probability, 4),
        "raw_probability": round(probability, 4),
        "calibrated_probability": round(probability, 4),
        "decision": "PASS",
        "source_decision": "PASS",
        "decision_reason": reason,
        "units": 0,
    }


def generate_nhl_picks(
    date_iso: str,
    *,
    games: list[dict[str, Any]] | None = None,
    ratings: dict[str, Any] | None = None,
    fetch_json: Any = None,
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
        try:
            quote_summary = fetch_pregame_quotes(slate, fetch_json=fetch_json)
        except Exception as exc:
            quote_summary = {"feeds_read": 0, "error": str(exc), "matched_games": 0}
    else:
        slate, slate_source = games, "injected"
        quote_summary = {"feeds_read": 0, "matched_games": 0, "injected": True}
    model_version = str(loaded.get("model_version") or "nhl_poisson_v1")
    picks: list[dict[str, Any]] = []
    games_out: list[dict[str, Any]] = []
    started = 0
    skipped: dict[str, int] = {
        "h2h": 0,
        "spread": 0,
        "totals": 0,
        "team_total": 0,
        "player_props": 0,
    }
    player_prior_missing = 0
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
        matchup = f"{away} @ {home}"
        start = str(game.get("start_time") or "")
        reason = _reason(game, bool(projection["evidence_ok"]))
        odds_source = str(game.get("odds_source") or "") or None
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
            "units": 0,
            "features": _round_features(projection),
        }
        published_markets: list[str] = []
        pick_home = home_probability >= 0.5
        team = home if pick_home else away
        side_probability = home_probability if pick_home else 1.0 - home_probability
        side_odds = _american(game.get("home_moneyline") if pick_home else game.get("away_moneyline"))
        opposite_odds = _american(game.get("away_moneyline") if pick_home else game.get("home_moneyline"))
        if side_odds is None:
            skipped["h2h"] += 1
        else:
            picks.append(_row_shell(
                base=base,
                pick=f"{team} ML ({matchup})",
                market="h2h",
                probability=side_probability,
                odds=side_odds,
                odds_source=odds_source,
                reason=reason,
                extra={
                    "team": team,
                    "selection": team,
                    "side": "home" if pick_home else "away",
                    "opposite_odds": opposite_odds,
                    "model_home_win_probability": round(home_probability, 4),
                },
            ))
            published_markets.append("h2h")

        home_spread = _num(game.get("spread_line"))
        if home_spread is None:
            skipped["spread"] += 1
        elif abs(abs(home_spread) - 1.5) > 1e-9:
            skipped["spread"] += 1
        else:
            home_is_minus = home_spread < 0
            if home_is_minus:
                cover = float(projection["puckline_home_cover_probability"])
                pick_home_spread = cover >= 0.5
            else:
                cover = float(projection["puckline_away_cover_probability"])
                pick_home_spread = cover < 0.5
            spread_team = home if pick_home_spread else away
            team_line = -1.5 if pick_home_spread else 1.5
            if home_is_minus:
                spread_probability = cover if pick_home_spread else 1.0 - cover
            else:
                spread_probability = (1.0 - cover) if pick_home_spread else cover
            spread_odds = _american(game.get("home_spread_odds") if pick_home_spread else game.get("away_spread_odds"))
            if spread_odds is None:
                skipped["spread"] += 1
            else:
                picks.append(_row_shell(
                    base=base,
                    pick=f"{spread_team} {team_line:+g} ({matchup})",
                    market="spread",
                    probability=spread_probability,
                    odds=spread_odds,
                    odds_source=odds_source,
                    reason=reason,
                    extra={
                        "team": spread_team,
                        "selection": spread_team,
                        "side": "home" if pick_home_spread else "away",
                        "line": team_line,
                        "market_line": -1.5 if home_is_minus else 1.5,
                    },
                ))
                published_markets.append("spread")

        total_line = _num(game.get("total_line"))
        if total_line is None:
            skipped["totals"] += 1
        else:
            total_probability = over_probability(
                float(projection["lambda_home"]),
                float(projection["lambda_away"]),
                total_line,
            )
            over = total_probability >= 0.5
            direction = "over" if over else "under"
            published_probability = total_probability if over else 1.0 - total_probability
            total_odds = _american(game.get("over_odds") if over else game.get("under_odds"))
            if total_odds is None:
                skipped["totals"] += 1
            else:
                picks.append(_row_shell(
                    base=base,
                    pick=f"{direction.title()} {total_line:g} ({matchup})",
                    market="totals",
                    probability=published_probability,
                    odds=total_odds,
                    odds_source=odds_source,
                    reason=reason,
                    extra={
                        "direction": direction,
                        "selection": direction.title(),
                        "line": total_line,
                        "market_line": total_line,
                        "model_expected_total": round(float(projection["expected_total_with_ot_goal"]), 3),
                    },
                ))
                published_markets.append("totals")

        team_totals = game.get("team_totals") if isinstance(game.get("team_totals"), dict) else {}
        if not team_totals:
            skipped["team_total"] += 1
        else:
            published_team_total = False
            ot_rate = projection.get("ot_home_win_rate")
            for side, row in team_totals.items():
                if side not in {"home", "away"} or not isinstance(row, dict):
                    continue
                line = _num(row.get("line"))
                if line is None:
                    continue
                if side == "home":
                    over_prob = team_total_over_probability(
                        float(projection["lambda_home"]),
                        float(projection["lambda_away"]),
                        line,
                        ot_win_rate=float(ot_rate) if isinstance(ot_rate, (int, float)) else None,
                    )
                    team_name = str(row.get("team") or home)
                else:
                    away_ot = None if not isinstance(ot_rate, (int, float)) else 1.0 - float(ot_rate)
                    over_prob = team_total_over_probability(
                        float(projection["lambda_away"]),
                        float(projection["lambda_home"]),
                        line,
                        ot_win_rate=away_ot,
                    )
                    team_name = str(row.get("team") or away)
                take_over = over_prob >= 0.5
                direction = "over" if take_over else "under"
                probability = over_prob if take_over else 1.0 - over_prob
                price = _american(row.get("over_odds") if take_over else row.get("under_odds"))
                if price is None:
                    continue
                picks.append(_row_shell(
                    base=base,
                    pick=f"{team_name} team total {direction} {line:g} ({matchup})",
                    market="team_total",
                    probability=probability,
                    odds=price,
                    odds_source=odds_source,
                    reason=reason,
                    extra={
                        "team": team_name,
                        "selection": f"{team_name} {direction.title()}",
                        "side": side,
                        "direction": direction,
                        "line": line,
                        "market_line": line,
                    },
                ))
                published_team_total = True
            if published_team_total:
                published_markets.append("team_total")
            else:
                skipped["team_total"] += 1

        raw_props = game.get("player_props") if isinstance(game.get("player_props"), list) else []
        if not raw_props:
            skipped["player_props"] += 1
        else:
            published_prop = False
            for prop in raw_props:
                if not isinstance(prop, dict):
                    continue
                line = _num(prop.get("line"))
                mean = _num(prop.get("mean"))
                player = str(prop.get("player") or "").strip()
                if line is None or not player:
                    continue
                if mean is None:
                    player_prior_missing += 1
                    continue
                over_prob = counting_over_probability(mean, line)
                take_over = over_prob >= 0.5
                direction = "over" if take_over else "under"
                probability = over_prob if take_over else 1.0 - over_prob
                price = _american(prop.get("over_odds") if take_over else prop.get("under_odds"))
                if price is None:
                    continue
                stat_label = str(prop.get("stat_label") or prop.get("stat") or "prop")
                picks.append(_row_shell(
                    base=base,
                    pick=f"{player} {stat_label} {direction} {line:g} ({matchup})",
                    market="player_props",
                    probability=probability,
                    odds=price,
                    odds_source=odds_source,
                    reason=reason,
                    extra={
                        "player": player,
                        "player_name": player,
                        "team": str(prop.get("team") or ""),
                        "selection": f"{player} {direction.title()}",
                        "stat": str(prop.get("stat") or stat_label),
                        "stat_label": stat_label,
                        "direction": direction,
                        "line": line,
                        "market_line": line,
                        "projection": round(mean, 3),
                    },
                ))
                published_prop = True
            if published_prop:
                published_markets.append("player_props")
            else:
                skipped["player_props"] += 1

        games_out.append({
            "game_id": base["game_id"],
            "matchup": matchup,
            "start_time": start,
            "season_type": base["season_type"],
            "home_win_probability": round(home_probability, 4),
            "expected_total": round(float(projection["expected_total_with_ot_goal"]), 3),
            "ratings_found": bool(projection["evidence_ok"]),
            "published_markets": published_markets,
            "posted_market_names": list(game.get("posted_market_names") or []),
        })
    season_types = sorted({str(game.get("season_type") or "") for game in games_out if game.get("season_type")})
    if not slate:
        note = f"NHL active slate: 0 game(s), 0 row(s). No NHL games scheduled for {date_iso}."
    else:
        note = (
            f"NHL active slate: {len(games_out)} pregame game(s), {len(picks)} priced row(s). "
            "No validated staking segment; priced rows are PASS at 0 units."
        )
        if started:
            note += f" {started} started game(s) skipped."
        missing = [name for name, count in skipped.items() if count]
        if missing:
            note += " Markets with no observed price were skipped: " + ", ".join(missing) + "."
        if player_prior_missing:
            note += (
                f" {player_prior_missing} posted player prop(s) had a price but no observed player mean, so no side was published."
            )
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
        "quote_summary": quote_summary,
        "markets_skipped": skipped,
        "player_props_without_prior": player_prior_missing,
        "coverage": {
            "official_games": len(slate),
            "pregame_games": len(games_out),
            "started_games": started,
            "staked_rows": 0,
            "priced_rows": len(picks),
        },
    }


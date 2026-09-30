"""Daily NHL publisher.

Moneyline, puck line, and game total rows need an observed American price.
Team totals and player props also need an observed scoring mean. Preseason
games are blocked. The fixed EV rule emits research shadow candidates only;
without an unused priced holdout, every visible decision stays PASS at 0u.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

try:
    from nhl_core import canonical_abbrev, counting_outcomes, load_ratings, project_game, total_outcomes
    from nhl_data import is_pregame, load_slate
    from nhl_markets import fetch_pregame_quotes
    from nhl_observed import fetch_player_rates, lookup_player_mean, stat_key_from_market, team_observed_mean
    from nhl_policy import priced_choice, shadow_candidate
except ImportError:
    from .nhl_core import canonical_abbrev, counting_outcomes, load_ratings, project_game, total_outcomes
    from .nhl_data import is_pregame, load_slate
    from .nhl_markets import fetch_pregame_quotes
    from .nhl_observed import fetch_player_rates, lookup_player_mean, stat_key_from_market, team_observed_mean
    from .nhl_policy import priced_choice, shadow_candidate


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


def _price_fields(odds: int | None, odds_source: str | None, quote_at: str | None) -> dict[str, Any]:
    if odds is None:
        return {"odds": None, "pricing_type": "unpriced", "odds_source": None, "market_priced": False}
    fields = {
        "odds": odds,
        "pricing_type": "market",
        "odds_source": odds_source or "observed_quote",
        "market_priced": True,
    }
    if quote_at:
        fields["market_retrieved_at"] = quote_at
    return fields


def _round_features(projection: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in projection.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return rounded


def _reason(evidence_ok: bool) -> str:
    if not evidence_ok:
        return "research_only:team_ratings_unavailable"
    return "research_only:no_validated_segment"


def _add_shadow(
    row: dict[str, Any], *, choice: dict[str, Any] | None, policy: dict[str, Any],
    market: str, evidence_ok: bool, odds_source: str | None, quote_at: str | None,
    published_at: str, start_at: str,
) -> None:
    if choice is not None:
        row["opposite_odds"] = choice["opposite_odds"]
        row["market_implied_probability"] = round(choice["market_implied_probability"], 4)
        row["market_probability"] = row["market_implied_probability"]
        row["edge"] = round(choice["edge"] * 100.0, 2)
        row["expected_value"] = round(choice["expected_value"], 4)
        row["push_probability"] = round(choice["push_probability"], 4)
    decision, units, reason = shadow_candidate(
        policy, market=market, choice=choice, evidence_ok=evidence_ok,
        odds_source=odds_source, quote_at=quote_at,
        published_at=published_at, start_at=start_at,
    )
    row["decision_reason"] = reason
    if decision in {"BET", "LEAN"}:
        row["source_decision"] = decision
        row["source_units"] = units
        row["shadow_decision"] = decision
        row["shadow_units"] = units
        row["staking_policy"] = "awaiting_approved_holdout"


def _prices_seen(game: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    if _american(game.get("home_moneyline")) is not None and _american(game.get("away_moneyline")) is not None:
        seen.append("h2h")
    if _num(game.get("spread_line")) is not None and (
        _american(game.get("home_spread_odds")) is not None or _american(game.get("away_spread_odds")) is not None
    ):
        seen.append("spread")
    if _num(game.get("total_line")) is not None and (
        _american(game.get("over_odds")) is not None or _american(game.get("under_odds")) is not None
    ):
        seen.append("totals")
    team_totals = game.get("team_totals") if isinstance(game.get("team_totals"), dict) else {}
    if any(
        isinstance(row, dict) and _num(row.get("line")) is not None and (
            _american(row.get("over_odds")) is not None or _american(row.get("under_odds")) is not None
        )
        for row in team_totals.values()
    ):
        seen.append("team_total")
    props = game.get("player_props") if isinstance(game.get("player_props"), list) else []
    if any(
        isinstance(prop, dict) and _num(prop.get("line")) is not None and (
            _american(prop.get("over_odds")) is not None or _american(prop.get("under_odds")) is not None
        )
        for prop in props
    ):
        seen.append("player_props")
    return seen


def _needs_player_rates(slate: list[dict[str, Any]]) -> bool:
    for game in slate:
        if not isinstance(game, dict) or not is_pregame(game):
            continue
        if str(game.get("season_type") or "") == "PRE":
            continue
        for prop in game.get("player_props") or []:
            if isinstance(prop, dict) and _num(prop.get("mean")) is None and prop.get("player") and _num(prop.get("line")) is not None:
                return True
    return False


def _row_shell(
    *,
    base: dict[str, Any],
    pick: str,
    market: str,
    probability: float,
    odds: int | None,
    odds_source: str | None,
    quote_at: str | None,
    reason: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base,
        "pick": pick,
        "market": market,
        "market_type": market,
        **extra,
        **_price_fields(odds, odds_source, quote_at),
        "probability": round(probability, 4),
        "raw_probability": round(probability, 4),
        "probability_source": "prior_season_poisson",
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
    player_rates: dict[str, dict[str, Any]] | None = None,
    now: datetime | None = None,
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
    if player_rates is None and games is None and _needs_player_rates(slate):
        try:
            player_rates = fetch_player_rates(str(loaded.get("prior_season") or ""))
        except Exception:
            player_rates = {}
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    clock_iso = clock.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    model_version = str(loaded.get("model_version") or "nhl_poisson_v1")
    policy = loaded.get("decision_policy") if isinstance(loaded.get("decision_policy"), dict) else {}
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
    team_mean_missing = 0
    preseason_blocked = 0
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
        reason = _reason(bool(projection["evidence_ok"]))
        odds_source = str(game.get("odds_source") or "") or None
        quote_at = str(game.get("market_retrieved_at") or "") or None
        prop_quote_at = str(game.get("prop_market_retrieved_at") or quote_at or "") or None
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
            "certification_timing": {
                "trusted": True,
                "published_at": clock_iso,
                "data_as_of": clock_iso,
                "source": "nhl-model-generate",
            },
        }
        prices_seen = _prices_seen(game)
        if str(game.get("season_type") or "") == "PRE":
            preseason_blocked += 1
            games_out.append({
                "game_id": base["game_id"],
                "matchup": matchup,
                "start_time": start,
                "season_type": "PRE",
                "ratings_found": bool(projection["evidence_ok"]),
                "published_markets": [],
                "posted_market_names": list(game.get("posted_market_names") or []),
                "prices_seen": prices_seen,
                "side_published": False,
                "side_block": "preseason_prior_not_validated",
                "prior_season": loaded.get("prior_season"),
            })
            continue
        published_markets: list[str] = []
        home_ml = _american(game.get("home_moneyline"))
        away_ml = _american(game.get("away_moneyline"))
        ml_choice = priced_choice(
            {"home": home_probability, "away": 1.0 - home_probability},
            {"home": home_ml, "away": away_ml},
        ) if projection["ot_home_win_rate"] is not None else None
        pick_home = ml_choice["side"] == "home" if ml_choice else home_probability >= 0.5
        team = home if pick_home else away
        side_probability = home_probability if pick_home else 1.0 - home_probability
        side_odds = home_ml if pick_home else away_ml
        opposite_odds = away_ml if pick_home else home_ml
        if side_odds is None or projection["ot_home_win_rate"] is None:
            skipped["h2h"] += 1
        else:
            pick_row = _row_shell(
                base=base,
                pick=f"{team} ML ({matchup})",
                market="h2h",
                probability=side_probability,
                odds=side_odds,
                odds_source=odds_source,
                quote_at=quote_at,
                reason=reason,
                extra={
                    "team": team,
                    "selection": team,
                    "side": "home" if pick_home else "away",
                    "opposite_odds": opposite_odds,
                    "model_home_win_probability": round(home_probability, 4),
                },
            )
            _add_shadow(
                pick_row, choice=ml_choice, policy=policy, market="h2h",
                evidence_ok=bool(projection["evidence_ok"]), odds_source=odds_source,
                quote_at=quote_at, published_at=clock_iso, start_at=start,
            )
            picks.append(pick_row)
            published_markets.append("h2h")

        home_spread = _num(game.get("spread_line"))
        if home_spread is None:
            skipped["spread"] += 1
        elif abs(abs(home_spread) - 1.5) > 1e-9:
            skipped["spread"] += 1
        else:
            home_cover = (
                float(projection["puckline_home_cover_probability"])
                if home_spread < 0 else 1.0 - float(projection["puckline_away_cover_probability"])
            )
            home_spread_odds = _american(game.get("home_spread_odds"))
            away_spread_odds = _american(game.get("away_spread_odds"))
            spread_choice = priced_choice(
                {"home": home_cover, "away": 1.0 - home_cover},
                {"home": home_spread_odds, "away": away_spread_odds},
            )
            pick_home_spread = spread_choice["side"] == "home" if spread_choice else home_cover >= 0.5
            spread_team = home if pick_home_spread else away
            team_line = home_spread if pick_home_spread else -home_spread
            spread_probability = home_cover if pick_home_spread else 1.0 - home_cover
            spread_odds = home_spread_odds if pick_home_spread else away_spread_odds
            if spread_odds is None:
                skipped["spread"] += 1
            else:
                pick_row = _row_shell(
                    base=base,
                    pick=f"{spread_team} {team_line:+g} ({matchup})",
                    market="spread",
                    probability=spread_probability,
                    odds=spread_odds,
                    odds_source=odds_source,
                    quote_at=quote_at,
                    reason=reason,
                    extra={
                        "team": spread_team,
                        "selection": spread_team,
                        "side": "home" if pick_home_spread else "away",
                        "line": team_line,
                        "market_line": team_line,
                        "posted_home_line": home_spread,
                        "opposite_odds": away_spread_odds if pick_home_spread else home_spread_odds,
                    },
                )
                _add_shadow(
                    pick_row, choice=spread_choice, policy=policy, market="spread",
                    evidence_ok=bool(projection["evidence_ok"]), odds_source=odds_source,
                    quote_at=quote_at, published_at=clock_iso, start_at=start,
                )
                picks.append(pick_row)
                published_markets.append("spread")

        total_line = _num(game.get("total_line"))
        if total_line is None:
            skipped["totals"] += 1
        else:
            over_prob, under_prob, push_prob = total_outcomes(
                float(projection["lambda_home"]),
                float(projection["lambda_away"]),
                total_line,
            )
            over_odds = _american(game.get("over_odds"))
            under_odds = _american(game.get("under_odds"))
            total_choice = priced_choice(
                {"over": over_prob, "under": under_prob},
                {"over": over_odds, "under": under_odds},
                push=push_prob,
            )
            over = total_choice["side"] == "over" if total_choice else over_prob >= under_prob
            direction = "over" if over else "under"
            published_probability = over_prob if over else under_prob
            total_odds = over_odds if over else under_odds
            if total_odds is None:
                skipped["totals"] += 1
            else:
                pick_row = _row_shell(
                    base=base,
                    pick=f"{direction.title()} {total_line:g} ({matchup})",
                    market="totals",
                    probability=published_probability,
                    odds=total_odds,
                    odds_source=odds_source,
                    quote_at=quote_at,
                    reason=reason,
                    extra={
                        "direction": direction,
                        "selection": direction.title(),
                        "line": total_line,
                        "market_line": total_line,
                        "model_expected_total": round(float(projection["expected_total_with_ot_goal"]), 3),
                        "opposite_odds": under_odds if over else over_odds,
                        "push_probability": round(push_prob, 4),
                    },
                )
                _add_shadow(
                    pick_row, choice=total_choice, policy=policy, market="totals",
                    evidence_ok=bool(projection["evidence_ok"]), odds_source=odds_source,
                    quote_at=quote_at, published_at=clock_iso, start_at=start,
                )
                picks.append(pick_row)
                published_markets.append("totals")

        team_totals = game.get("team_totals") if isinstance(game.get("team_totals"), dict) else {}
        if not team_totals:
            skipped["team_total"] += 1
        else:
            published_team_total = False
            saw_team_price = False
            for side, row in team_totals.items():
                if side not in {"home", "away"} or not isinstance(row, dict):
                    continue
                line = _num(row.get("line"))
                if line is None:
                    continue
                if _american(row.get("over_odds")) is not None or _american(row.get("under_odds")) is not None:
                    saw_team_price = True
                team_name = str(row.get("team") or (home if side == "home" else away))
                abbrev = game.get("home_abbrev") if side == "home" else game.get("away_abbrev")
                explicit_mean = _num(row.get("mean"))
                if explicit_mean is not None:
                    observed_mean = explicit_mean
                    mean_source = str(row.get("mean_source") or "observed_quote")
                    mean_games = row.get("mean_games")
                else:
                    observed = team_observed_mean(loaded, abbrev)
                    if observed is None:
                        team_mean_missing += 1
                        continue
                    observed_mean = float(observed["mean"])
                    mean_source = str(observed["mean_source"])
                    mean_games = observed.get("games_played")
                # Standings goals-for already reflects final scores; do not
                # add the game's deciding OT/SO goal to this observed mean.
                over_prob, under_prob, push_prob = counting_outcomes(observed_mean, line)
                take_over = over_prob >= under_prob
                direction = "over" if take_over else "under"
                probability = over_prob if take_over else under_prob
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
                    quote_at=prop_quote_at,
                    reason=reason,
                    extra={
                        "team": team_name,
                        "selection": f"{team_name} {direction.title()}",
                        "side": side,
                        "direction": direction,
                        "line": line,
                        "market_line": line,
                        "mean": round(observed_mean, 4),
                        "mean_source": mean_source,
                        "mean_games": mean_games,
                        "push_probability": round(push_prob, 4),
                    },
                ))
                published_team_total = True
            if published_team_total:
                published_markets.append("team_total")
            elif not saw_team_price:
                skipped["team_total"] += 1

        raw_props = game.get("player_props") if isinstance(game.get("player_props"), list) else []
        if not raw_props:
            skipped["player_props"] += 1
        else:
            published_prop = False
            saw_prop_price = False
            for prop in raw_props:
                if not isinstance(prop, dict):
                    continue
                line = _num(prop.get("line"))
                player = str(prop.get("player") or "").strip()
                if line is None or not player:
                    continue
                if _american(prop.get("over_odds")) is not None or _american(prop.get("under_odds")) is not None:
                    saw_prop_price = True
                mean = _num(prop.get("mean"))
                mean_source = str(prop.get("mean_source") or "observed_quote") if mean is not None else None
                mean_games = prop.get("mean_games")
                stat_key = stat_key_from_market(prop.get("stat") or prop.get("stat_label"))
                if mean is None:
                    looked_up = lookup_player_mean(player_rates, player, stat_key)
                    if looked_up is None:
                        player_prior_missing += 1
                        continue
                    mean = float(looked_up["mean"])
                    mean_source = str(looked_up.get("mean_source") or "")
                    mean_games = looked_up.get("games_played")
                over_prob, under_prob, push_prob = counting_outcomes(mean, line)
                take_over = over_prob >= under_prob
                direction = "over" if take_over else "under"
                probability = over_prob if take_over else under_prob
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
                    quote_at=prop_quote_at,
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
                        "mean": round(mean, 4),
                        "mean_source": mean_source,
                        "mean_games": mean_games,
                        "projection": round(mean, 4),
                        "push_probability": round(push_prob, 4),
                    },
                ))
                published_prop = True
            if published_prop:
                published_markets.append("player_props")
            elif not saw_prop_price:
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
            "prices_seen": prices_seen,
            "side_published": bool(published_markets),
            "side_block": None,
        })
    from scripts.frozen_staking_candidate import stamp_candidate
    from scripts.model_stake_policy import apply_stake_policy
    stamp_candidate(picks, "nhl")
    apply_stake_policy({"models": {"nhl": {"picks": picks}}}, model_keys={"nhl"})
    if not slate:
        note = f"NHL active slate: 0 game(s), 0 row(s). No NHL games scheduled for {date_iso}."
    else:
        note = (
            f"NHL active slate: {len(games_out)} pregame game(s), {len(picks)} priced row(s). "
            "Only exact approved candidates stake; unapproved candidates remain shadow PASS at 0 units."
        )
        if started:
            note += f" {started} started game(s) skipped."
        missing = [name for name, count in skipped.items() if count]
        if missing:
            note += " Markets with no observed price were skipped: " + ", ".join(missing) + "."
        if team_mean_missing:
            note += f" {team_mean_missing} team total(s) had a line but no observed team scoring mean, so no side was published."
        if player_prior_missing:
            note += (
                f" {player_prior_missing} posted player prop(s) had a price but no observed player mean, so no side was published."
            )
        if preseason_blocked:
            note += (
                f" {preseason_blocked} preseason game(s) were not sided. "
                "The completed regular-season prior is not a validated preseason model."
            )
        if any(not game["ratings_found"] for game in games_out):
            note += " One or more teams had no prior-season rating; those rows use league-average rates."
    return {
        "ok": True,
        "date": date_iso,
        "model": "NHLModel",
        "model_version": model_version,
        "decision_policy": policy,
        "shadow_mode": False,
        "actionability": "approved" if any(p["decision"] in {"BET", "LEAN"} for p in picks) else "research",
        "slate_source": slate_source,
        "prior_season": loaded.get("prior_season"),
        "standings_date": loaded.get("standings_date"),
        "picks": picks,
        "games": games_out,
        "note": note,
        "quote_summary": quote_summary,
        "markets_skipped": skipped,
        "player_props_without_prior": player_prior_missing,
        "team_totals_without_mean": team_mean_missing,
        "preseason_blocked": preseason_blocked,
        "shadow_candidate_rows": sum(pick.get("shadow_decision") in {"BET", "LEAN"} for pick in picks),
        "coverage": {
            "official_games": len(slate),
            "pregame_games": len(games_out),
            "started_games": started,
            "staked_rows": sum(pick["decision"] in {"BET", "LEAN"} for pick in picks),
            "shadow_candidate_rows": sum(pick.get("shadow_decision") in {"BET", "LEAN"} for pick in picks),
            "priced_rows": len(picks),
            "preseason_blocked": preseason_blocked,
        },
    }

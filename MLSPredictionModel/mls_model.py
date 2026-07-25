"""MLS model v2 — trained Dixon-Coles engine for moneyline, spread, and total.

The v1 engine here was the FIFA World Cup roster-power heuristic pointed at
MLS: hand-set constants, hundreds of roster/athlete API calls whose league
strength collapsed to "usa.1" for every player, flat home-goal bumps, and no
training or validation of any kind. v2 replaces it with the model documented
in this package's README:

* Time-decayed weighted-MLE Dixon-Coles attack/defense ratings fit from the
  2012-present football-data.co.uk MLS archive (committed with the model),
  hyperparameters chosen by walk-forward validation on 2019-2022 and reported
  against an untouched 2023+ test walk.
* One bivariate score grid prices all three markets, push-aware, at posted
  ESPN/DraftKings lines only — no fabricated lines.
* Published probabilities are the trained vector-scaling calibration of the
  grid, blended with the devigged live market at the validated weight.
* Decisions are confidence gates on that blended probability, mirroring the
  tennis model. The walk-forward backtest showed model-over-market "edge" gates
  invert out of sample (artifacts/backtest.json: edge_gate_rejected), so edge
  is published as information and never required. A fixed price cap keeps
  decided tiers off extreme juice (nothing shorter than -250 publishes as
  BET/LEAN — a fixed cutoff, chosen because every *relative* model-vs-price
  filter anti-selected model blind spots in the backtest). Frozen held-out hit
  rates at the shipped gates with the cap: moneyline BET >=0.60 hit 67.8%
  (n=90), all decided >=0.55 hit 63.2% (n=337) across 2023-2026.

Ratings are refit at serving time from the committed archive plus a
same-morning workbook refresh and an ESPN scoreboard backfill for the last few
days — the exact fit the validation walk scored, not an approximation of it.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

import requests

from .mls_core import (
    FitConfig,
    TotalCalibration,
    VectorScaling,
    american_to_probability,
    conditional_probability,
    devig,
    fit_ratings,
    handicap_split,
    one_x_two,
    score_grid,
    total_split,
)
from .mls_data import (
    ARTIFACT_DIR,
    MlsMatch,
    espn_completed_matches,
    fetch_football_data_matches,
    load_matches,
    merge_matches,
)

ESPN_SITE_API = "https://site.api.espn.com/apis/site/v2/sports/soccer"
USER_AGENT = "PickLedgerMLSModel/2.0"
MODEL_PATH = ARTIFACT_DIR / "mls_model.json"
# ESPN backfill covers the window between the committed archive / workbook and
# today; the cap bounds request volume if artifacts go stale.
MAX_BACKFILL_DAYS = 45


class EspnClient:
    def __init__(self, session: requests.Session | None = None, timeout: int = 18):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout

    def get_json(self, url: str) -> dict[str, Any]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def scoreboard(self, date_iso: str) -> dict[str, Any]:
        compact = date_iso.replace("-", "")
        return self.get_json(f"{ESPN_SITE_API}/usa.1/scoreboard?dates={compact}&limit=100")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _american_odds(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number != 0 else None


def _status_state(event: dict[str, Any]) -> str:
    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
    return str(status_type.get("state") or "").lower()


def _closed_market_value(odds: dict[str, Any], market: str, side: str, field: str = "odds") -> Any:
    market_data = odds.get(market) if isinstance(odds.get(market), dict) else {}
    side_data = market_data.get(side) if isinstance(market_data.get(side), dict) else {}
    close = side_data.get("close") if isinstance(side_data.get("close"), dict) else {}
    open_data = side_data.get("open") if isinstance(side_data.get("open"), dict) else {}
    return close.get(field) if close.get(field) not in {"", None} else open_data.get(field)


def _parse_games(scoreboard: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for event in scoreboard.get("events") if isinstance(scoreboard.get("events"), list) else []:
        if not isinstance(event, dict) or _status_state(event) == "post":
            continue
        competitions = event.get("competitions") if isinstance(event.get("competitions"), list) else []
        competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
        competitors = competition.get("competitors") if isinstance(competition.get("competitors"), list) else []
        home = next((item for item in competitors if isinstance(item, dict) and item.get("homeAway") == "home"), None)
        away = next((item for item in competitors if isinstance(item, dict) and item.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_team = home.get("team") if isinstance(home.get("team"), dict) else {}
        away_team = away.get("team") if isinstance(away.get("team"), dict) else {}
        odds_items = competition.get("odds") if isinstance(competition.get("odds"), list) else []
        venue = competition.get("venue") if isinstance(competition.get("venue"), dict) else {}
        games.append({
            "game_id": str(event.get("id") or ""),
            "start_time": str(event.get("date") or ""),
            "home_id": str(home_team.get("id") or ""),
            "away_id": str(away_team.get("id") or ""),
            "home_name": str(home_team.get("displayName") or home_team.get("name") or "Home"),
            "away_name": str(away_team.get("displayName") or away_team.get("name") or "Away"),
            "venue": str(venue.get("fullName") or venue.get("name") or ""),
            "odds": odds_items[0] if odds_items and isinstance(odds_items[0], dict) else {},
        })
    return games


def _load_artifacts() -> dict[str, Any]:
    import json

    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError("MLS model artifact is malformed")
    return payload


def _refresh_matches(
    archived: list[MlsMatch],
    target: date,
    client: EspnClient,
) -> tuple[list[MlsMatch], dict[str, Any]]:
    """Committed archive + fresh workbook + ESPN backfill, all before ``target``."""
    matches = archived
    meta: dict[str, Any] = {}
    try:
        fresh = fetch_football_data_matches(strict=False)
        before = len(matches)
        matches = merge_matches(matches, fresh)
        meta["workbook_refresh"] = f"merged {len(matches) - before} new workbook matches"
    except Exception as exc:
        meta["workbook_refresh"] = f"unavailable ({type(exc).__name__})"

    last_known = max((match.date for match in matches), default=target - timedelta(days=1))
    start = max(last_known + timedelta(days=1), target - timedelta(days=MAX_BACKFILL_DAYS))
    backfill: list[MlsMatch] = []
    day = start
    days_walked = 0
    while day < target:
        try:
            backfill.extend(espn_completed_matches(client.scoreboard(day.isoformat())))
        except Exception:
            pass
        days_walked += 1
        day += timedelta(days=1)
    if backfill:
        matches = merge_matches(matches, backfill)
    meta["espn_backfill"] = {"days": days_walked, "matches": len(backfill)}
    matches = [match for match in matches if match.date < target]
    meta["ratings_through"] = max((match.date for match in matches), default=last_known).isoformat()
    return matches, meta


def _team_form(matches: list[MlsMatch], team: str, last: int = 5) -> dict[str, Any]:
    recent = [match for match in matches if team in (match.home, match.away)][-last:]
    wins = draws = losses = goals_for = goals_against = 0
    for match in recent:
        us, them = (match.home_goals, match.away_goals) if match.home == team else (match.away_goals, match.home_goals)
        goals_for += us
        goals_against += them
        if us > them:
            wins += 1
        elif us == them:
            draws += 1
        else:
            losses += 1
    return {
        "games": len(recent),
        "record": f"{wins}-{draws}-{losses}",
        "goals_for": goals_for,
        "goals_against": goals_against,
    }


def _blend(model: float, market: float | None, weight: float) -> float:
    if market is None:
        return model
    return (1.0 - weight) * model + weight * market


def _decision(
    blended: float,
    bet_threshold: float,
    lean_threshold: float,
    priced: bool,
    implied: float | None = None,
    max_implied: float | None = None,
) -> str:
    if implied is not None and max_implied is not None and implied > max_implied:
        return "PASS"
    if blended >= bet_threshold and priced:
        return "BET"
    if blended >= lean_threshold:
        return "LEAN"
    return "PASS"


def _units(blended: float, lean_threshold: float, decision: str) -> float:
    if decision == "PASS":
        return 0.0
    return round(_clamp(0.25 + (blended - lean_threshold) * 6.0, 0.25, 1.0), 2)


def _edge_pp(model: float, market: float | None) -> float | None:
    return round((model - market) * 100.0, 2) if market is not None else None


def _round_probs(probabilities: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in probabilities.items()}


def generate_mls_picks(
    date_str: str | None = None,
    *,
    client: EspnClient | None = None,
) -> dict[str, Any]:
    """Generate the cache-ready MLS model bucket for one slate date."""
    date_iso = (
        datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        if date_str
        else datetime.now().strftime("%Y-%m-%d")
    )
    target = date.fromisoformat(date_iso)
    api = client or EspnClient()

    artifact = _load_artifacts()
    config = FitConfig(
        half_life_days=artifact["config"].get("half_life_days"),
        ridge=float(artifact["config"].get("ridge") or 0.0),
    )
    calibration = artifact.get("calibration") or {}
    scaling_params = calibration.get("vector_scaling") or {}
    scaling = VectorScaling(
        gamma=float(scaling_params.get("gamma") or 1.0),
        bias_home=float(scaling_params.get("bias_home") or 0.0),
        bias_away=float(scaling_params.get("bias_away") or 0.0),
    )
    total_params = calibration.get("total") or {}
    total_calibration = TotalCalibration(
        intercept=float(total_params.get("intercept") or 0.0),
        slope=float(total_params.get("slope") or 1.0),
    )
    blend_weight = float(artifact.get("market_blend_weight") or 0.6)
    gates = artifact.get("gates") or {}
    ml_gates = gates.get("moneyline") or {}
    grid_gates = gates.get("grid_markets") or {}
    ml_bet = float(ml_gates.get("bet_blended_probability") or 0.60)
    ml_lean = float(ml_gates.get("lean_blended_probability") or 0.55)
    grid_bet = float(grid_gates.get("bet_blended_probability") or 0.58)
    grid_lean = float(grid_gates.get("lean_blended_probability") or 0.545)
    max_implied = float(gates.get("max_decided_implied_probability") or 0.7143)
    min_games = float(artifact.get("min_effective_games") or 8.0)

    matches, refresh_meta = _refresh_matches(load_matches(), target, api)
    fit = fit_ratings(matches, target, config)

    scoreboard = api.scoreboard(date_iso)
    games = _parse_games(scoreboard)
    model_meta = {
        "model_version": str(artifact.get("model_version") or "mls_dixon_coles_v2"),
        "trained_at": artifact.get("trained_at"),
        "artifact_data_through": artifact.get("data_through"),
        "market_blend_weight": blend_weight,
        "matches_used": fit.matches_used,
        **refresh_meta,
    }
    model_basis = (
        "time-decayed Dixon-Coles ratings trained on 2012-present MLS results; "
        "calibrated score-grid pricing at posted lines; market-blended "
        "probabilities with confidence-gated decisions"
    )
    if not games:
        return {
            "ok": True,
            "date": date_iso,
            "model": model_meta["model_version"],
            "picks": [],
            "games": [],
            "team_ratings": [],
            "calibration_excluded": True,
            "meta": model_meta,
            "schedule_source": "ESPN MLS scoreboard",
            "note": f"No MLS games on ESPN for {date_iso}.",
        }

    picks: list[dict[str, Any]] = []
    game_summaries: list[dict[str, Any]] = []
    for game in games:
        home_key, away_key = game["home_id"], game["away_id"]
        lam_raw, mu_raw = fit.rates(home_key, away_key)
        lam, mu = total_calibration.scale(lam_raw, mu_raw)
        grid = score_grid(lam, mu, fit.rho)
        model_1x2 = scaling.apply(one_x_two(grid))
        ratings_ready = (
            fit.known(home_key)
            and fit.known(away_key)
            and fit.effective_games.get(home_key, 0.0) >= min_games
            and fit.effective_games.get(away_key, 0.0) >= min_games
        )

        odds = game["odds"]
        market_1x2_raw = [
            american_to_probability(_number(_closed_market_value(odds, "moneyline", side)))
            for side in ("home", "draw", "away")
        ]
        market_1x2 = devig(market_1x2_raw)
        market_by_side = {"home": market_1x2[0], "draw": market_1x2[1], "away": market_1x2[2]}
        blended_1x2 = {
            side: _blend(model_1x2[side], market_by_side[side], blend_weight)
            for side in ("home", "draw", "away")
        }

        matchup = f"{game['away_name']} @ {game['home_name']}"
        home_form = _team_form(matches, home_key)
        away_form = _team_form(matches, away_key)
        common = {
            "source": "MLS Model",
            "sport": "MLS",
            "league": "Major League Soccer",
            "date": date_iso,
            "game": matchup,
            "matchup": matchup,
            "away_team": game["away_name"],
            "home_team": game["home_name"],
            "game_id": game["game_id"],
            "start_time": game["start_time"],
            "game_start_time": game["start_time"],
            "venue": game["venue"],
            "calibration_excluded": True,
            "model_basis": model_basis,
            "model_version": model_meta["model_version"],
            "projected_home_goals": round(lam, 3),
            "projected_away_goals": round(mu, 3),
            "projected_total": round(lam + mu, 2),
        }

        # Moneyline: blended-favorite side, team sides only (draw probability
        # is reported, never picked), gated purely on blended confidence.
        side = "home" if blended_1x2["home"] >= blended_1x2["away"] else "away"
        side_team = game["home_name"] if side == "home" else game["away_name"]
        ml_odds = _american_odds(_closed_market_value(odds, "moneyline", side))
        ml_priced = ml_odds is not None and market_by_side[side] is not None
        ml_decision = _decision(
            blended_1x2[side], ml_bet, ml_lean, ml_priced,
            implied=american_to_probability(ml_odds), max_implied=max_implied)
        if not ratings_ready:
            ml_decision = "PASS"
        side_form = home_form if side == "home" else away_form
        picks.append({
            **common,
            "pick": f"{side_team} ML ({matchup})",
            "team": side_team,
            "market": "moneyline",
            "market_type": "soccer_moneyline",
            "odds": ml_odds,
            "probability": round(blended_1x2[side], 4),
            "model_probability": round(model_1x2[side], 4),
            "market_probability": round(market_by_side[side], 4) if market_by_side[side] is not None else None,
            "draw_probability": round(blended_1x2["draw"], 4),
            "edge": _edge_pp(model_1x2[side], market_by_side[side]),
            "decision": ml_decision,
            "units": _units(blended_1x2[side], ml_lean, ml_decision),
            "reason": (
                f"Dixon-Coles rates this {lam:.2f}-{mu:.2f}; blended with the posted 3-way "
                f"market, {side_team} wins {blended_1x2[side]:.1%} (model {model_1x2[side]:.1%}, "
                f"market {market_by_side[side]:.1%} devigged). Last 5: {side_form['record']}, "
                f"{side_form['goals_for']}-{side_form['goals_against']} goals."
                if market_by_side[side] is not None else
                f"Dixon-Coles rates this {lam:.2f}-{mu:.2f}; {side_team} wins "
                f"{blended_1x2[side]:.1%} on the model alone (no posted moneyline)."
            ),
            "key_factors": [
                f"Attack/defense ratings from {fit.matches_used} weighted MLS results",
                f"Projected goals {game['away_name']} {mu:.2f}, {game['home_name']} {lam:.2f}",
                f"{game['home_name']} last 5: {home_form['record']}; {game['away_name']} last 5: {away_form['record']}",
                f"Fitted home advantage {fit.home_advantage:+.2f} log-goals",
                "Confidence-gated on the market-blended probability; no edge claim",
            ],
        })

        # Total: only at the posted line, push-aware.
        total_line = _number(odds.get("overUnder"))
        if total_line is not None:
            split = total_split(grid, total_line)
            over_prob = conditional_probability(split["over"], split["under"])
            model_side = "over" if over_prob >= 0.5 else "under"
            model_prob = over_prob if model_side == "over" else 1.0 - over_prob
            over_price = american_to_probability(_number(_closed_market_value(odds, "total", "over")))
            under_price = american_to_probability(_number(_closed_market_value(odds, "total", "under")))
            market_two_way = devig([over_price, under_price])
            market_prob = market_two_way[0] if model_side == "over" else market_two_way[1]
            blended = _blend(model_prob, market_prob, blend_weight)
            total_odds = _american_odds(_closed_market_value(odds, "total", model_side))
            priced = total_odds is not None and market_prob is not None
            total_decision = _decision(
                blended, grid_bet, grid_lean, priced,
                implied=american_to_probability(total_odds), max_implied=max_implied)
            if not ratings_ready:
                total_decision = "PASS"
            picks.append({
                **common,
                "pick": f"{model_side.title()} {total_line:g} ({matchup})",
                "team": "",
                "market": "total",
                "market_type": "soccer_total",
                "line": total_line,
                "odds": total_odds,
                "probability": round(blended, 4),
                "model_probability": round(model_prob, 4),
                "market_probability": round(market_prob, 4) if market_prob is not None else None,
                "push_probability": round(split["push"], 4),
                "edge": _edge_pp(model_prob, market_prob),
                "decision": total_decision,
                "units": _units(blended, grid_lean, total_decision),
                "reason": (
                    f"Score grid projects {lam:.2f}-{mu:.2f} ({lam + mu:.2f} total); "
                    f"{model_side} {total_line:g} lands {blended:.1%} blended with the posted "
                    f"price (model {model_prob:.1%}, push {split['push']:.1%})."
                ),
                "key_factors": [
                    f"Projected total {lam + mu:.2f} vs posted line {total_line:g}",
                    f"Trained total-rate calibration ({total_calibration.intercept:+.2f} "
                    f"{total_calibration.slope:.2f}x) applied to the grid",
                    "Push-aware pricing at the posted line only",
                ],
            })

        # Spread: every posted Asian-handicap side, best blended probability.
        spread_candidates = []
        for spread_side in ("home", "away"):
            line = _number(_closed_market_value(odds, "pointSpread", spread_side, "line"))
            if line is None:
                continue
            split = handicap_split(grid, spread_side, line)
            model_prob = conditional_probability(split["win"], split["loss"])
            own_price = american_to_probability(_number(_closed_market_value(odds, "pointSpread", spread_side)))
            other_price = american_to_probability(
                _number(_closed_market_value(odds, "pointSpread", "away" if spread_side == "home" else "home")))
            market_two_way = devig([own_price, other_price])
            market_prob = market_two_way[0]
            blended = _blend(model_prob, market_prob, blend_weight)
            spread_candidates.append({
                "side": spread_side,
                "line": line,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "blended": blended,
                "push": split["push"],
                "odds": _american_odds(_closed_market_value(odds, "pointSpread", spread_side)),
            })
        if spread_candidates:
            spread = max(spread_candidates, key=lambda item: item["blended"])
            spread_team = game["home_name"] if spread["side"] == "home" else game["away_name"]
            priced = spread["odds"] is not None and spread["market_prob"] is not None
            spread_decision = _decision(
                spread["blended"], grid_bet, grid_lean, priced,
                implied=american_to_probability(spread["odds"]), max_implied=max_implied)
            if not ratings_ready:
                spread_decision = "PASS"
            line_label = f"{spread['line']:+g}"
            picks.append({
                **common,
                "pick": f"{spread_team} {line_label} ({matchup})",
                "team": spread_team,
                "market": "spread",
                "market_type": "soccer_handicap",
                "line": spread["line"],
                "odds": spread["odds"],
                "probability": round(spread["blended"], 4),
                "model_probability": round(spread["model_prob"], 4),
                "market_probability": round(spread["market_prob"], 4) if spread["market_prob"] is not None else None,
                "push_probability": round(spread["push"], 4),
                "edge": _edge_pp(spread["model_prob"], spread["market_prob"]),
                "decision": spread_decision,
                "units": _units(spread["blended"], grid_lean, spread_decision),
                "reason": (
                    f"{spread_team} {line_label} covers {spread['blended']:.1%} blended "
                    f"(model {spread['model_prob']:.1%}) off the {lam:.2f}-{mu:.2f} score grid."
                ),
                "key_factors": [
                    f"Projected margin {lam - mu:+.2f} goals vs handicap {line_label}",
                    f"Push mass {spread['push']:.1%} settled Asian-style",
                    "Best blended-probability side of the posted handicap only",
                ],
            })

        game_summaries.append({
            **common,
            "home_win_probability": round(blended_1x2["home"], 4),
            "draw_probability": round(blended_1x2["draw"], 4),
            "away_win_probability": round(blended_1x2["away"], 4),
            "model_probabilities": _round_probs(model_1x2),
            "ratings_ready": ratings_ready,
        })

    team_ratings = []
    slate_teams = {key for game in games for key in (game["home_id"], game["away_id"])}
    for team in sorted(fit.teams, key=lambda item: -(fit.attack[item] + fit.defense[item])):
        expected_for = math.exp(fit.intercept + fit.home_advantage / 2.0 + fit.attack[team])
        expected_against = math.exp(fit.intercept + fit.home_advantage / 2.0 - fit.defense[team])
        team_ratings.append({
            "team_id": team,
            "on_slate": team in slate_teams,
            "attack": round(fit.attack[team], 4),
            "defense": round(fit.defense[team], 4),
            "expected_goals_for": round(expected_for, 3),
            "expected_goals_against": round(expected_against, 3),
            "effective_games": round(fit.effective_games.get(team, 0.0), 1),
        })
    for rank, rating in enumerate(team_ratings, start=1):
        rating["slate_rank"] = rank

    decided = sum(1 for pick in picks if pick["decision"] in {"BET", "LEAN"})
    return {
        "ok": True,
        "date": date_iso,
        "model": model_meta["model_version"],
        "picks": picks,
        "games": game_summaries,
        "team_ratings": team_ratings,
        "calibration_excluded": True,
        "meta": model_meta,
        "schedule_source": "ESPN MLS scoreboard",
        "note": (
            f"Priced {len(games)} games ({len(picks)} market rows, {decided} decided) from "
            f"{fit.matches_used} weighted results through {refresh_meta.get('ratings_through')}."
        ),
    }

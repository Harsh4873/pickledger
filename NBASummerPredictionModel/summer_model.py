#!/usr/bin/env python3
"""Conservative NBA Summer League model.

The Summer League slate is separate from the normal NBA scoreboard, and its
rosters are too volatile for the regular-season NBA model. This model uses
only current Summer League results, record context, rest, and venue metadata.
It intentionally avoids full-strength franchise priors except as neutral
fallbacks when no summer sample exists.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ESPN_SUMMER_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba-summer/scoreboard"
)
USER_AGENT = "Mozilla/5.0 PickLedgerPro NBASummer/1.0"

SOURCE_LABEL = "NBA Summer League"
SPORT_LABEL = "NBA SUMMER"
LEAGUE_LABEL = "NBA Summer League"
SUMMER_AVG_TOTAL = 171.5
SUMMER_MARGIN_CAP = 13.0
SUMMER_HOME_ADVANTAGE = 0.25
SUMMER_NEUTRAL_ADVANTAGE = 0.10
SUMMER_LOGISTIC_K = 0.155


@dataclass
class SummerGame:
    game_id: str
    date: str
    start_time: str
    home_team: str
    away_team: str
    home_abbr: str
    away_abbr: str
    venue: str
    tournament: str
    status_state: str
    status_detail: str
    completed: bool
    has_started: bool
    neutral_site: bool
    home_score: int | None = None
    away_score: int | None = None
    home_record: tuple[int, int] | None = None
    away_record: tuple[int, int] | None = None
    market: dict[str, Any] = field(default_factory=dict)

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"


@dataclass
class TeamProfile:
    team: str
    abbr: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    margins: list[tuple[str, float]] = field(default_factory=list)
    record_wins: int | None = None
    record_losses: int | None = None

    def add_game(self, date_str: str, points_for: int, points_against: int) -> None:
        margin = float(points_for) - float(points_against)
        self.games += 1
        self.points_for += float(points_for)
        self.points_against += float(points_against)
        self.margins.append((date_str, margin))
        if margin > 0:
            self.wins += 1
        else:
            self.losses += 1

    def apply_record(self, record: tuple[int, int] | None) -> None:
        if record is None:
            return
        wins, losses = record
        if wins + losses <= 0:
            return
        self.record_wins = wins
        self.record_losses = losses

    @property
    def effective_games(self) -> int:
        record_games = (
            (self.record_wins or 0) + (self.record_losses or 0)
            if self.record_wins is not None and self.record_losses is not None
            else 0
        )
        return max(self.games, record_games)

    @property
    def win_pct(self) -> float:
        if self.games > 0:
            return self.wins / self.games
        if self.record_wins is not None and self.record_losses is not None:
            total = self.record_wins + self.record_losses
            if total:
                return self.record_wins / total
        return 0.5

    @property
    def net_margin(self) -> float:
        return self.points_for / self.games - self.points_against / self.games if self.games else 0.0

    @property
    def recent_margin(self) -> float:
        recent = [margin for _, margin in sorted(self.margins, reverse=True)[:3]]
        if not recent:
            return self.net_margin
        weights = list(range(len(recent), 0, -1))
        denom = sum(weights)
        return sum(value * weight for value, weight in zip(recent, weights)) / denom

    @property
    def avg_points_for(self) -> float:
        return self.points_for / self.games if self.games else SUMMER_AVG_TOTAL / 2.0

    @property
    def avg_points_against(self) -> float:
        return self.points_against / self.games if self.games else SUMMER_AVG_TOTAL / 2.0

    @property
    def last_game_date(self) -> str | None:
        return max((date_str for date_str, _ in self.margins), default=None)


def _normalize_date(raw_value: str | None) -> str:
    if not raw_value:
        return dt.date.today().isoformat()
    value = str(raw_value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return dt.date.today().isoformat()


def _date_key(date_str: str) -> str:
    return dt.date.fromisoformat(date_str).strftime("%Y%m%d")


def _parse_datetime(value: str) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _request_scoreboard(date_str: str) -> dict[str, Any]:
    query = urlencode({"dates": _date_key(date_str), "limit": 100})
    request = Request(
        f"{ESPN_SUMMER_SCOREBOARD_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _team_for_side(competition: dict[str, Any], side: str) -> dict[str, Any] | None:
    for competitor in competition.get("competitors") or []:
        if not isinstance(competitor, dict):
            continue
        if str(competitor.get("homeAway") or "").strip().lower() == side:
            return competitor
    return None


def _team_name(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
    return str(team.get("displayName") or team.get("shortDisplayName") or team.get("name") or "").strip()


def _team_short_name(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
    return str(team.get("name") or team.get("shortDisplayName") or team.get("displayName") or "").strip()


def _team_abbr(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
    return str(team.get("abbreviation") or "").strip().upper()


def _score_value(competitor: dict[str, Any]) -> int | None:
    try:
        return int(float(competitor.get("score")))
    except (TypeError, ValueError):
        return None


def _record_tuple(competitor: dict[str, Any]) -> tuple[int, int] | None:
    for record in competitor.get("records") or []:
        if not isinstance(record, dict):
            continue
        summary = str(record.get("summary") or "").strip()
        match = re.match(r"^(\d+)-(\d+)$", summary)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _parse_american_odds(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _american_to_implied(odds: int | None) -> float | None:
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _remove_vig(home_odds: int | None, away_odds: int | None) -> tuple[float | None, float | None]:
    home_raw = _american_to_implied(home_odds)
    away_raw = _american_to_implied(away_odds)
    if home_raw is None or away_raw is None:
        return None, None
    denom = home_raw + away_raw
    if denom <= 0:
        return None, None
    return home_raw / denom, away_raw / denom


def _quarter_kelly_units(probability: float, odds: int | None, cap: float = 1.0) -> float:
    if odds is None:
        return 0.0
    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / abs(odds)
    if b <= 0:
        return 0.0
    kelly = max((b * probability - (1.0 - probability)) / b, 0.0)
    return round(min(cap, kelly * 0.25), 2)


def _extract_market(competition: dict[str, Any]) -> dict[str, Any]:
    odds_payloads = competition.get("odds")
    odds_payload = odds_payloads[0] if isinstance(odds_payloads, list) and odds_payloads else {}
    if not isinstance(odds_payload, dict):
        odds_payload = {}

    moneyline = odds_payload.get("moneyline") if isinstance(odds_payload.get("moneyline"), dict) else {}
    home_team_odds = odds_payload.get("homeTeamOdds") if isinstance(odds_payload.get("homeTeamOdds"), dict) else {}
    away_team_odds = odds_payload.get("awayTeamOdds") if isinstance(odds_payload.get("awayTeamOdds"), dict) else {}

    home_ml = _parse_american_odds(((moneyline.get("home") or {}).get("close") or {}).get("odds"))
    away_ml = _parse_american_odds(((moneyline.get("away") or {}).get("close") or {}).get("odds"))
    home_ml = home_ml if home_ml is not None else _parse_american_odds(home_team_odds.get("moneyLine"))
    away_ml = away_ml if away_ml is not None else _parse_american_odds(away_team_odds.get("moneyLine"))

    return {
        "provider": str(((odds_payload.get("provider") or {}).get("name") or "ESPN odds")).strip(),
        "home_ml": home_ml,
        "away_ml": away_ml,
    }


def _parse_games(
    payload: dict[str, Any],
    target_date: str,
    *,
    now_utc: dt.datetime | None = None,
) -> list[SummerGame]:
    games: list[SummerGame] = []
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    now_utc = now_utc.astimezone(dt.timezone.utc)
    target_today = target_date == now_utc.date().isoformat()

    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions") if isinstance(event.get("competitions"), list) else []
        competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
        home = _team_for_side(competition, "home")
        away = _team_for_side(competition, "away")
        if not home or not away:
            continue

        status = competition.get("status") if isinstance(competition.get("status"), dict) else {}
        status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
        state = str(status_type.get("state") or "").strip().lower()
        completed = bool(status_type.get("completed")) or state == "post"
        event_dt = _parse_datetime(str(competition.get("date") or event.get("date") or ""))
        has_started = (
            completed
            or state in {"in", "post"}
            or (target_today and event_dt is not None and event_dt <= now_utc)
        )
        venue = competition.get("venue") if isinstance(competition.get("venue"), dict) else {}
        notes = competition.get("notes") or event.get("notes") or []
        tournament = ""
        if isinstance(notes, list) and notes:
            first = notes[0] if isinstance(notes[0], dict) else {}
            tournament = str(first.get("headline") or "").strip()

        games.append(
            SummerGame(
                game_id=str(event.get("id") or competition.get("id") or "").strip(),
                date=target_date,
                start_time=event_dt.isoformat().replace("+00:00", "Z") if event_dt else str(event.get("date") or ""),
                home_team=_team_name(home),
                away_team=_team_name(away),
                home_abbr=_team_abbr(home),
                away_abbr=_team_abbr(away),
                venue=str(venue.get("fullName") or venue.get("name") or "").strip(),
                tournament=tournament or LEAGUE_LABEL,
                status_state=state or "pre",
                status_detail=str(status_type.get("shortDetail") or status_type.get("detail") or status_type.get("description") or "").strip(),
                completed=completed,
                has_started=has_started,
                neutral_site=bool(competition.get("neutralSite", True)),
                home_score=_score_value(home),
                away_score=_score_value(away),
                home_record=_record_tuple(home),
                away_record=_record_tuple(away),
                market=_extract_market(competition),
            )
        )
    return games


def _calendar_history_dates(target_date: str, payload: dict[str, Any]) -> list[str]:
    target = dt.date.fromisoformat(target_date)
    dates: set[str] = set()
    for league in payload.get("leagues") or []:
        if not isinstance(league, dict):
            continue
        for raw in league.get("calendar") or []:
            parsed = _parse_datetime(str(raw))
            if parsed is None:
                continue
            value = parsed.date()
            if value < target:
                dates.add(value.isoformat())
    if not dates:
        for offset in range(1, 15):
            dates.add((target - dt.timedelta(days=offset)).isoformat())
    return sorted(dates)


def _profile_key(team_name: str, abbr: str) -> str:
    return abbr or re.sub(r"[^A-Za-z0-9]+", "", team_name).upper()


def _profiles_from_history(target_date: str, target_payload: dict[str, Any]) -> dict[str, TeamProfile]:
    profiles: dict[str, TeamProfile] = {}

    def profile(team_name: str, abbr: str) -> TeamProfile:
        key = _profile_key(team_name, abbr)
        if key not in profiles:
            profiles[key] = TeamProfile(team=team_name, abbr=abbr)
        return profiles[key]

    history_games: list[SummerGame] = []
    for date_str in _calendar_history_dates(target_date, target_payload):
        try:
            payload = _request_scoreboard(date_str)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
            continue
        history_games.extend(_parse_games(payload, date_str))

    for game in history_games:
        if not game.completed:
            continue
        if game.home_score is None or game.away_score is None:
            continue
        profile(game.home_team, game.home_abbr).add_game(game.date, game.home_score, game.away_score)
        profile(game.away_team, game.away_abbr).add_game(game.date, game.away_score, game.home_score)

    for game in _parse_games(target_payload, target_date):
        profile(game.home_team, game.home_abbr).apply_record(game.home_record)
        profile(game.away_team, game.away_abbr).apply_record(game.away_record)

    return profiles


def _rest_days(profile: TeamProfile, target_date: str) -> int | None:
    last_date = profile.last_game_date
    if not last_date:
        return None
    try:
        return (dt.date.fromisoformat(target_date) - dt.date.fromisoformat(last_date)).days
    except ValueError:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _margin_to_probability(margin: float) -> float:
    raw = 1.0 / (1.0 + math.exp(-SUMMER_LOGISTIC_K * margin))
    return _clamp(raw, 0.18, 0.82)


def _project_matchup(game: SummerGame, home: TeamProfile, away: TeamProfile) -> dict[str, Any]:
    min_sample = min(home.effective_games, away.effective_games)
    sample_weight = _clamp(min_sample / 3.0, 0.25, 1.0)
    home_edge = SUMMER_NEUTRAL_ADVANTAGE if game.neutral_site else SUMMER_HOME_ADVANTAGE
    record_edge = (home.win_pct - away.win_pct) * 4.5
    form_edge = ((home.net_margin - away.net_margin) * 0.55) + ((home.recent_margin - away.recent_margin) * 0.25)

    home_rest = _rest_days(home, game.date)
    away_rest = _rest_days(away, game.date)
    rest_edge = 0.0
    if home_rest is not None and away_rest is not None:
        rest_diff = home_rest - away_rest
        rest_edge = _clamp(rest_diff * 0.45, -1.1, 1.1)
        if away_rest == 1 and home_rest and home_rest > 1:
            rest_edge += 0.45
        if home_rest == 1 and away_rest and away_rest > 1:
            rest_edge -= 0.45

    projected_margin = _clamp(home_edge + record_edge + (form_edge * sample_weight) + rest_edge, -SUMMER_MARGIN_CAP, SUMMER_MARGIN_CAP)
    home_probability = _margin_to_probability(projected_margin)
    projected_total = _clamp(
        ((home.avg_points_for + away.avg_points_against) / 2.0)
        + ((away.avg_points_for + home.avg_points_against) / 2.0),
        145.0,
        195.0,
    )

    return {
        "projected_margin": round(projected_margin, 2),
        "home_probability": round(home_probability, 4),
        "projected_total": round(projected_total, 1),
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "min_sample": min_sample,
        "sample_weight": round(sample_weight, 3),
        "home_profile": {
            "games": home.effective_games,
            "win_pct": round(home.win_pct, 3),
            "net_margin": round(home.net_margin, 2),
            "recent_margin": round(home.recent_margin, 2),
            "points_for": round(home.avg_points_for, 1),
            "points_against": round(home.avg_points_against, 1),
        },
        "away_profile": {
            "games": away.effective_games,
            "win_pct": round(away.win_pct, 3),
            "net_margin": round(away.net_margin, 2),
            "recent_margin": round(away.recent_margin, 2),
            "points_for": round(away.avg_points_for, 1),
            "points_against": round(away.avg_points_against, 1),
        },
        "components": {
            "home_edge": round(home_edge, 2),
            "record_edge": round(record_edge, 2),
            "form_edge": round(form_edge * sample_weight, 2),
            "rest_edge": round(rest_edge, 2),
        },
    }


def _decision_for_pick(
    pick_probability: float,
    pick_margin: float,
    min_sample: int,
    market_edge: float | None,
) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    abs_margin = abs(pick_margin)
    if min_sample < 1:
        reasons.append("no completed Summer League sample for both teams")
    elif min_sample < 2:
        reasons.append("thin Summer League sample")

    decision = "PASS"
    if market_edge is not None:
        if market_edge >= 0.045 and pick_probability >= 0.56 and abs_margin >= 2.0:
            decision = "BET"
        elif market_edge >= 0.025 and pick_probability >= 0.53:
            decision = "LEAN"
    else:
        if min_sample >= 2 and pick_probability >= 0.62 and abs_margin >= 3.5:
            decision = "BET"
        elif min_sample >= 1 and pick_probability >= 0.57 and abs_margin >= 2.0:
            decision = "LEAN"

    if min_sample < 2 and decision == "BET":
        decision = "LEAN"
        reasons.append("BET capped at LEAN until both teams have two summer results")

    if decision == "PASS":
        units = 0.0
    else:
        units = 0.25 + max(0.0, pick_probability - 0.55) * 4.0 + min(abs_margin, 8.0) * 0.035
        if decision == "LEAN":
            units *= 0.65
        units = round(_clamp(units, 0.25, 1.25), 2)
    return decision, units, reasons


def _build_pick(game: SummerGame, projection: dict[str, Any]) -> dict[str, Any]:
    home_probability = float(projection["home_probability"])
    projected_margin = float(projection["projected_margin"])
    pick_is_home = home_probability >= 0.5
    pick_team = game.home_team if pick_is_home else game.away_team
    pick_probability = home_probability if pick_is_home else 1.0 - home_probability
    pick_margin = projected_margin if pick_is_home else -projected_margin

    home_ml = game.market.get("home_ml")
    away_ml = game.market.get("away_ml")
    home_market_prob, away_market_prob = _remove_vig(home_ml, away_ml)
    market_pick_prob = home_market_prob if pick_is_home else away_market_prob
    market_pick_odds = home_ml if pick_is_home else away_ml
    market_edge = pick_probability - market_pick_prob if market_pick_prob is not None else None

    decision, units, reasons = _decision_for_pick(
        pick_probability=pick_probability,
        pick_margin=pick_margin,
        min_sample=int(projection["min_sample"]),
        market_edge=market_edge,
    )
    if decision != "PASS" and market_pick_odds is not None:
        kelly_units = _quarter_kelly_units(pick_probability, market_pick_odds, cap=1.25)
        units = min(units, kelly_units) if kelly_units > 0 else 0.0
        if units == 0.0:
            decision = "PASS"
            reasons.append("market price removed the stake edge")

    confidence_label = "High" if decision == "BET" else "Medium" if decision == "LEAN" else "Low"
    factors = [
        f"Summer form margin {projection['components']['form_edge']:+.1f}",
        f"Record edge {projection['components']['record_edge']:+.1f}",
        f"Rest edge {projection['components']['rest_edge']:+.1f}",
        f"Projected total {projection['projected_total']:.1f}",
    ]
    if market_edge is not None:
        factors.append(f"Market edge {market_edge * 100:+.1f}%")

    return {
        "source": SOURCE_LABEL,
        "sport": SPORT_LABEL,
        "league": LEAGUE_LABEL,
        "date": game.date,
        "game_id": game.game_id,
        "start_time": game.start_time,
        "game_start_time": game.start_time,
        "game": game.matchup,
        "matchup": game.matchup,
        "away_team": game.away_team,
        "home_team": game.home_team,
        "team": pick_team,
        "pick": f"{pick_team} ML ({game.matchup})",
        "market_type": "h2h",
        "selection": pick_team,
        "probability": round(pick_probability, 4),
        "confidence": round(pick_probability * 100.0, 1),
        "confidence_label": confidence_label,
        "decision": decision,
        "units": units,
        "odds": market_pick_odds,
        "market_pick_odds": market_pick_odds,
        "market_pick_prob": round(market_pick_prob, 4) if market_pick_prob is not None else None,
        "market_edge": round(market_edge, 4) if market_edge is not None else None,
        "has_market_price": market_pick_prob is not None,
        "market_source": game.market.get("provider") if market_pick_prob is not None else None,
        "model_prediction": round(pick_margin, 2),
        "projected_margin": projected_margin,
        "projected_total": projection["projected_total"],
        "neutral_site": game.neutral_site,
        "venue": game.venue,
        "tournament": game.tournament,
        "sample_games": projection["min_sample"],
        "key_factors": factors,
        "guardrail_reasons": reasons,
        "notes": " | ".join(factors + reasons),
    }


def generate_nba_summer_picks(
    date_str: str | None = None,
    echo: bool = False,
    *,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Return a model-cache-ready NBA Summer League payload for *date_str*."""
    target_date = _normalize_date(date_str)
    try:
        target_payload = _request_scoreboard(target_date)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "error": f"NBA Summer scoreboard unavailable: {exc}"}

    games = _parse_games(target_payload, target_date, now_utc=now_utc)
    if not games:
        return {
            "ok": True,
            "picks": [],
            "games": [],
            "note": f"No NBA Summer League games on ESPN scoreboard for {target_date}.",
            "slate_games": 0,
            "schedule_source": "ESPN nba-summer scoreboard",
        }

    profiles = _profiles_from_history(target_date, target_payload)
    picks: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    skipped_started = 0

    for game in games:
        game_row = {
            "game_id": game.game_id,
            "date": game.date,
            "start_time": game.start_time,
            "game_start_time": game.start_time,
            "away_team": game.away_team,
            "home_team": game.home_team,
            "game": game.matchup,
            "matchup": game.matchup,
            "status_state": game.status_state,
            "status_detail": game.status_detail,
            "neutral_site": game.neutral_site,
            "venue": game.venue,
            "tournament": game.tournament,
        }
        if game.has_started:
            game_row["skipped_reason"] = "game already started"
            skipped_started += 1
            game_rows.append(game_row)
            continue

        home = profiles.get(_profile_key(game.home_team, game.home_abbr), TeamProfile(game.home_team, game.home_abbr))
        away = profiles.get(_profile_key(game.away_team, game.away_abbr), TeamProfile(game.away_team, game.away_abbr))
        home.apply_record(game.home_record)
        away.apply_record(game.away_record)
        projection = _project_matchup(game, home, away)
        game_row["projection"] = projection
        game_rows.append(game_row)
        pick = _build_pick(game, projection)
        picks.append(pick)
        if echo:
            print(
                f"{SOURCE_LABEL} | {game.matchup} | {pick['pick']} | "
                f"{pick['confidence']}% | {pick['decision']} | {pick['units']}u"
            )

    visible_picks = [pick for pick in picks if str(pick.get("decision") or "").upper() in {"BET", "LEAN"}]
    return {
        "ok": True,
        "picks": picks,
        "games": game_rows,
        "slate_games": len(games),
        "eligible_games": len(games) - skipped_started,
        "skipped_started_games": skipped_started,
        "visible_picks": len(visible_picks),
        "schedule_source": "ESPN nba-summer scoreboard",
        "model_version": "nba_summer_v1.0.0",
        "note": (
            f"NBA Summer League model evaluated {len(games) - skipped_started} "
            f"pregame matchup(s); skipped {skipped_started} started game(s)."
        ),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the NBA Summer League model.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    parser.add_argument("--echo", action="store_true", help="Print readable picks before JSON.")
    args = parser.parse_args()
    payload = generate_nba_summer_picks(args.date or None, echo=args.echo)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

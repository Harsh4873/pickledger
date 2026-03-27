from __future__ import annotations

from datetime import date, datetime
from typing import Mapping


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _wind_direction(row: Mapping[str, object]) -> str:
    return str(row.get("wind_direction") or "unknown").lower()


def _game_month(row: Mapping[str, object]) -> int | None:
    game_date = row.get("game_date")
    if isinstance(game_date, datetime):
        return game_date.month
    if isinstance(game_date, date):
        return game_date.month
    if isinstance(game_date, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(game_date, fmt).month
            except ValueError:
                continue
    return None


def _season_phase_total_adjustment(row: Mapping[str, object]) -> float:
    month = _game_month(row)
    if month in {3, 4}:
        return 0.8
    return 0.0


def calculate_layer1_base_rate(row: Mapping[str, object]) -> float:
    """
    Historical win baseline from team strength only.

    The value is oriented from the home team's perspective so it can be used
    directly as a home-win probability feature in the hybrid model.
    """
    home_season = _float(row, "home_team_win_pct_shrunk", _float(row, "home_season_win_pct", 0.5))
    home_recent = _float(row, "home_form_30d_win_pct_shrunk", _float(row, "home_form_30d_win_pct", 0.5))
    home_prior = _float(row, "home_prior_season_win_pct", 0.5)

    away_season = _float(row, "away_team_win_pct_shrunk", _float(row, "away_season_win_pct", 0.5))
    away_recent = _float(row, "away_form_30d_win_pct_shrunk", _float(row, "away_form_30d_win_pct", 0.5))
    away_prior = _float(row, "away_prior_season_win_pct", 0.5)

    home_strength = 0.45 * home_season + 0.30 * home_recent + 0.25 * home_prior
    away_strength = 0.45 * away_season + 0.30 * away_recent + 0.25 * away_prior

    total_strength = home_strength + away_strength
    if total_strength <= 0:
        return 0.5
    return max(0.05, min(0.95, home_strength / total_strength))


def calculate_layer2_situational(row: Mapping[str, object]) -> tuple[float, str]:
    adj = 0.0
    reasons: list[str] = []

    wind_direction = _wind_direction(row)
    wind_speed = _float(row, "wind_speed_mph")
    park_factor = _float(row, "park_factor_runs", 100.0)

    if wind_direction == "out" and wind_speed >= 15:
        adj += 0.01
        reasons.append("wind out >=15 mph")
    elif wind_direction == "in" and wind_speed >= 15:
        adj -= 0.01
        reasons.append("wind in >=15 mph")

    if park_factor >= 107:
        adj += 0.01
        reasons.append("hitter park")
    elif park_factor <= 95:
        adj -= 0.01
        reasons.append("pitcher park")

    home_rest = _float(row, "home_rest_days")
    away_rest = _float(row, "away_rest_days")
    if home_rest - away_rest >= 1:
        adj += 0.015
        reasons.append("home rest edge")
    elif away_rest - home_rest >= 1:
        adj -= 0.015
        reasons.append("away rest edge")

    home_travel = _float(row, "home_travel_flag")
    away_travel = _float(row, "away_travel_flag")
    if away_travel > home_travel:
        adj += 0.02
        reasons.append("away travel fatigue")
    elif home_travel > away_travel:
        adj -= 0.02
        reasons.append("home travel fatigue")

    bullpen_usage_edge = _float(row, "away_bullpen_pitches_3d") - _float(row, "home_bullpen_pitches_3d")
    if bullpen_usage_edge >= 35:
        adj += 0.02
        reasons.append("away bullpen taxed")
    elif bullpen_usage_edge <= -35:
        adj -= 0.02
        reasons.append("home bullpen taxed")

    adj = max(-0.12, min(0.12, adj))
    return adj, ", ".join(reasons) if reasons else "neutral context"


def calculate_layer3_pitcher_modifier(row: Mapping[str, object]) -> tuple[float, str]:
    home_fip = _float(row, "home_starter_fip_shrunk", _float(row, "home_starter_fip", 4.2))
    away_fip = _float(row, "away_starter_fip_shrunk", _float(row, "away_starter_fip", 4.2))
    fip_edge = away_fip - home_fip
    fip_edge = max(-1.8, min(1.8, fip_edge))
    modifier = (fip_edge / 0.5) * 0.025
    modifier = max(-0.09, min(0.09, modifier))
    return modifier, f"starter FIP edge {fip_edge:+.2f}"


def extremize_probability(raw_prob: float, factor: float = 1.12) -> float:
    extremized = 0.50 + (raw_prob - 0.50) * factor
    return max(0.05, min(0.95, extremized))


def heuristic_home_win_probability(row: Mapping[str, object]) -> float:
    base = calculate_layer1_base_rate(row)
    situational, _ = calculate_layer2_situational(row)
    pitcher, _ = calculate_layer3_pitcher_modifier(row)
    raw = max(0.05, min(0.95, base + situational + pitcher))
    return extremize_probability(raw)


def heuristic_features(row: Mapping[str, object]) -> dict[str, float]:
    base = calculate_layer1_base_rate(row)
    situational, _ = calculate_layer2_situational(row)
    pitcher, _ = calculate_layer3_pitcher_modifier(row)
    raw = max(0.05, min(0.95, base + situational + pitcher))
    heuristic_prob = extremize_probability(raw)

    return {
        "heuristic_layer1_home_prob": base,
        "heuristic_layer2_adj": situational,
        "heuristic_layer3_adj": pitcher,
        "heuristic_raw_home_prob": raw,
        "heuristic_home_win_prob": heuristic_prob,
    }


def predict_total_runs(row: Mapping[str, object]) -> float:
    """
    Heuristic total used as a feature and safe fallback.

    The dedicated totals model added later uses this as one input, not as the
    final answer.
    """
    base_runs = 8.7
    if base_runs < 9.0:
        base_runs = 9.0
    base_runs += _season_phase_total_adjustment(row)
    starter_component = (
        (_float(row, "home_starter_fip_shrunk", _float(row, "home_starter_fip", 4.2)) - 4.2)
        + (_float(row, "away_starter_fip_shrunk", _float(row, "away_starter_fip", 4.2)) - 4.2)
    )
    bullpen_component = (
        ((_float(row, "home_bullpen_era_30d", 4.2) - 4.2) * 0.35)
        + ((_float(row, "away_bullpen_era_30d", 4.2) - 4.2) * 0.35)
    )
    lineup_component = (
        ((_float(row, "home_lineup_ops_proxy_shrunk", _float(row, "home_lineup_ops_proxy", 0.710)) - 0.710) * 8.0)
        + ((_float(row, "away_lineup_ops_proxy_shrunk", _float(row, "away_lineup_ops_proxy", 0.710)) - 0.710) * 8.0)
    )
    park_component = (_float(row, "park_factor_runs", 100.0) - 100.0) * 0.05

    weather_component = 0.0
    wind_direction = _wind_direction(row)
    wind_speed = _float(row, "wind_speed_mph")
    temperature = _float(row, "temperature_f", 72.0)
    if wind_direction == "out":
        weather_component += min(1.6, wind_speed * 0.08)
    elif wind_direction == "in":
        weather_component -= min(1.4, wind_speed * 0.07)
    if temperature >= 85:
        weather_component += 0.35
    elif 0 < temperature <= 50:
        weather_component -= 0.45

    total = base_runs + starter_component + bullpen_component + lineup_component + park_component + weather_component
    return round(max(5.5, min(13.5, total)), 3)


def predict_spread(home_prob: float) -> float:
    return round((home_prob - 0.50) * 9.0, 3)

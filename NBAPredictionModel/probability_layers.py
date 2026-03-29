import math
import re
from functools import lru_cache

from data_models import Team, GameContext

_PROB_EPSILON = 1e-6
_MIN_ON_OFF_SAMPLE_GAMES = 10
_DEFAULT_UNMATCHED_PLAYER_PENALTY = 0.05
_ROTATION_BROKEN_MULTIPLIER = 1.5
_MAX_TOTAL_INJURY_ADJ = 0.25
_LEAGUE_AVG_PACE = 99.0
_DEFAULT_OPP_TOV_PCT = 0.135
_DEFAULT_DREB_PCT = 0.720
_RECENT_FORM_METRIC_FIELDS = {
    "recent_5_point_diff": ("raw_recent_5_point_diff", "capped_recent_5_point_diff"),
    "recent_10_point_diff": ("raw_recent_10_point_diff", "capped_recent_10_point_diff"),
    "weighted_point_diff": ("raw_weighted_point_diff", "capped_weighted_point_diff"),
}


def _clamp_probability(prob: float) -> float:
    return max(_PROB_EPSILON, min(1.0 - _PROB_EPSILON, prob))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _coerce_float(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _get_recent_form_metric(team_stats, metric_name: str, use_capped_form: bool = False) -> float:
    raw_attr, capped_attr = _RECENT_FORM_METRIC_FIELDS.get(metric_name, (metric_name, metric_name))
    raw_default = _coerce_float(
        getattr(team_stats, raw_attr, getattr(team_stats, metric_name, getattr(team_stats, "net_rating", 0.0))),
        _coerce_float(getattr(team_stats, "net_rating", 0.0), 0.0),
    )
    if not use_capped_form:
        return raw_default
    return _coerce_float(getattr(team_stats, capped_attr, raw_default), raw_default)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _logit(prob: float) -> float:
    clean_prob = _clamp_probability(prob)
    return math.log(clean_prob / (1.0 - clean_prob))


def _normalize_player_key(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _match_player_name(player_name: str, candidate_names: list[str]) -> str | None:
    target = _normalize_player_key(player_name)
    if not target:
        return None

    normalized_candidates = []
    for candidate in candidate_names:
        normalized = _normalize_player_key(candidate)
        if normalized:
            normalized_candidates.append((normalized, candidate))

    for normalized, candidate in normalized_candidates:
        if normalized == target:
            return candidate

    for normalized, candidate in normalized_candidates:
        if target in normalized or normalized in target:
            return candidate

    return None


def _position_group(position: str) -> str:
    cleaned = re.sub(r"[^A-Z]", " ", str(position or "").upper())
    tokens = {token for token in cleaned.split() if token}

    if tokens & {"C", "PF", "FC", "CF"} or "CENTER" in cleaned:
        return "Bigs"
    if tokens & {"SF", "F"}:
        return "Wings"
    if tokens & {"PG", "SG", "G"}:
        return "Guards"
    if "G" in cleaned:
        return "Guards"
    if "F" in cleaned:
        return "Wings"
    return "Unknown"


@lru_cache(maxsize=64)
def _fetch_team_on_off_data(team_name: str, season: str) -> dict:
    try:
        from injury_impact import fetch_player_on_off
    except Exception as exc:
        print(f"  WARNING: Could not import injury on/off module for {team_name}: {exc}")
        return {}

    try:
        return fetch_player_on_off(team_name, season) or {}
    except Exception as exc:
        print(f"  WARNING: Could not fetch on/off data for {team_name}: {exc}")
        return {}


@lru_cache(maxsize=64)
def _fetch_team_roster(team_name: str, season: str) -> list:
    try:
        from live_data import fetch_roster
    except Exception as exc:
        print(f"  WARNING: Could not import roster lookup for {team_name}: {exc}")
        return []

    try:
        return fetch_roster(team_name, season) or []
    except Exception as exc:
        print(f"  WARNING: Could not fetch roster for {team_name}: {exc}")
        return []


def _build_roster_context(team_name: str, season: str, player_data: dict) -> tuple[dict, dict]:
    roster = _fetch_team_roster(team_name, season)
    position_lookup = {}
    depth_chart = {"Guards": [], "Wings": [], "Bigs": []}
    on_off_names = list(player_data.keys())

    for player in roster:
        roster_name = str(player.get("name", "")).strip()
        if not roster_name:
            continue

        normalized_name = _normalize_player_key(roster_name)
        group = _position_group(player.get("position", ""))
        matched_on_off = _match_player_name(roster_name, on_off_names)
        mpg = 0.0
        if matched_on_off:
            mpg = float(player_data.get(matched_on_off, {}).get("mpg", 0.0) or 0.0)

        position_lookup[normalized_name] = {
            "name": roster_name,
            "position": str(player.get("position", "")).strip(),
            "position_group": group,
            "matched_on_off": matched_on_off,
            "mpg": mpg,
        }

        if group in depth_chart:
            depth_chart[group].append({
                "name": roster_name,
                "normalized_name": normalized_name,
                "mpg": mpg,
            })

    for players in depth_chart.values():
        players.sort(key=lambda item: item["mpg"], reverse=True)

    return position_lookup, depth_chart


def _calculate_expected_player_penalty(player_name: str, status: str, absence_probability: float, player_data: dict) -> tuple[float, str, dict]:
    matched = _match_player_name(player_name, list(player_data.keys()))
    if not matched:
        expected_penalty = _DEFAULT_UNMATCHED_PLAYER_PENALTY * absence_probability
        return (
            -expected_penalty,
            f"{player_name} {status} ({absence_probability:.0%} miss, default EV {-expected_penalty*100:+.1f}%)",
            {
                "matched_name": player_name,
                "base_penalty": _DEFAULT_UNMATCHED_PLAYER_PENALTY,
                "expected_penalty": expected_penalty,
                "mpg": 0.0,
                "gp": 0,
                "impact": 0.0,
            },
        )

    data = player_data.get(matched, {})
    gp = int(data.get("gp", 0) or 0)
    mpg = float(data.get("mpg", 0.0) or 0.0)
    raw_impact = float(data.get("impact", 0.0) or 0.0)

    if gp < _MIN_ON_OFF_SAMPLE_GAMES:
        return (
            0.0,
            f"{matched} {status} ({absence_probability:.0%} miss, only {gp} GP so skipped)",
            {
                "matched_name": matched,
                "base_penalty": 0.0,
                "expected_penalty": 0.0,
                "mpg": mpg,
                "gp": gp,
                "impact": raw_impact,
            },
        )

    minute_weight = min(1.0, mpg / 30.0) if mpg > 0 else 0.0
    if raw_impact <= 0:
        if mpg >= 25.0:
            base_penalty = 0.03
            expected_penalty = base_penalty * absence_probability
            return (
                -expected_penalty,
                (
                    f"{matched} {status} ({absence_probability:.0%} miss, {mpg:.0f} MPG starter floor "
                    f"EV {-expected_penalty*100:+.1f}%)"
                ),
                {
                    "matched_name": matched,
                    "base_penalty": base_penalty,
                    "expected_penalty": expected_penalty,
                    "mpg": mpg,
                    "gp": gp,
                    "impact": raw_impact,
                },
            )
        return (
            0.0,
            f"{matched} {status} ({absence_probability:.0%} miss, negative on/off impact so neutral)",
            {
                "matched_name": matched,
                "base_penalty": 0.0,
                "expected_penalty": 0.0,
                "mpg": mpg,
                "gp": gp,
                "impact": raw_impact,
            },
        )

    base_penalty = min(0.12, raw_impact * 0.03 * minute_weight)
    expected_penalty = base_penalty * absence_probability
    return (
        -expected_penalty,
        (
            f"{matched} {status} ({absence_probability:.0%} miss, Impact {raw_impact:+.1f}, "
            f"{mpg:.0f} MPG EV {-expected_penalty*100:+.1f}%)"
        ),
        {
            "matched_name": matched,
            "base_penalty": base_penalty,
            "expected_penalty": expected_penalty,
            "mpg": mpg,
            "gp": gp,
            "impact": raw_impact,
        },
    )


def calculate_injury_adjustment(team_name: str, injury_entries: list[dict], season: str = "2025-26") -> tuple[float, str]:
    """
    NBANEW injury adjustment based on expected absence value and positional depth.

    Each player's standard on/off penalty is scaled by the probability they sit.
    When the top two players in a position group both carry >50% absence risk,
    the group's combined expected penalty is multiplied by 1.5 to reflect a
    broken rotation.
    """
    if not injury_entries:
        return 0.0, "No expected injury absences"

    player_data = _fetch_team_on_off_data(team_name, season)
    position_lookup, depth_chart = _build_roster_context(team_name, season, player_data)

    total_adj = 0.0
    reasons = []
    modeled_players = []

    for entry in injury_entries:
        player_name = str(entry.get("name", "")).strip()
        status = str(entry.get("status", "")).strip() or "Unknown"
        absence_probability = float(entry.get("absence_probability", 0.0) or 0.0)
        if not player_name or absence_probability <= 0.0:
            continue

        player_adj, reason, details = _calculate_expected_player_penalty(
            player_name,
            status,
            absence_probability,
            player_data,
        )
        total_adj += player_adj
        reasons.append(reason)

        roster_match = position_lookup.get(_normalize_player_key(player_name), {})
        position_group = roster_match.get("position_group", "Unknown")
        matched_name = details.get("matched_name", player_name)

        if position_group == "Unknown" and matched_name:
            roster_match = position_lookup.get(_normalize_player_key(matched_name), {})
            position_group = roster_match.get("position_group", "Unknown")

        modeled_players.append({
            "name": player_name,
            "matched_name": matched_name,
            "absence_probability": absence_probability,
            "expected_penalty": abs(player_adj),
            "position_group": position_group,
        })

    injury_probability_lookup = {}
    for player in modeled_players:
        injury_probability_lookup[_normalize_player_key(player["name"])] = player["absence_probability"]
        injury_probability_lookup[_normalize_player_key(player["matched_name"])] = player["absence_probability"]

    for group_name, players in depth_chart.items():
        if len(players) < 2:
            continue

        top_two = players[:2]
        if top_two[0]["mpg"] <= 0.0 and top_two[1]["mpg"] <= 0.0:
            continue
        if not all(injury_probability_lookup.get(player["normalized_name"], 0.0) > 0.5 for player in top_two):
            continue

        group_penalty = sum(
            player["expected_penalty"]
            for player in modeled_players
            if player["position_group"] == group_name
        )
        if group_penalty <= 0.0:
            continue

        extra_penalty = group_penalty * (_ROTATION_BROKEN_MULTIPLIER - 1.0)
        total_adj -= extra_penalty
        top_two_names = ", ".join(player["name"] for player in top_two)
        reasons.append(
            f"Rotation Broken {group_name} ({top_two_names}) x{_ROTATION_BROKEN_MULTIPLIER:.1f} EV {-extra_penalty*100:+.1f}%"
        )

    total_adj = max(-_MAX_TOTAL_INJURY_ADJ, total_adj)
    reason_str = ", ".join(reasons) if reasons else "No injury adjustments"
    return total_adj, reason_str


def calculate_layer1_base_rate(
    team: Team,
    opp_team: Team,
    h2h_win_pct_2yr: float,
    use_capped_form: bool = False,
) -> float:
    """
    Base rate blends season quality with recency-sensitive 5/10 game form.
    """
    net_rtg_weight = 0.28
    recent_5_weight = 0.22
    recent_10_weight = 0.18
    weighted_form_weight = 0.12
    recent_point_diff_weight = 0.10
    season_weight = 0.05
    h2h_weight = 0.05
    
    # Net Rating Diff to Probability mapping
    net_rtg_diff = team.team_stats.net_rating - opp_team.team_stats.net_rating
    # A 1 net rating diff is roughly 3% win prob edge from 50%
    net_rtg_prob = 0.50 + (net_rtg_diff * 0.03)
    net_rtg_prob = max(0.05, min(0.95, net_rtg_prob))

    recent_point_diff = _get_recent_form_metric(
        team.team_stats,
        "weighted_point_diff",
        use_capped_form=use_capped_form,
    ) - _get_recent_form_metric(
        opp_team.team_stats,
        "weighted_point_diff",
        use_capped_form=use_capped_form,
    )
    recent_point_prob = max(0.05, min(0.95, 0.50 + (recent_point_diff * 0.018)))
    
    home_court_prior = 0.015 if team.is_home else -0.015

    base_rate = (
        (net_rtg_prob * net_rtg_weight) +
        (team.team_stats.recent_5_win_pct * recent_5_weight) +
        (team.team_stats.recent_10_win_pct * recent_10_weight) +
        (team.team_stats.weighted_win_pct * weighted_form_weight) +
        (recent_point_prob * recent_point_diff_weight) +
        (team.team_stats.season_win_pct * season_weight) +
        (h2h_win_pct_2yr * h2h_weight) +
        home_court_prior
    )
                
    return base_rate

def calculate_layer2_situational(
    team: Team,
    opp_team: Team,
    game_ctx: GameContext,
    use_advanced_fatigue: bool = False,
) -> tuple[float, str]:
    """
    Apply adjustments to the base rate based on tonight's specific context.
    Cap total situational adjustment at ±15%.

    The advanced schedule-density penalties are reserved for NBANEW via the
    `use_advanced_fatigue` gate so the legacy pipeline keeps its current math.
    """
    adj = 0.0
    reasons = []
    
    # 1. Schedule Loss (2nd night of B2B)
    if team.team_stats.is_b2b_second_leg:
        adj -= 0.08
        reasons.append("Schedule Loss (B2B) -8.0%")

    rest_edge = team.team_stats.rest_days - opp_team.team_stats.rest_days
    if abs(rest_edge) >= 0.5:
        rest_adj = max(-0.05, min(0.05, rest_edge * 0.015))
        adj += rest_adj
        reasons.append(f"Rest day edge {rest_adj*100:+.1f}%")

    if team.team_stats.back_to_back_flag and not team.team_stats.is_b2b_second_leg:
        adj -= 0.04
        reasons.append("Back-to-back flag -4.0%")

    if team.team_stats.is_3_in_4_nights:
        adj -= 0.03
        reasons.append("3 games in 4 nights -3.0%")

    if use_advanced_fatigue:
        if bool(getattr(team.team_stats, "is_4_in_5_nights", False)):
            adj -= 0.025
            reasons.append("4 games in 5 nights -2.5%")

        if bool(getattr(team.team_stats, "is_5_in_7_nights", False)):
            adj -= 0.015
            reasons.append("5 games in 7 nights -1.5%")

        try:
            current_road_trip_length = max(
                0,
                int(getattr(team.team_stats, "current_road_trip_length", 0) or 0),
            )
        except (TypeError, ValueError):
            current_road_trip_length = 0

        if current_road_trip_length >= 4:
            adj -= 0.015
            reasons.append(f"Road Weary ({current_road_trip_length} straight road games) -1.5%")
        
    # 2. Key Star Player Out
    if team.key_stars_out:
        adj -= 0.10
        reasons.append("Key Star Player Out (Usage > 25%) -10.0%")
        # "Next Man Up" (Ewing Theory): If it's JUST the star missing and the depth is healthy
        if team.rotation_players_out == 0:
            adj += 0.03
            reasons.append("Next Man Up Focus (Single Star Out) +3.0%")
            
    # Additional Rotation Players Out (Testing Depth)
    if team.rotation_players_out > 0:
        penalty = team.rotation_players_out * 0.02
        adj -= penalty
        reasons.append(f"Missing {team.rotation_players_out} Rotation/Bench Player(s) -{penalty*100:.1f}%")
        
    # 3. Starting Center Out
    if team.starting_center_out:
        adj -= 0.05
        reasons.append("Starting Center Out -5.0%")
        
    # 4. Motivation / Elimination Game
    if team.motivation_elimination_game:
        adj += 0.03
        reasons.append("Motivation/Elimination Game +3.0%")

    if team.injury_flag:
        injury_adj = min(0.08, 0.015 + (team.injury_severity * 0.45))
        adj -= injury_adj
        reasons.append(f"Injury flag -{injury_adj*100:.1f}%")

    # 5. Home floor routine / road penalty
    if team.is_home:
        adj += 0.005
        reasons.append("Home floor routine +0.5%")
    else:
        adj -= 0.005
        reasons.append("Road travel context -0.5%")
        
    # Cap total situational adjustment at ±15% (0.15)
    adj = max(-0.15, min(0.15, adj))
    
    reason_str = ", ".join(reasons) if reasons else "No major situational adjustments"
    return adj, reason_str


def calculate_layer3_matchup_modifier(
    team: Team,
    opp_team: Team,
    tempo_context: dict | None = None,
    use_capped_form: bool = False,
) -> tuple[float, str]:
    """
    Matchup Modifier (Pace & Efficiency)
    - Pace edge = High pace vs bottom-10 defense -> +3%
    - Rebound edge = REB% > 3% higher -> +2%
    """
    adj = 0.0
    reasons = []
    if tempo_context:
        team_side = "home" if team.is_home else "away"
        team_weight = _coerce_float(tempo_context.get(f"{team_side}_weight", 0.50), 0.50)
        opp_weight = _coerce_float(
            tempo_context.get("away_weight" if team_side == "home" else "home_weight", 0.50),
            0.50,
        )
        preferred_pace = _coerce_float(getattr(team.team_stats, "pace", _LEAGUE_AVG_PACE), _LEAGUE_AVG_PACE)
        if preferred_pace >= (_LEAGUE_AVG_PACE + 1.0) and team_weight > (opp_weight + 0.05):
            tempo_bonus = 0.006
            tempo_bonus += max(0.0, preferred_pace - 100.0) * 0.001
            tempo_bonus += max(0.0, team_weight - 0.58) * 0.02
            tempo_bonus = min(0.02, tempo_bonus)
            adj += tempo_bonus
            reasons.append(
                f"Tempo dictation fast-style edge +{tempo_bonus*100:.1f}% "
                f"({team_weight*100:.0f}/{opp_weight*100:.0f} control)"
            )
    else:
        pace_diff = team.team_stats.pace - opp_team.team_stats.pace
        if team.team_stats.pace > 100.0 and team.team_stats.off_rating_10 > 115.0 and opp_team.team_stats.def_rating_10 > 115.0:
            adj += 0.03
            reasons.append("Pace edge vs bottom-10 def +3.0%")
        elif pace_diff > 2.0:
            adj += 0.02
            reasons.append("Pace differential edge +2.0%")

    # Rebound Edge
    if (team.team_stats.reb_pct - opp_team.team_stats.reb_pct) > 0.03:
        adj += 0.02
        reasons.append("Rebound edge +2.0%")

    recent_form_edge = _get_recent_form_metric(
        team.team_stats,
        "weighted_point_diff",
        use_capped_form=use_capped_form,
    ) - _get_recent_form_metric(
        opp_team.team_stats,
        "weighted_point_diff",
        use_capped_form=use_capped_form,
    )
    if abs(recent_form_edge) >= 2.0:
        form_adj = max(-0.04, min(0.04, recent_form_edge * 0.004))
        adj += form_adj
        using_capped_form = use_capped_form and (
            hasattr(team.team_stats, "capped_weighted_point_diff")
            or hasattr(opp_team.team_stats, "capped_weighted_point_diff")
        )
        reason_label = "Recent point-diff edge (capped form)" if using_capped_form else "Recent point-diff edge"
        reasons.append(f"{reason_label} {form_adj*100:+.1f}%")

    # Mild home-court shooting execution bump in favorable offensive matchups.
    if team.is_home and team.team_stats.off_rating_10 >= opp_team.team_stats.def_rating_10:
        adj += 0.005
        reasons.append("Home-court shot-making +0.5%")
        
    reason_str = ", ".join(reasons) if reasons else "No matchup modifiers"
    return adj, reason_str


def combine_home_win_probability(
    layer_probability: float,
    predicted_spread: float,
    home_team: Team,
    away_team: Team,
) -> float:
    """
    Blend the probability layers with the independent spread estimate.

    The spread input is kept intentionally light here. Probability layers own
    the team-strength, home-court, and recent-form signal; spread is a
    cross-check from the separate matchup margin path, not a second full pass
    over the same inputs.
    """
    base_prob = _clamp_probability(layer_probability)
    blended_logit = _logit(base_prob) + (predicted_spread / 10.0) * 0.30
    return _sigmoid(blended_logit)


def legacy_extremize_probability(raw_prob: float, factor: float = 1.3) -> float:
    """
    Extremized prob = 50% + (Raw prob - 50%) * 1.3
    Cap at 95%, floor at 5%.
    """
    extremized = 0.50 + (raw_prob - 0.50) * factor
    return max(0.05, min(0.95, extremized))


def extremize_probability(raw_prob: float, factor: float = 1.3) -> float:
    """
    Extremize in log-odds space instead of hard clipping at 95%.

    The old linear `min(0.95, ...)` ceiling was wrong because very different
    game states were flattened into the same 95.0% output, which hid uncertainty
    instead of modeling it. Scaling the logit keeps the same "push away from
    50/50" intent while only approaching 0/1 asymptotically.
    """
    return _sigmoid(_logit(raw_prob) * factor)


def _team_offense_per_100(team_stats) -> float:
    pace = max(_coerce_float(getattr(team_stats, "pace", _LEAGUE_AVG_PACE), _LEAGUE_AVG_PACE), 1e-6)
    points_per_game = _coerce_float(
        getattr(team_stats, "points_per_game", getattr(team_stats, "off_rating_10", 112.0) * pace / 100.0),
        getattr(team_stats, "off_rating_10", 112.0) * pace / 100.0,
    )
    return (points_per_game / pace) * 100.0


def _team_defense_per_100(team_stats) -> float:
    pace = max(_coerce_float(getattr(team_stats, "pace", _LEAGUE_AVG_PACE), _LEAGUE_AVG_PACE), 1e-6)
    opp_points_per_game = _coerce_float(
        getattr(team_stats, "opp_points_per_game", getattr(team_stats, "def_rating_10", 112.0) * pace / 100.0),
        getattr(team_stats, "def_rating_10", 112.0) * pace / 100.0,
    )
    return (opp_points_per_game / pace) * 100.0


def _project_matchup_points(home_stats, away_stats, projected_pace: float) -> tuple[float, float, float, float]:
    home_offense_per_100 = _team_offense_per_100(home_stats)
    away_offense_per_100 = _team_offense_per_100(away_stats)
    home_defense_per_100 = _team_defense_per_100(home_stats)
    away_defense_per_100 = _team_defense_per_100(away_stats)

    home_expected_rating = (home_offense_per_100 + away_defense_per_100) / 2.0
    away_expected_rating = (away_offense_per_100 + home_defense_per_100) / 2.0

    home_expected_points = home_expected_rating * projected_pace / 100.0
    away_expected_points = away_expected_rating * projected_pace / 100.0
    return home_expected_points, away_expected_points, home_expected_rating, away_expected_rating


def _calculate_team_pace_control(team_stats, opp_stats, use_capped_form: bool = False) -> dict:
    opponent_turnover_pressure = getattr(team_stats, "opp_tov_pct", None)
    if opponent_turnover_pressure is None:
        opponent_turnover_pressure = _coerce_float(
            getattr(opp_stats, "tov_pct", _DEFAULT_OPP_TOV_PCT),
            _DEFAULT_OPP_TOV_PCT,
        )
    else:
        opponent_turnover_pressure = _coerce_float(opponent_turnover_pressure, _DEFAULT_OPP_TOV_PCT)
        opponent_ball_security = _coerce_float(
            getattr(opp_stats, "tov_pct", opponent_turnover_pressure),
            opponent_turnover_pressure,
        )
        opponent_turnover_pressure = (opponent_turnover_pressure * 0.75) + (opponent_ball_security * 0.25)

    dreb_pct = getattr(team_stats, "dreb_pct", None)
    if dreb_pct is None:
        dreb_pct = _clamp(
            _coerce_float(getattr(team_stats, "reb_pct", 0.50), 0.50) + 0.22,
            0.68,
            0.79,
        )
    else:
        dreb_pct = _coerce_float(dreb_pct, _DEFAULT_DREB_PCT)

    recent_form = _get_recent_form_metric(
        team_stats,
        "weighted_point_diff",
        use_capped_form=use_capped_form,
    )

    turnover_component = (opponent_turnover_pressure - _DEFAULT_OPP_TOV_PCT) * 125.0
    rebound_component = (dreb_pct - _DEFAULT_DREB_PCT) * 85.0
    recent_form_component = recent_form * 0.16
    control_score = turnover_component + rebound_component + recent_form_component

    return {
        "score": control_score,
        "turnover_pressure": opponent_turnover_pressure,
        "dreb_pct": dreb_pct,
        "recent_form": recent_form,
        "reason": (
            f"Force TOV {opponent_turnover_pressure*100:.1f}% | "
            f"DREB {dreb_pct*100:.1f}% | "
            f"Form {recent_form:+.1f}"
        ),
    }


def calculate_dictated_pace(away_stats, home_stats, use_capped_form: bool = False) -> tuple[float, dict]:
    """
    Skew the matchup pace toward the team most likely to impose its style.
    """
    away_profile = _calculate_team_pace_control(away_stats, home_stats, use_capped_form=use_capped_form)
    home_profile = _calculate_team_pace_control(home_stats, away_stats, use_capped_form=use_capped_form)

    home_pace = _coerce_float(getattr(home_stats, "pace", _LEAGUE_AVG_PACE), _LEAGUE_AVG_PACE)
    away_pace = _coerce_float(getattr(away_stats, "pace", _LEAGUE_AVG_PACE), _LEAGUE_AVG_PACE)
    neutral_pace = (home_pace + away_pace) / 2.0

    control_gap = home_profile["score"] - away_profile["score"]
    if abs(control_gap) < 0.15:
        home_weight = 0.50
    else:
        home_weight = _clamp(0.50 + (control_gap * 0.10), 0.20, 0.80)
    away_weight = 1.0 - home_weight
    dictated_pace = (away_pace * away_weight) + (home_pace * home_weight)

    if home_weight > away_weight + 0.02:
        dictating_side = "home"
    elif away_weight > home_weight + 0.02:
        dictating_side = "away"
    else:
        dictating_side = "neutral"

    context = {
        "dictated_pace": dictated_pace,
        "neutral_pace": neutral_pace,
        "home_weight": home_weight,
        "away_weight": away_weight,
        "control_gap": control_gap,
        "dictating_side": dictating_side,
        "home_control_score": home_profile["score"],
        "away_control_score": away_profile["score"],
        "home_turnover_pressure": home_profile["turnover_pressure"],
        "away_turnover_pressure": away_profile["turnover_pressure"],
        "home_dreb_pct": home_profile["dreb_pct"],
        "away_dreb_pct": away_profile["dreb_pct"],
        "home_recent_form": home_profile["recent_form"],
        "away_recent_form": away_profile["recent_form"],
        "home_reason": home_profile["reason"],
        "away_reason": away_profile["reason"],
        "recent_form_mode": "capped" if use_capped_form else "raw",
    }
    return dictated_pace, context


def predict_total_points(game_ctx: GameContext, pace_context: dict | None = None) -> float:
    """
    Project the total by blending each team's scoring with the opponent's defense.
    """
    home = game_ctx.home_team.team_stats
    away = game_ctx.away_team.team_stats

    if pace_context:
        projected_pace = _coerce_float(pace_context.get("dictated_pace", _LEAGUE_AVG_PACE), _LEAGUE_AVG_PACE)
        home_expected_pts, away_expected_pts, home_expected_rating, away_expected_rating = _project_matchup_points(
            home,
            away,
            projected_pace,
        )
        pace_context["home_expected_points"] = home_expected_pts
        pace_context["away_expected_points"] = away_expected_pts
        pace_context["home_expected_rating"] = home_expected_rating
        pace_context["away_expected_rating"] = away_expected_rating
        pace_context["projected_total"] = away_expected_pts + home_expected_pts
        return pace_context["projected_total"]

    away_points_per_game = getattr(away, "points_per_game", away.off_rating_10 * away.pace / 100.0)
    home_points_per_game = getattr(home, "points_per_game", home.off_rating_10 * home.pace / 100.0)
    home_opp_points_per_game = getattr(home, "opp_points_per_game", home.def_rating_10 * home.pace / 100.0)
    away_opp_points_per_game = getattr(away, "opp_points_per_game", away.def_rating_10 * away.pace / 100.0)

    away_expected_pts = (away_points_per_game + home_opp_points_per_game) / 2.0
    home_expected_pts = (home_points_per_game + away_opp_points_per_game) / 2.0
    return away_expected_pts + home_expected_pts


def legacy_predict_total_points(game_ctx: GameContext) -> float:
    """
    Legacy total model retained for side-by-side dashboard comparisons.
    """
    base_points = 225.0

    h_pace = game_ctx.home_team.team_stats.pace
    a_pace = game_ctx.away_team.team_stats.pace
    pace_adj = ((h_pace - 99.0) + (a_pace - 99.0)) * 2.0

    h_def = game_ctx.home_team.team_stats.def_rating_10
    a_def = game_ctx.away_team.team_stats.def_rating_10
    def_adj = ((h_def - 114.0) + (a_def - 114.0)) * 0.8

    return base_points + pace_adj + def_adj


def legacy_predict_spread(home_prob: float) -> float:
    """
    Spread: (Extremized Prob - 0.50) * 30
    [Rule of thumb: 10% prob edge = ~3 point spread]
    """
    prob_diff = home_prob - 0.50
    return prob_diff * 30.0


def _get_team_efficiency_pct(team_stats) -> float:
    efg_pct = getattr(team_stats, "efg_pct", None)
    if efg_pct is not None:
        return efg_pct
    return getattr(team_stats, "ts_pct", 0.5)


def _is_altitude_home_team(team: Team) -> bool:
    normalized_name = str(team.name).upper()
    return any(marker in normalized_name for marker in ("DENVER", "NUGGETS", "DEN", "UTAH", "JAZZ", "UTA"))


def predict_spread(home_team: Team, away_team: Team, pace_context: dict | None = None) -> float:
    """
    Predict the expected home scoring margin directly from core per-game stats.
    """
    home = home_team.team_stats
    away = away_team.team_stats

    home_points_per_game = getattr(home, "points_per_game", home.off_rating_10 * home.pace / 100.0)
    home_opp_points_per_game = getattr(home, "opp_points_per_game", home.def_rating_10 * home.pace / 100.0)
    away_points_per_game = getattr(away, "points_per_game", away.off_rating_10 * away.pace / 100.0)
    away_opp_points_per_game = getattr(away, "opp_points_per_game", away.def_rating_10 * away.pace / 100.0)

    if pace_context:
        projected_pace = _coerce_float(pace_context.get("dictated_pace", _LEAGUE_AVG_PACE), _LEAGUE_AVG_PACE)
        home_expected_pts, away_expected_pts, home_expected_rating, away_expected_rating = _project_matchup_points(
            home,
            away,
            projected_pace,
        )
        pace_context["home_expected_points"] = home_expected_pts
        pace_context["away_expected_points"] = away_expected_pts
        pace_context["home_expected_rating"] = home_expected_rating
        pace_context["away_expected_rating"] = away_expected_rating
        pace_context["pace_based_spread"] = home_expected_pts - away_expected_pts
        base_spread = pace_context["pace_based_spread"]
    else:
        # Keep the published spread aligned with the rest of NBANEW: positive means
        # the home team is favored by that many points.
        base_spread = (home_points_per_game - home_opp_points_per_game) - (
            away_points_per_game - away_opp_points_per_game
        )
    altitude_home = _is_altitude_home_team(home_team)
    base_spread += 5.0 if altitude_home else 3.5

    home_efg = _get_team_efficiency_pct(home)
    away_efg = _get_team_efficiency_pct(away)
    home_reb = getattr(home, "reb_pct", 0.5)
    away_reb = getattr(away, "reb_pct", 0.5)
    home_tov = getattr(home, "tov_pct", 0.13)
    away_tov = getattr(away, "tov_pct", 0.13)

    efficiency_diff = (home_efg - away_efg) * 20.0
    reb_diff = (home_reb - away_reb) * 10.0
    turnover_diff = (away_tov - home_tov) * 12.0
    four_factors_adj = efficiency_diff + reb_diff + turnover_diff

    away_form = away.last_10_win_pct - 0.5
    home_form = home.last_10_win_pct - 0.5
    form_adj = (home_form - away_form) * 5.0
    if home_form > 0 and (home_points_per_game - home_opp_points_per_game) < 0:
        form_adj -= home_form * 2.5
    if away_form > 0 and (away_points_per_game - away_opp_points_per_game) < 0:
        form_adj += away_form * 2.5

    rest_adj = 0.0
    if getattr(away, "is_b2b", getattr(away, "back_to_back_flag", False)):
        rest_adj += 3.5
        if altitude_home:
            rest_adj += 2.0
    if getattr(home, "is_b2b", getattr(home, "back_to_back_flag", False)):
        rest_adj -= 2.0

    home_rest_days = getattr(home, "rest_days", 0)
    away_rest_days = getattr(away, "rest_days", 0)
    if home_rest_days >= 2 and away_rest_days == 0:
        rest_adj += 1.5
    elif away_rest_days >= 2 and home_rest_days == 0:
        rest_adj -= 1.5

    projected_margin = base_spread + four_factors_adj + form_adj + rest_adj
    projected_margin = max(-22.0, min(22.0, projected_margin))
    return projected_margin

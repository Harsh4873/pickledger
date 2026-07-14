from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "NBAPredictionModel"):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)


def _series_3_down_0_2():
    return {
        "round": "Conference Semifinals",
        "headline": "West Semifinals - Game 3",
        "game_number": 3,
        "is_game_1": False,
        "is_game_2": False,
        "is_game_7": False,
        "home_wins": 0,
        "away_wins": 2,
        "home_trailing": True,
        "away_trailing": False,
        "home_elimination": False,
        "away_elimination": False,
        "home_closeout": False,
        "away_closeout": False,
        "repeat_matchups": 2,
    }


def _stub_market(home_spread: float, away_spread: float):
    return {
        "home_spread": home_spread,
        "away_spread": away_spread,
        "provider": "test",
    }


def test_big_dog_with_huge_edge_is_not_a_bet():
    """Lakers +295 vs OKC scenario from 2026-05-09: 27.8% edge on a +295 dog
    must not be a BET — it should be PASS or LEAN at most because the dog
    needs >=60% conviction and the spread layer disagrees with the market."""
    from NBAPlayoffsPredictionModel.run_live import evaluate_playoff_decision

    result = evaluate_playoff_decision(
        pick_team="Lakers",
        pick_prob=0.5208,
        pick_odds=295,
        edge=0.278,
        predicted_spread=0.15,
        market=_stub_market(home_spread=8.5, away_spread=-8.5),
        home_name="Lakers",
        away_name="Thunder",
        injuries={"placeholder": []},
        series_context=_series_3_down_0_2(),
        adjustments=[{"value": 0.04}, {"value": -0.04}],
    )
    assert result["decision"] != "BET"
    reasons = " | ".join(result["reasons"]).lower()
    assert "ceiling" in reasons or "conviction" in reasons or "narrative" in reasons


def test_short_favorite_with_real_edge_can_bet():
    """A short home favorite with a 6+% edge, 60%+ pick prob, and a model
    spread that broadly agrees with the market line should fire BET."""
    from NBAPlayoffsPredictionModel.run_live import evaluate_playoff_decision

    series_neutral = {
        "round": "Conference Semifinals",
        "headline": "East Semifinals - Game 5",
        "game_number": 5,
        "is_game_1": False,
        "is_game_2": False,
        "is_game_7": False,
        "home_wins": 2,
        "away_wins": 2,
        "home_trailing": False,
        "away_trailing": False,
        "home_elimination": False,
        "away_elimination": False,
        "home_closeout": False,
        "away_closeout": False,
        "repeat_matchups": 4,
    }

    result = evaluate_playoff_decision(
        pick_team="Celtics",
        pick_prob=0.62,
        pick_odds=-180,
        edge=0.07,
        predicted_spread=4.6,
        market=_stub_market(home_spread=-4.5, away_spread=4.5),
        home_name="Celtics",
        away_name="Heat",
        injuries={"placeholder": []},
        series_context=series_neutral,
        adjustments=[{"value": 0.03}, {"value": 0.02}],
    )
    assert result["decision"] == "BET"
    assert result["confidence"] in {"High", "Medium"}


def test_spread_disagreement_blocks_bet():
    """If the model margin disagrees with the market line by >5 pts the
    pick must drop to LEAN/PASS even if the moneyline edge looks fine."""
    from NBAPlayoffsPredictionModel.run_live import evaluate_playoff_decision

    series_state = {
        "round": "Conference Finals",
        "headline": "East Finals - Game 4",
        "game_number": 4,
        "is_game_1": False,
        "is_game_2": False,
        "is_game_7": False,
        "home_wins": 1,
        "away_wins": 2,
        "home_trailing": True,
        "away_trailing": False,
        "home_elimination": False,
        "away_elimination": False,
        "home_closeout": False,
        "away_closeout": False,
        "repeat_matchups": 3,
    }

    result = evaluate_playoff_decision(
        pick_team="Pacers",
        pick_prob=0.56,
        pick_odds=-130,
        edge=0.06,
        predicted_spread=-4.0,  # model says home loses by 4
        market=_stub_market(home_spread=-3.5, away_spread=3.5),  # market says home -3.5
        home_name="Pacers",
        away_name="Knicks",
        injuries={"placeholder": []},
        series_context=series_state,
        adjustments=[{"value": 0.02}, {"value": 0.02}],
    )
    assert result["decision"] != "BET"


def test_series_form_signal_with_two_blowout_losses():
    """Lakers were down 0-2 to OKC by 18 in each game. The series-form
    signal should drag the home team's implied probability well below 50%
    and shift the margin estimate strongly in OKC's favor."""
    from NBAPlayoffsPredictionModel.run_live import (
        compute_series_form_signal,
        PLAYOFF_MARGIN_RMSE,
    )

    history = [
        {"date": "2026-05-05", "is_home_for_target": False, "margin_for_target": -18.0},
        {"date": "2026-05-07", "is_home_for_target": False, "margin_for_target": -18.0},
    ]
    signal = compute_series_form_signal(history, PLAYOFF_MARGIN_RMSE)

    assert signal["games"] == 2
    assert signal["avg_margin"] == -18.0
    assert signal["implied_prob_for_home"] is not None
    # Two -18 results, with predictive variance widened by sqrt(1.5) for the
    # 2-game sample, imply a Game 3 home win prob in the 8-15% range.
    assert 0.05 < signal["implied_prob_for_home"] < 0.16
    # Evidence weight scales with sqrt(games); 2 games ≈ 0.226.
    assert 0.20 < signal["evidence_weight"] < 0.27
    # Margin shift uses 45% of avg margin = -8.1.
    assert -8.5 < signal["margin_shift"] < -7.5
    # RMSE inflation kicks in once any individual game margin exceeds 12.
    assert signal["rmse_inflation"] > 0.0


def test_series_form_no_history_returns_zero_signal():
    from NBAPlayoffsPredictionModel.run_live import (
        compute_series_form_signal,
        PLAYOFF_MARGIN_RMSE,
    )

    signal = compute_series_form_signal([], PLAYOFF_MARGIN_RMSE)
    assert signal["games"] == 0
    assert signal["evidence_weight"] == 0.0
    assert signal["implied_prob_for_home"] is None
    assert signal["margin_shift"] == 0.0
    assert signal["rmse_inflation"] == 0.0


def test_series_form_pulls_lakers_base_rate_down():
    """Re-running the calculate_base_rate for the Lakers Game 3 scenario
    with series form should produce a much lower probability than the
    pre-patch base rate, demonstrating the model is now learning from
    the prior series games."""
    from NBAPlayoffsPredictionModel.run_live import (
        calculate_base_rate,
        compute_series_form_signal,
        PLAYOFF_MARGIN_RMSE,
    )

    home_stats = {
        "win_pct": 0.55,
        "recent_10_win_pct": 0.55,
    }
    last20_context = {"Lakers": {"last20_win_pct": 0.55, "last20_point_diff": 1.0}}
    ranks = {"Lakers": 8, "Thunder": 1}
    h2h = {"home_win_pct": 0.40, "games": 4, "point_diff": -6.0, "note": "Lakers 1-3 vs OKC RS"}

    base_no_series, _ = calculate_base_rate(
        "Lakers", "Thunder", home_stats, last20_context, ranks, h2h, None
    )

    series_form = compute_series_form_signal(
        [
            {"date": "2026-05-05", "is_home_for_target": False, "margin_for_target": -18.0},
            {"date": "2026-05-07", "is_home_for_target": False, "margin_for_target": -18.0},
        ],
        PLAYOFF_MARGIN_RMSE,
    )
    base_with_series, notes = calculate_base_rate(
        "Lakers", "Thunder", home_stats, last20_context, ranks, h2h, series_form
    )

    # Series form should pull Lakers' base rate well below the season-only rate.
    assert base_with_series < base_no_series - 0.05
    assert any("series-form blend" in note for note in notes)


def test_espn_fallback_lookback_window_covers_finals():
    """The ESPN scoreboard fallback should scan ≥14 days back so a Finals
    Game 1 → Game 7 stretch (which can span 17 days) doesn't lose history."""
    from NBAPlayoffsPredictionModel.run_live import (
        SERIES_HISTORY_ESPN_LOOKBACK_DAYS,
        SERIES_HISTORY_MAX_GAMES,
    )

    assert SERIES_HISTORY_ESPN_LOOKBACK_DAYS >= 14
    assert SERIES_HISTORY_MAX_GAMES == 6   # max prior games before Game 7


def test_nba_team_stats_fall_back_to_espn_when_nba_api_times_out(monkeypatch):
    import live_data

    class BrokenTeamStats:
        def __init__(self, **_kwargs):
            raise TimeoutError("stats.nba.com timed out")

    fallback = {
        "Spurs": {"stats_source": "ESPN team statistics fallback"},
        "Knicks": {"stats_source": "ESPN team statistics fallback"},
    }
    monkeypatch.setattr(live_data.leaguedashteamstats, "LeagueDashTeamStats", BrokenTeamStats)
    monkeypatch.setattr(live_data, "fetch_espn_team_stats_fallback", lambda _games: fallback)

    result = live_data.fetch_all_team_stats(
        upcoming_games=[{"away_team": "Spurs", "home_team": "Knicks"}],
    )

    assert result == fallback


def test_nba_roster_falls_back_to_espn_when_nba_api_times_out(monkeypatch):
    import live_data

    class BrokenRoster:
        def __init__(self, **_kwargs):
            raise TimeoutError("stats.nba.com timed out")

    fallback = [{"name": "Test Player", "source": "ESPN roster fallback"}]
    monkeypatch.setattr(live_data.commonteamroster, "CommonTeamRoster", BrokenRoster)
    monkeypatch.setattr(live_data, "fetch_espn_roster_fallback", lambda _team: fallback)

    result = live_data.fetch_roster("Spurs")

    assert result == fallback


def test_nba_playoffs_pick_payload_records_team_stats_source():
    source = (REPO_ROOT / "NBAPlayoffsPredictionModel" / "run_live.py").read_text(encoding="utf-8")

    assert '"stats_source": stats_source' in source
    assert "Team efficiency and recent-form stats fetched from {stats_source}" in source
    assert "Missing NBA API team stats" not in source


def test_missing_injury_feed_caps_at_lean():
    """An empty injury feed leaves the model running blind on availability;
    even a good-looking edge should drop to LEAN."""
    from NBAPlayoffsPredictionModel.run_live import evaluate_playoff_decision

    series_state = {
        "round": "First Round",
        "headline": "East First Round - Game 2",
        "game_number": 2,
        "is_game_1": False,
        "is_game_2": True,
        "is_game_7": False,
        "home_wins": 0,
        "away_wins": 1,
        "home_trailing": True,
        "away_trailing": False,
        "home_elimination": False,
        "away_elimination": False,
        "home_closeout": False,
        "away_closeout": False,
        "repeat_matchups": 1,
    }

    result = evaluate_playoff_decision(
        pick_team="Knicks",
        pick_prob=0.58,
        pick_odds=-150,
        edge=0.05,
        predicted_spread=3.0,
        market=_stub_market(home_spread=-3.0, away_spread=3.0),
        home_name="Knicks",
        away_name="Pistons",
        injuries={},  # empty feed
        series_context=series_state,
        adjustments=[{"value": 0.02}],
    )
    assert result["decision"] != "BET"
    assert any("injury" in reason.lower() for reason in result["reasons"])

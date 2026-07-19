from __future__ import annotations


def test_wnba_context_only_edge_is_passed():
    from WNBAPredictionModel.wnba_picks import assess_spread_edge
    from WNBAPredictionModel.wnba_probability_layers import calculate_wnba_matchup

    partial_home = {
        "eFG_pct": 0.56,
        "TOV_pct": 0.12,
        "FTR": 0.30,
    }
    partial_away = {
        "eFG_pct": 0.42,
        "TOV_pct": 0.18,
        "FTR": 0.19,
    }
    context = {
        "home_rest_days": 7,
        "away_rest_days": 1,
        "away_is_b2b": True,
        "home_injury_penalty": 0.0,
        "away_injury_penalty": 0.45,
    }

    result = calculate_wnba_matchup("WAS", "NY", partial_home, partial_away, context)
    guardrail = assess_spread_edge(result, partial_home, partial_away, context)

    assert result["data_quality"] == "partial"
    assert guardrail["decision"] == "PASS"
    assert "no two-team NRtg baseline" in guardrail["reasons"]


def test_wnba_full_baseline_can_emit_bet():
    from WNBAPredictionModel.wnba_picks import assess_spread_edge
    from WNBAPredictionModel.wnba_probability_layers import calculate_wnba_matchup

    # Realistic 2026 paces (league runs ~75-82); pace now compounds
    # (home + away - league_avg), so unrealistically low fixture paces
    # artificially shrink the projected margin.
    home = {"NRtg": 8.0, "ORtg": 108.0, "DRtg": 100.0, "Pace": 80.0, "W": 8, "L": 3}
    away = {"NRtg": -2.0, "ORtg": 101.0, "DRtg": 103.0, "Pace": 79.0, "W": 4, "L": 7}
    context = {
        "home_rest_days": 3,
        "away_rest_days": 1,
        "away_is_b2b": False,
        "home_injury_penalty": 0.0,
        "away_injury_penalty": 0.1,
    }

    result = calculate_wnba_matchup("IND", "MIN", home, away, context)
    guardrail = assess_spread_edge(result, home, away, context)

    assert result["data_quality"] == "full"
    assert guardrail["decision"] == "BET"
    assert guardrail["confidence_label"] == "High"


def test_pace_compounds_instead_of_averaging():
    """Two fast teams play faster than either's average; two grinders play
    slower — a weighted average shaved both tails off every total."""
    from WNBAPredictionModel.wnba_probability_layers import blend_pace, WNBA_LEAGUE_AVG_PACE

    fast = blend_pace(81.5, 81.0)
    slow = blend_pace(76.0, 77.0)
    assert fast == 81.5 + 81.0 - WNBA_LEAGUE_AVG_PACE > 81.5
    assert slow == 76.0 + 77.0 - WNBA_LEAGUE_AVG_PACE < 76.0
    # One-sided and missing inputs keep their old behavior.
    assert blend_pace(None, None) == WNBA_LEAGUE_AVG_PACE
    assert blend_pace(78.0, None) == 78.0


def test_away_b2b_shaves_projected_total():
    """Second night of a back-to-back costs shooting efficiency, so the
    projected total drops when the away team is on a B2B."""
    from WNBAPredictionModel.wnba_probability_layers import calculate_wnba_matchup

    home = {"NRtg": 2.0, "ORtg": 104.0, "DRtg": 102.0, "Pace": 80.0, "W": 10, "L": 8}
    away = {"NRtg": 1.0, "ORtg": 103.0, "DRtg": 102.0, "Pace": 79.5, "W": 9, "L": 9}
    rested = calculate_wnba_matchup("IND", "MIN", home, away, {"away_is_b2b": False})
    fatigued = calculate_wnba_matchup("IND", "MIN", home, away, {"away_is_b2b": True})
    assert fatigued["projected_total"] < rested["projected_total"]


def test_tightened_spread_and_total_gates():
    """2026-07-19 tightening: spread ran 10-20 and totals 22-37 at the old
    gates, so both markets now demand materially larger disagreement."""
    from WNBAPredictionModel.wnba_picks import (
        WNBA_SPREAD_BET_EDGE,
        WNBA_SPREAD_LEAN_EDGE,
        WNBA_SPREAD_BET_COVER,
        WNBA_SPREAD_LEAN_COVER,
        WNBA_TOTAL_LEAN_EDGE,
        WNBA_TOTAL_MIN_GAP,
    )

    assert WNBA_SPREAD_BET_EDGE >= 0.06
    assert WNBA_SPREAD_LEAN_EDGE >= 0.045
    assert WNBA_SPREAD_BET_COVER >= 3.5
    assert WNBA_SPREAD_LEAN_COVER >= 2.5
    assert WNBA_TOTAL_LEAN_EDGE >= 0.045
    assert WNBA_TOTAL_MIN_GAP >= 7.0


def test_wnba_away_favorite_confidence_uses_favorite_side():
    from WNBAPredictionModel.wnba_picks import get_confidence_label

    assert get_confidence_label(0.25) == "High"
    assert get_confidence_label(0.36) == "Medium"


def test_wnba_h2h_signal_with_two_blowout_wins():
    """Two prior H2H games where the home team won by 14 each should
    produce a positive H2H margin shift (capped) and non-zero evidence
    weight that nudges the predicted margin up without dominating it."""
    from WNBAPredictionModel.wnba_probability_layers import (
        compute_h2h_signal,
        WNBA_H2H_ADJ_CAP,
    )

    games = [
        {"date": "2026-05-20", "is_home_for_target": True, "margin_for_target": 14.0},
        {"date": "2026-06-04", "is_home_for_target": False, "margin_for_target": 14.0},
    ]
    signal = compute_h2h_signal(games)

    assert signal["games"] == 2
    assert signal["avg_margin"] == 14.0
    # 14 * 0.40 = 5.6, but capped at WNBA_H2H_ADJ_CAP (3.5).
    assert 0.0 < signal["margin_shift"] <= WNBA_H2H_ADJ_CAP
    # Evidence weight scales with sqrt(games); 2 games -> ~0.198.
    assert 0.15 < signal["evidence_weight"] < 0.25


def test_wnba_h2h_signal_empty_returns_no_shift():
    from WNBAPredictionModel.wnba_probability_layers import compute_h2h_signal

    signal = compute_h2h_signal([])
    assert signal["games"] == 0
    assert signal["margin_shift"] == 0.0
    assert signal["evidence_weight"] == 0.0


def test_wnba_units_scale_with_conviction():
    """Higher projected margins and stronger probabilities should produce
    materially larger stake recommendations than borderline picks."""
    from WNBAPredictionModel.wnba_picks import assess_spread_edge

    home = {"NRtg": 8.0, "ORtg": 108.0, "DRtg": 100.0, "Pace": 70.0, "W": 8, "L": 3}
    away = {"NRtg": -2.0, "ORtg": 101.0, "DRtg": 103.0, "Pace": 69.0, "W": 4, "L": 7}
    base_ctx = {
        "home_rest_days": 3,
        "away_rest_days": 1,
        "away_is_b2b": False,
        "home_injury_penalty": 0.0,
        "away_injury_penalty": 0.1,
    }

    big = assess_spread_edge(
        {"adjusted_margin": 11.0, "win_prob": 0.78, "projected_total": 162.0,
         "h2h_signal": {"games": 2}},
        home, away, base_ctx,
    )
    small = assess_spread_edge(
        {"adjusted_margin": 5.0, "win_prob": 0.66, "projected_total": 162.0,
         "h2h_signal": {"games": 0}},
        home, away, base_ctx,
    )
    pass_pick = assess_spread_edge(
        {"adjusted_margin": 1.0, "win_prob": 0.54, "projected_total": 162.0,
         "h2h_signal": {"games": 0}},
        home, away, base_ctx,
    )

    assert big["decision"] == "BET"
    assert small["decision"] == "LEAN"
    assert pass_pick["decision"] == "PASS"
    assert big["units"] > small["units"] > 0.0
    assert pass_pick["units"] == 0.0
    # Stakes stay inside the [0.25, 1.75] envelope.
    assert 0.25 <= big["units"] <= 1.75
    assert 0.25 <= small["units"] <= 1.75


def test_wnba_total_falls_back_to_ppg_when_ortg_missing():
    """When ORtg is unavailable but rolling_pts / pts_per_game exist, the
    projected total should still be emitted instead of None."""
    from WNBAPredictionModel.wnba_probability_layers import compute_projected_total

    home = {"Pace": 72.0, "rolling_pts": 84.0}
    away = {"Pace": 70.0, "pts_per_game": 78.5}
    total = compute_projected_total(home, away)
    assert total is not None
    assert 130.0 <= total <= 185.0


def test_wnba_total_injury_adjustment_is_bounded():
    from WNBAPredictionModel.wnba_probability_layers import compute_projected_total

    home = {"ORtg": 108.0, "Pace": 80.0}
    away = {"ORtg": 104.0, "Pace": 80.0}
    baseline = compute_projected_total(home, away)
    injured = compute_projected_total(
        home,
        away,
        home_injury_penalty=1.0,
        away_injury_penalty=1.0,
    )

    assert baseline is not None and injured is not None
    assert baseline - injured == 8.0


def test_wnba_market_vig_removal_and_kelly():
    """Vig-removed two-sided ML should sum to 1.0; Kelly stake scales
    with edge and decimal odds."""
    from WNBAPredictionModel.wnba_market import (
        american_to_implied,
        remove_vig,
        quarter_kelly_units,
    )

    # -120 / +110 → raw 0.5455 + 0.4762 = 1.0217; vig-removed sums to 1.0.
    h, a = remove_vig(-120, 110)
    assert abs((h + a) - 1.0) < 1e-9
    assert h > a  # favorite > dog

    # -110 ≈ 52.4% raw implied
    assert abs(american_to_implied(-110) - 0.5238) < 1e-3
    # +200 ≈ 33.3%
    assert abs(american_to_implied(200) - 0.3333) < 1e-3

    # Quarter-Kelly: 5% edge at +100 (b=1) → 0.05/1/4 = 0.0125u → rounds to 0.01
    assert quarter_kelly_units(0.05, 100) == 0.01
    # 10% edge at -110 (b≈0.909) → 0.10/0.909/4 ≈ 0.0275 → rounds to 0.03
    units = quarter_kelly_units(0.10, -110)
    assert 0.02 < units < 0.05
    # Negative edge always returns 0u.
    assert quarter_kelly_units(-0.05, -110) == 0.0


def test_wnba_compute_edge_uses_market_prob_for_pick_side():
    """compute_edge_units should hand back the market price for the picked
    side and the difference vs the model probability for that side."""
    from WNBAPredictionModel.wnba_market import (
        EdgeAssessment,
        MarketOdds,
        compute_edge_units,
    )

    market = MarketOdds(
        home_team_nickname="Mystics",
        away_team_nickname="Liberty",
        home_ml=140,
        away_ml=-160,
        spread_home=3.5,
        spread_away=-3.5,
        total_line=158.5,
        fetched_at="2026-06-15T18:00:00Z",
    )
    # Model thinks home is 50%; market says home is ~42% (vig-removed).
    # Edge for home pick = +0.08
    home = compute_edge_units(True, 0.50, market)
    assert home.market_pick_odds == 140
    assert home.market_pick_prob is not None and home.market_pick_prob < 0.50
    assert home.edge is not None and home.edge > 0.05
    assert home.kelly_units is not None and home.kelly_units > 0

    # Picking the away team at 50% model prob — market says away ~58%, so edge
    # is negative, Kelly units = 0.
    away = compute_edge_units(False, 0.50, market)
    assert away.edge is not None and away.edge < 0
    assert away.kelly_units == 0.0


def test_wnba_espn_market_normalizes_favorite_spread_sign(monkeypatch):
    from WNBAPredictionModel import wnba_market

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "events": [{
                    "competitions": [{
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "IND"}},
                            {"homeAway": "away", "team": {"abbreviation": "CHI"}},
                        ],
                        "odds": [{
                            "spread": 10.5,
                            "overUnder": 172.5,
                            "homeTeamOdds": {
                                "favorite": True,
                                "moneyLine": -500,
                                "spreadOdds": -108,
                            },
                            "awayTeamOdds": {
                                "favorite": False,
                                "moneyLine": 360,
                                "spreadOdds": -112,
                            },
                        }],
                    }],
                }],
            }

    monkeypatch.setattr(wnba_market.requests, "get", lambda *args, **kwargs: Response())

    market = wnba_market._lookup_espn_market_odds("IND", "CHI", "2026-06-11")

    assert market is not None
    assert market.spread_home == -10.5
    assert market.spread_away == 10.5
    assert market.spread_odds == -108
    assert market.total_line == 172.5
    assert market.total_odds == -110


def test_wnba_spread_market_selects_cover_side_without_changing_moneyline():
    from WNBAPredictionModel.wnba_market import MarketOdds
    from WNBAPredictionModel.wnba_picks import assess_wnba_spread_market

    market = MarketOdds(
        home_team_nickname="Wings",
        away_team_nickname="Mercury",
        home_ml=-150,
        away_ml=130,
        spread_home=-8.0,
        spread_away=8.0,
        total_line=165.5,
        fetched_at="2026-06-11T12:00:00Z",
        spread_odds=-110,
        total_odds=-110,
    )
    stats = {"NRtg": 4.0, "W": 8, "L": 4}

    spread = assess_wnba_spread_market(
        {"adjusted_margin": 2.0},
        market,
        stats,
        stats,
        {},
    )

    assert spread["decision"] == "BET"
    assert spread["pick_team_is_home"] is False
    assert spread["market_line"] == 8.0
    assert spread["model_team_margin"] == -2.0
    assert spread["cover_margin"] == 6.0
    assert spread["units"] > 0.0


def test_wnba_total_market_uses_stricter_gap_and_sample_guardrails():
    from WNBAPredictionModel.wnba_market import MarketOdds
    from WNBAPredictionModel.wnba_picks import assess_wnba_total_market

    market = MarketOdds(
        home_team_nickname="Wings",
        away_team_nickname="Mercury",
        home_ml=-150,
        away_ml=130,
        spread_home=-3.5,
        spread_away=3.5,
        total_line=165.5,
        fetched_at="2026-06-11T12:00:00Z",
        total_odds=-110,
    )
    established = {"NRtg": 4.0, "W": 8, "L": 4}
    thin = {"NRtg": 4.0, "W": 1, "L": 1}

    bet = assess_wnba_total_market(
        {"projected_total": 155.0},
        market,
        established,
        established,
        {},
    )
    passed = assess_wnba_total_market(
        {"projected_total": 155.0},
        market,
        thin,
        thin,
        {},
    )

    assert bet["direction"] == "Under"
    assert bet["decision"] == "LEAN"
    assert bet["units"] > 0.0
    assert any("capped at LEAN" in reason for reason in bet["reasons"])
    assert passed["decision"] == "PASS"
    assert passed["units"] == 0.0


def test_wnba_decision_uses_real_edge_when_market_present():
    """When SportsLine odds are available, BET requires real 3% market
    edge; without them, the older internal-only thresholds apply."""
    from WNBAPredictionModel.wnba_market import EdgeAssessment
    from WNBAPredictionModel.wnba_picks import assess_spread_edge

    home = {"NRtg": 8.0, "ORtg": 108.0, "DRtg": 100.0, "Pace": 70.0, "W": 8, "L": 3}
    away = {"NRtg": -2.0, "ORtg": 101.0, "DRtg": 103.0, "Pace": 69.0, "W": 4, "L": 7}
    base_ctx = {
        "home_rest_days": 3, "away_rest_days": 1, "away_is_b2b": False,
        "home_injury_penalty": 0.0, "away_injury_penalty": 0.0,
    }
    base_result = {
        "adjusted_margin": 7.0, "win_prob": 0.71, "projected_total": 162.0,
        "h2h_signal": {"games": 1},
    }

    bet_market = EdgeAssessment(market_pick_odds=-180, market_pick_prob=0.62, edge=0.07, kelly_units=0.40)
    pass_market = EdgeAssessment(market_pick_odds=-260, market_pick_prob=0.72, edge=-0.01, kelly_units=0.0)

    bet = assess_spread_edge(base_result, home, away, base_ctx, market_edge=bet_market)
    pass_pick = assess_spread_edge(base_result, home, away, base_ctx, market_edge=pass_market)

    assert bet["decision"] == "BET"
    assert bet["has_market_price"] is True
    assert bet["market_pick_odds"] == -180
    assert bet["units"] > 0.0

    # Same model output, but now market disagrees — should not be a BET
    assert pass_pick["decision"] == "PASS"
    assert pass_pick["units"] == 0.0


def test_wnba_lineup_quality_downgrades_bet_when_starter_out():
    """A BET with 1 starter Out should drop to LEAN; 2+ Out should drop
    to PASS even if the model edge is large."""
    from WNBAPredictionModel.wnba_lineup_quality import LineupQuality
    from WNBAPredictionModel.wnba_market import EdgeAssessment
    from WNBAPredictionModel.wnba_picks import assess_spread_edge

    home = {"NRtg": 8.0, "ORtg": 108.0, "DRtg": 100.0, "Pace": 70.0, "W": 8, "L": 3}
    away = {"NRtg": -2.0, "ORtg": 101.0, "DRtg": 103.0, "Pace": 69.0, "W": 4, "L": 7}
    base_ctx = {
        "home_rest_days": 3, "away_rest_days": 1, "away_is_b2b": False,
        "home_injury_penalty": 0.0, "away_injury_penalty": 0.0,
    }
    big_edge_market = EdgeAssessment(market_pick_odds=-150, market_pick_prob=0.60, edge=0.08, kelly_units=0.50)
    base_result = {
        "adjusted_margin": 8.0, "win_prob": 0.72, "projected_total": 162.0,
        "h2h_signal": {"games": 1},
    }

    one_out = LineupQuality(
        starters_total=5, starters_healthy=4,
        starters_questionable=[],
        starters_out=["Star Player"],
        minutes_restriction_penalty=0.0,
        lineup_uncertainty_penalty=0.06,
    )
    two_out = LineupQuality(
        starters_total=5, starters_healthy=3,
        starters_questionable=[],
        starters_out=["Star A", "Star B"],
        minutes_restriction_penalty=0.0,
        lineup_uncertainty_penalty=0.12,
    )

    one = assess_spread_edge(base_result, home, away, base_ctx, market_edge=big_edge_market, pick_team_lineup=one_out)
    two = assess_spread_edge(base_result, home, away, base_ctx, market_edge=big_edge_market, pick_team_lineup=two_out)

    # The big-edge BET should drop to LEAN with one star out.
    assert one["decision"] == "LEAN"
    assert any("OUT" in r for r in one["reasons"])
    # And to PASS with two starters out.
    assert two["decision"] == "PASS"
    assert two["units"] == 0.0


def test_wnba_lineup_quality_module_classifies_questionable_vs_out():
    """get_lineup_quality should tally Out vs Questionable from the injury
    report and produce both a minutes-restriction and uncertainty penalty."""
    from WNBAPredictionModel.wnba_lineup_quality import get_lineup_quality

    # Build a synthetic injury report keyed by normalized name (the way
    # wnba_injuries normalizes them).
    report = {
        "caitlin clark": {
            "team_abbr": "IND",
            "status": "Out",
            "player_name": "Caitlin Clark",
        },
    }
    quality = get_lineup_quality("IND", report)

    assert quality.starters_total >= 1
    assert "Caitlin Clark" in quality.starters_out
    assert quality.lineup_uncertainty_penalty > 0.0

    # Questionable star bumps minutes-restriction penalty, not lineup
    # uncertainty penalty.
    report_q = {
        "caitlin clark": {
            "team_abbr": "IND",
            "status": "Questionable",
            "player_name": "Caitlin Clark",
        },
    }
    quality_q = get_lineup_quality("IND", report_q)
    assert "Caitlin Clark" in quality_q.starters_questionable
    assert quality_q.minutes_restriction_penalty > 0.0
    assert quality_q.lineup_uncertainty_penalty == 0.0


def test_wnba_h2h_lifts_predicted_margin():
    """End-to-end: passing two blowout wins as h2h_games shifts the
    adjusted margin and win prob in the home team's favor compared to
    the same matchup with no H2H signal."""
    from WNBAPredictionModel.wnba_probability_layers import calculate_wnba_matchup

    home = {"NRtg": 1.0, "ORtg": 102.0, "DRtg": 101.0, "Pace": 70.0, "W": 5, "L": 5}
    away = {"NRtg": 0.0, "ORtg": 101.5, "DRtg": 101.5, "Pace": 70.0, "W": 5, "L": 5}
    base_ctx = {
        "home_rest_days": 2,
        "away_rest_days": 2,
        "away_is_b2b": False,
        "home_injury_penalty": 0.0,
        "away_injury_penalty": 0.0,
    }

    no_h2h = calculate_wnba_matchup("HOM", "AWY", home, away, base_ctx)
    with_h2h = calculate_wnba_matchup(
        "HOM",
        "AWY",
        home,
        away,
        {**base_ctx, "h2h_games": [
            {"date": "2026-05-20", "is_home_for_target": True, "margin_for_target": 12.0},
            {"date": "2026-06-04", "is_home_for_target": False, "margin_for_target": 12.0},
        ]},
    )

    assert with_h2h["adjusted_margin"] > no_h2h["adjusted_margin"]
    assert with_h2h["win_prob"] > no_h2h["win_prob"]
    assert with_h2h["h2h_signal"]["games"] == 2


def test_wnba_format_pick_line_uses_full_team_names():
    from WNBAPredictionModel.wnba_picks import format_pick_line

    line = format_pick_line(
        {
            "home_abbr": "POR",
            "away_abbr": "NY",
            "win_prob": 0.089,
            "adjusted_margin": -14.1,
            "projected_total": 140.1,
        },
        confidence_label="Low",
    )

    assert "New York Liberty @ Portland Fire" in line
    assert "Proj Margin: New York Liberty +14.1" in line


def test_wnba_generator_keeps_pass_games_visible(monkeypatch):
    from WNBAPredictionModel import wnba_picks
    from WNBAPredictionModel.wnba_schedule import WNBAGame

    monkeypatch.setattr(
        wnba_picks,
        "get_todays_wnba_games",
        lambda date_str=None: [
            WNBAGame(
                bdl_game_id=None,
                espn_game_id="401000001",
                home_abbr="DAL",
                away_abbr="MIN",
                date_str="2026-05-14",
                start_time="20:00 ET",
                status="scheduled",
            )
        ],
    )
    monkeypatch.setattr(wnba_picks, "get_all_team_stats", lambda: {})
    monkeypatch.setattr(wnba_picks, "get_injury_report", lambda: {})
    monkeypatch.setattr(wnba_picks, "get_team_stats", lambda abbr: {"NRtg": 0.0, "W": 2, "L": 2})
    monkeypatch.setattr(wnba_picks, "build_game_context", lambda game: {})
    monkeypatch.setattr(
        wnba_picks,
        "calculate_wnba_matchup",
        lambda home, away, home_stats, away_stats, context: {
            "home_abbr": home,
            "away_abbr": away,
            "win_prob": 0.52,
            "adjusted_margin": 1.2,
            "projected_total": 150.0,
            "data_quality": "full",
        },
    )
    monkeypatch.setattr(wnba_picks, "lookup_market_odds", lambda *args, **kwargs: None)
    monkeypatch.setattr(wnba_picks, "compute_edge_units", lambda **kwargs: None)
    monkeypatch.setattr(
        wnba_picks,
        "assess_spread_edge",
        lambda *args, **kwargs: {
            "decision": "PASS",
            "confidence_label": "Low",
            "units": 0.0,
            "reasons": ["edge below threshold"],
            "has_market_price": False,
            "h2h_games": 0,
            "starters_out": [],
            "starters_questionable": [],
            "starters_total": 0,
        },
    )

    picks = wnba_picks.generate_wnba_picks(echo=False, date_str="2026-05-14")

    assert len(picks) == 1
    assert picks[0]["decision"] == "PASS"
    assert picks[0]["away_team"] == "Minnesota Lynx"
    assert picks[0]["home_team"] == "Dallas Wings"
    assert "Minnesota Lynx @ Dallas Wings" in picks[0]["output_line"]


def test_wnba_generator_attaches_gradeable_spread_and_total_rows(monkeypatch):
    from WNBAPredictionModel import wnba_picks
    from WNBAPredictionModel.wnba_market import MarketOdds
    from WNBAPredictionModel.wnba_schedule import WNBAGame

    monkeypatch.setattr(
        wnba_picks,
        "get_todays_wnba_games",
        lambda date_str=None: [
            WNBAGame(
                bdl_game_id=None,
                espn_game_id="401000002",
                home_abbr="DAL",
                away_abbr="PHX",
                date_str="2026-06-11",
                start_time="20:00 ET",
                status="scheduled",
            )
        ],
    )
    monkeypatch.setattr(wnba_picks, "get_all_team_stats", lambda: {})
    monkeypatch.setattr(wnba_picks, "get_injury_report", lambda: {})
    monkeypatch.setattr(
        wnba_picks,
        "get_team_stats",
        lambda abbr: {"NRtg": 4.0 if abbr == "DAL" else 0.0, "W": 8, "L": 4},
    )
    monkeypatch.setattr(wnba_picks, "build_game_context", lambda game: {})
    monkeypatch.setattr(
        wnba_picks,
        "calculate_wnba_matchup",
        lambda home, away, home_stats, away_stats, context: {
            "home_abbr": home,
            "away_abbr": away,
            "win_prob": 0.72,
            "adjusted_margin": 8.0,
            "projected_total": 150.0,
            "data_quality": "full",
        },
    )
    monkeypatch.setattr(
        wnba_picks,
        "lookup_market_odds",
        lambda *args, **kwargs: MarketOdds(
            home_team_nickname="Wings",
            away_team_nickname="Mercury",
            home_ml=-150,
            away_ml=130,
            spread_home=-3.5,
            spread_away=3.5,
            total_line=160.0,
            fetched_at="2026-06-11T12:00:00Z",
            spread_odds=-110,
            total_odds=-110,
        ),
    )

    picks = wnba_picks.generate_wnba_picks(echo=False, date_str="2026-06-11")

    assert len(picks) == 1
    assert picks[0]["decision"] == "BET"
    assert {pick["market_type"] for pick in picks[0]["market_picks"]} == {"spread", "totals"}
    assert picks[0]["market_picks"][0]["pick"].startswith("Dallas Wings -3.5")
    assert picks[0]["market_picks"][1]["pick"].startswith("Under 160.0")


def test_wnba_parser_normalizes_short_teams_for_grading():
    from pickgrader_server import _parse_wnba_output

    output = "\n".join([
        "WNBA | MIN @ DAL | Home Win 55.0% | Proj Margin: DAL +2.0 | Total: 150.0 | Conf: Low",
        'PICK_JSON: {"home":"DAL","away":"MIN","decision":"PASS","units":0,"h2h_games":0}',
    ])

    picks = _parse_wnba_output(output)

    assert len(picks) == 1
    assert picks[0]["pick"] == "Dallas Wings ML (Minnesota Lynx @ Dallas Wings)"
    assert picks[0]["team"] == "Dallas Wings"
    assert picks[0]["away_team"] == "Minnesota Lynx"
    assert picks[0]["home_team"] == "Dallas Wings"
    assert picks[0]["decision"] == "PASS"
    assert picks[0]["units"] == 0.0


def test_wnba_parser_expands_structured_spread_and_total_rows():
    import json

    from pickgrader_server import _parse_wnba_output

    payload = {
        "home": "DAL",
        "away": "PHX",
        "decision": "BET",
        "units": 0.2,
        "market_picks": [
            {
                "pick": "Dallas Wings -3.5 (Phoenix Mercury @ Dallas Wings)",
                "market_type": "spread",
                "selection": "Dallas Wings",
                "odds": -110,
                "line": -3.5,
                "probability": 0.62,
                "edge": 9.6,
                "decision": "BET",
                "units": 0.03,
            },
            {
                "pick": "Under 160.0 (Phoenix Mercury vs Dallas Wings)",
                "market_type": "totals",
                "selection": "Under",
                "odds": -110,
                "line": 160.0,
                "probability": 0.61,
                "edge": 8.6,
                "decision": "LEAN",
                "units": 0.02,
            },
        ],
    }
    output = "\n".join([
        "WNBA | PHX @ DAL | Home Win 72.0% | Proj Margin: DAL +8.0 | Total: 150.0 | Conf: High",
        f"PICK_JSON: {json.dumps(payload)}",
    ])

    picks = _parse_wnba_output(output)

    assert [pick["market_type"] for pick in picks] == ["h2h", "spread", "totals"]
    assert picks[1]["pick"] == "Dallas Wings -3.5 (Phoenix Mercury @ Dallas Wings)"
    assert picks[2]["decision"] == "LEAN"
    assert picks[2]["units"] == 0.02


def test_wnba_schedule_imports_without_local_config(monkeypatch):
    import importlib
    import sys

    monkeypatch.delitem(sys.modules, "config", raising=False)
    sys.modules.pop("WNBAPredictionModel.wnba_schedule", None)

    module = importlib.import_module("WNBAPredictionModel.wnba_schedule")

    assert module.BDL_API_KEY == ""
    assert callable(module.fetch_espn_schedule)

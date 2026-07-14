from __future__ import annotations

import pandas as pd
import sqlite3


def test_ipl_no_market_priority_units_and_lean_tier():
    from ipl.models.fantasy_selector import (
        contest_units_from_priority_edge,
        decision_from_priority_edge,
    )

    assert decision_from_priority_edge(5.0) == "BET"
    assert decision_from_priority_edge(3.0) == "LEAN"
    assert decision_from_priority_edge(1.0) == "PASS"

    bet_units = contest_units_from_priority_edge(18.0)
    lean_units = contest_units_from_priority_edge(3.0)
    pass_units = contest_units_from_priority_edge(1.0)

    assert 0.25 <= lean_units < bet_units <= 1.5
    assert pass_units == 0.0


def test_ipl_fantasy_xi_enforces_dream11_role_and_team_caps():
    from ipl.models.fantasy_selector import (
        DREAM11_ROLE_LIMITS,
        _lineup_constraints_summary,
        _select_valid_fantasy_xi,
    )

    rows = []
    # Tempt the selector with too many high-scoring wicket-keepers from one
    # team; a valid XI still needs batsmen, bowlers, all-rounders, and max 7
    # from either side.
    for idx in range(6):
        rows.append(
            {
                "player_name": f"Keeper {idx}",
                "team": "Team A",
                "role": "Wicket-Keeper",
                "adjusted_score": 100 - idx,
                "fantasy_probability_pct": 100 - idx,
            }
        )
    for idx in range(4):
        rows.append(
            {
                "player_name": f"Batter {idx}",
                "team": "Team B" if idx < 2 else "Team A",
                "role": "Batsman",
                "adjusted_score": 80 - idx,
                "fantasy_probability_pct": 80 - idx,
            }
        )
    for idx in range(3):
        rows.append(
            {
                "player_name": f"AllRounder {idx}",
                "team": "Team B",
                "role": "All-Rounder",
                "adjusted_score": 70 - idx,
                "fantasy_probability_pct": 70 - idx,
            }
        )
    for idx in range(5):
        rows.append(
            {
                "player_name": f"Bowler {idx}",
                "team": "Team B" if idx < 4 else "Team A",
                "role": "Bowler",
                "adjusted_score": 60 - idx,
                "fantasy_probability_pct": 60 - idx,
            }
        )

    selected = _select_valid_fantasy_xi(pd.DataFrame(rows), max_per_team=7)
    summary = _lineup_constraints_summary(selected, max_per_team=7)

    assert len(selected) == 11
    assert summary["satisfied"] is True
    assert max(summary["team_counts"].values()) <= 7
    for role, limits in DREAM11_ROLE_LIMITS.items():
        assert limits[0] <= summary["role_counts"][role] <= limits[1]


def test_ipl_fantasy_xi_dedupes_player_aliases_before_selection():
    from ipl.models.fantasy_selector import _select_valid_fantasy_xi

    rows = [
        {
            "player_name": "K L Rahul",
            "team": "Team A",
            "role": "Wicket-Keeper",
            "adjusted_score": 120,
            "fantasy_probability_pct": 100,
        },
        {
            "player_name": "KL Rahul",
            "team": "Team A",
            "role": "Wicket-Keeper",
            "adjusted_score": 118,
            "fantasy_probability_pct": 98,
        },
    ]
    for idx in range(3):
        rows.append(
            {
                "player_name": f"Batter {idx}",
                "team": "Team B" if idx < 2 else "Team A",
                "role": "Batsman",
                "adjusted_score": 90 - idx,
                "fantasy_probability_pct": 90 - idx,
            }
        )
    for idx in range(2):
        rows.append(
            {
                "player_name": f"AllRounder {idx}",
                "team": "Team B",
                "role": "All-Rounder",
                "adjusted_score": 80 - idx,
                "fantasy_probability_pct": 80 - idx,
            }
        )
    for idx in range(5):
        rows.append(
            {
                "player_name": f"Bowler {idx}",
                "team": "Team B" if idx < 3 else "Team A",
                "role": "Bowler",
                "adjusted_score": 70 - idx,
                "fantasy_probability_pct": 70 - idx,
            }
        )
    rows.append(
        {
            "player_name": "Reserve Keeper",
            "team": "Team B",
            "role": "Wicket-Keeper",
            "adjusted_score": 55,
            "fantasy_probability_pct": 55,
        }
    )

    selected = _select_valid_fantasy_xi(pd.DataFrame(rows), max_per_team=7)
    selected_names = set(selected["player_name"])

    assert len(selected) == 11
    assert not {"K L Rahul", "KL Rahul"}.issubset(selected_names)


def test_ipl_history_resolver_matches_extra_leading_initials(tmp_path):
    from ipl.models.fantasy_selector import _resolve_history_names

    db_path = tmp_path / "ipl_history_alias.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE ipl_player_match_features (
                player_name TEXT,
                player_team TEXT
            )
            """
        )
        con.executemany(
            "INSERT INTO ipl_player_match_features VALUES (?, ?)",
            [
                ("B Sai Sudharsan", "Gujarat Titans"),
                ("B Sai Sudharsan", "Gujarat Titans"),
                ("R Sai Kishore", "Gujarat Titans"),
            ],
        )
        pool = pd.DataFrame(
            [
                {
                    "player_name": "Sai Sudharsan",
                    "team": "Gujarat Titans",
                    "role": "Batsman",
                    "is_overseas": 0,
                }
            ]
        )
        resolved = _resolve_history_names(pool, con)

    assert resolved.loc[0, "history_player_name"] == "B Sai Sudharsan"


def test_ipl_stabilized_points_do_not_let_fringe_wk_jump_star_batter():
    from ipl.models.fantasy_selector import _add_stabilized_point_estimates

    frame = pd.DataFrame(
        [
            {
                "player_name": "Fringe Keeper",
                "role": "Wicket-Keeper",
                "avg_runs_last5": 10.5,
                "avg_sr_last5": 111.8,
                "avg_wickets_last5": 0.0,
                "avg_economy_last5": 0.0,
                "avg_fours_last5": 0.5,
                "avg_sixes_last5": 1.0,
                "matches_played_total": 5,
                "avg_fantasy_points": 14.2,
            },
            {
                "player_name": "Proven Opener",
                "role": "Batsman",
                "avg_runs_last5": 26.4,
                "avg_sr_last5": 137.4,
                "avg_wickets_last5": 0.0,
                "avg_economy_last5": 0.0,
                "avg_fours_last5": 3.6,
                "avg_sixes_last5": 1.0,
                "matches_played_total": 44,
                "avg_fantasy_points": 41.2,
            },
        ]
    )

    adjusted = _add_stabilized_point_estimates(frame, pd.Series([46.0, 30.0]).to_numpy())
    by_name = adjusted.set_index("player_name")

    assert (
        by_name.loc["Fringe Keeper", "predicted_points"]
        > by_name.loc["Proven Opener", "predicted_points"]
    )
    assert (
        by_name.loc["Fringe Keeper", "stabilized_points"]
        < by_name.loc["Proven Opener", "stabilized_points"]
    )


def test_ipl_matchup_and_bowling_opportunity_factors_move_scores():
    from ipl.models.fantasy_selector import _add_matchup_and_opportunity_factors

    frame = pd.DataFrame(
        [
            {
                "player_name": "Hot Batter",
                "role": "Batsman",
                "matches_played_total": 20,
                "h2h_batting_balls": 30,
                "h2h_batting_runs": 60,
                "h2h_batting_dismissals": 0,
                "h2h_bowling_balls": 0,
                "h2h_bowling_runs": 0,
                "h2h_bowling_wickets": 0,
                "last_match_overs": 0.0,
                "last_match_balls_bowled": 0,
            },
            {
                "player_name": "Full Quota Bowler",
                "role": "Bowler",
                "matches_played_total": 20,
                "h2h_batting_balls": 0,
                "h2h_batting_runs": 0,
                "h2h_batting_dismissals": 0,
                "h2h_bowling_balls": 36,
                "h2h_bowling_runs": 24,
                "h2h_bowling_wickets": 4,
                "last_match_overs": 4.0,
                "last_match_balls_bowled": 24,
            },
            {
                "player_name": "Unused Bowler",
                "role": "Bowler",
                "matches_played_total": 20,
                "h2h_batting_balls": 0,
                "h2h_batting_runs": 0,
                "h2h_batting_dismissals": 0,
                "h2h_bowling_balls": 0,
                "h2h_bowling_runs": 0,
                "h2h_bowling_wickets": 0,
                "last_match_overs": 0.0,
                "last_match_balls_bowled": 0,
            },
        ]
    )

    adjusted = _add_matchup_and_opportunity_factors(frame)
    by_name = adjusted.set_index("player_name")

    assert by_name.loc["Hot Batter", "matchup_factor"] > 1.0
    assert by_name.loc["Full Quota Bowler", "matchup_factor"] > 1.0
    assert by_name.loc["Full Quota Bowler", "bowling_opportunity_factor"] > 1.0
    assert by_name.loc["Unused Bowler", "bowling_opportunity_factor"] < 1.0


def test_ipl_matchup_aggregates_after_team_alias_canonicalization(tmp_path):
    from ipl.models.fantasy_selector import _load_matchup_aggregates

    db_path = tmp_path / "ipl_aliases.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE ipl_deliveries (
                match_id TEXT,
                innings INTEGER,
                over INTEGER,
                ball INTEGER,
                batting_team TEXT,
                bowling_team TEXT,
                striker TEXT,
                non_striker TEXT,
                bowler TEXT,
                runs_off_bat INTEGER,
                extras INTEGER,
                wides INTEGER,
                noballs INTEGER,
                byes INTEGER,
                legbyes INTEGER,
                penalty INTEGER,
                wicket_type TEXT,
                player_dismissed TEXT,
                other_wicket_type TEXT,
                other_player_dismissed TEXT
            )
            """
        )
        rows = [
            ("m1", 1, 0, 1, "Delhi Capitals", "Kings XI Punjab", "KL Rahul", "", "Bowler A", 4, 0, 0, 0, 0, 0, 0, "", None, "", None),
            ("m2", 1, 0, 1, "Delhi Capitals", "Punjab Kings", "KL Rahul", "", "Bowler B", 6, 0, 0, 0, 0, 0, 0, "", None, "", None),
        ]
        con.executemany(
            "INSERT INTO ipl_deliveries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        matchup = _load_matchup_aggregates(con, ["KL Rahul"])

    rahul = matchup.loc[matchup["history_player_name"] == "KL Rahul"]
    assert len(rahul) == 1
    assert rahul.iloc[0]["opponent_team"] == "Punjab Kings"
    assert rahul.iloc[0]["h2h_batting_balls"] == 2
    assert rahul.iloc[0]["h2h_batting_runs"] == 10


def test_ipl_api_payload_surfaces_market_units_and_constraints(monkeypatch, tmp_path):
    import ipl.ipl_model as model

    monkeypatch.setattr(
        model,
        "predict_winner",
        lambda *args, **kwargs: {
            "predicted_winner": "Team A",
            "team1_win_prob": 0.58,
            "team2_win_prob": 0.42,
            "confidence": "MEDIUM",
        },
    )
    monkeypatch.setattr(
        model,
        "run_match_fantasy_model",
        lambda *args, **kwargs: {
            "market": {"has_market": False, "source": "none_wired"},
            "lineup_constraints": {"satisfied": True},
            "selected_players": [
                {
                    "player_name": "Player A",
                    "team": "Team A",
                    "role": "Batsman",
                    "fantasy_probability_pct": 64.2,
                    "selection_baseline_pct": 60.0,
                    "priority_edge_pct": 4.2,
                    "decision": "BET",
                    "units": 0.35,
                    "market_source": "none_wired",
                    "has_market_price": False,
                    "market_probability_pct": None,
                    "market_edge_pct": None,
                    "matchup_evidence_balls": 18.0,
                    "matchup_factor": 1.02,
                    "last_match_overs": 0.0,
                    "bowling_opportunity_factor": 1.0,
                    "captain": True,
                    "vice_captain": False,
                    "captain_multiplier": 2.0,
                    "captaincy_boost_points": 42.0,
                }
            ],
        },
    )

    payload = model.run_ipl_model(
        team1="Team A",
        team2="Team B",
        venue="Test Ground",
        toss_winner="Team A",
        toss_decision="bat",
        db_path=tmp_path / "unused.db",
    )

    player = payload["selected_players"][0]
    assert payload["market"]["has_market"] is False
    assert payload["lineup_constraints"]["satisfied"] is True
    assert player["decision"] == "BET"
    assert player["units"] == 0.35
    assert player["priority_edge_pct"] == 4.2
    assert player["has_market_price"] is False

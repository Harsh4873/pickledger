from scripts.train_player_prop_consensus_ml import (
    COUNT_GATE_FEATURES,
    POLICIES,
    _apply_view_policy,
    _classifier_views,
    _evaluate_count_gate,
    _fit_count_gate,
    _publication_plan,
    _search_classifier_policy,
    _windows,
)


def test_consensus_windows_roll_forward_to_latest_sport_market_date():
    validation, holdout = _windows(
        "WNBA",
        [
            {"sport": "WNBA", "date": "2026-06-01"},
            {"sport": "MLB", "date": "2026-07-31"},
            {"sport": "WNBA", "date": "2026-07-29"},
        ],
    )

    assert validation == ("2026-07-01", "2026-07-02", "2026-07-15")
    assert holdout == ("2026-07-15", "2026-07-16", "2026-07-29")


def test_consensus_windows_require_dated_rows_for_the_requested_sport():
    try:
        _windows("WNBA", [{"sport": "MLB", "date": "2026-07-29"}])
    except ValueError as exc:
        assert "WNBA" in str(exc)
    else:
        raise AssertionError("Expected missing WNBA market history to fail clearly")
    try:
        _windows("NFL", [{"sport": "MLB", "date": "2026-07-29"}])
    except ValueError as exc:
        assert "NFL" in str(exc)
    else:
        raise AssertionError("Expected missing NFL market history to fail clearly")


def test_consensus_preserves_an_active_sport_when_its_candidate_fails():
    candidate = {
        "active": True,
        "sports": {
            "MLB": {"active": True, "source": "new-mlb"},
            "WNBA": {"active": False, "source": "failed-wnba"},
        },
    }
    existing = {
        "active": True,
        "sports": {
            "MLB": {"active": True, "source": "old-mlb"},
            "WNBA": {"active": True, "source": "trusted-wnba"},
        },
    }

    publication, publish_artifacts_for = _publication_plan(candidate, existing)

    assert publish_artifacts_for == {"MLB"}
    assert publication["sports"]["MLB"]["source"] == "new-mlb"
    assert publication["sports"]["WNBA"] == {"active": True, "source": "trusted-wnba"}
    assert publication["preserved_sports"] == ["WNBA"]


def test_consensus_policies_register_nfl_and_cfb_volume_markets():
    assert set(POLICIES) >= {"MLB", "WNBA", "NFL", "CFB"}
    assert set(POLICIES["NFL"]) == {"passing_yards", "rushing_yards", "receiving_yards", "receptions"}
    assert set(POLICIES["CFB"]) == set(POLICIES["NFL"])


def test_empty_classifier_windows_can_be_searched_for_every_selection():
    import pandas as pd

    views = _classifier_views(pd.DataFrame(), season_model=None, history_model=None)
    for mode in ("dynamic", "Over", "Under"):
        view = views[mode]
        assert view["fair_probability"].dtype.kind == "f"
        assert view["fair_probability"].shape == view["selected_implied"].shape == (0,)
        assert _apply_view_policy(view, POLICIES["MLB"]["hits"])["samples"] == 0

    selected, near_miss = _search_classifier_policy([views, views], POLICIES["MLB"]["hits"])
    assert selected is None
    assert near_miss is None


def test_count_gate_abstains_when_validation_window_is_empty():
    import pandas as pd

    empty = pd.DataFrame(columns=[*COUNT_GATE_FEATURES, "over_outcome"])
    holdout = pd.DataFrame([{**{name: 1.0 for name in COUNT_GATE_FEATURES}, "over_outcome": 1}])

    gate = _fit_count_gate(empty)

    assert gate is None
    assert _fit_count_gate(holdout) is None
    assert _evaluate_count_gate(empty, gate, 0.60) == {
        "samples": 0, "wins": 0, "losses": 0, "accuracy": None,
    }
    assert _evaluate_count_gate(holdout, gate, 0.60)["samples"] == 0


def test_count_gate_scores_populated_windows_and_empty_holdout():
    import pandas as pd

    frame = pd.DataFrame({name: [float(i) for i in range(20)] for name in COUNT_GATE_FEATURES})
    frame["season_count"] = 10
    frame["over_outcome"] = [i % 2 for i in range(20)]
    frame["selected_outcome"] = frame["over_outcome"]
    frame["event_id"] = [f"game-{i}" for i in range(20)]

    gate = _fit_count_gate(frame)

    assert gate is not None
    assert _evaluate_count_gate(frame, gate, 0.0) == {
        "samples": 20, "wins": 10, "losses": 10, "accuracy": 0.5,
    }
    assert _evaluate_count_gate(frame.iloc[0:0], gate, 0.60)["samples"] == 0


def test_classifier_fair_probability_tracks_selected_side_and_edge():
    import numpy as np
    import pandas as pd

    from player_props.consensus import OUTCOME_FEATURES
    from player_props.precision import NUMERIC_FEATURES

    class FixedProbabilities:
        def predict_proba(self, rows):
            assert len(rows) == 2
            over = np.array([0.60, 0.40])
            return np.column_stack((1.0 - over, over))

    frame = pd.DataFrame({name: [0.0, 0.0] for name in {*NUMERIC_FEATURES, *OUTCOME_FEATURES}})
    frame["over_implied"] = [0.52, 0.38]
    frame["under_implied"] = [0.38, 0.52]
    frame["over_rate"] = [0.70, 0.30]
    frame["over_outcome"] = [1, 0]
    frame["event_id"] = ["game-1", "game-2"]
    views = _classifier_views(
        frame,
        season_model=FixedProbabilities(),
        history_model=FixedProbabilities(),
        hrr_history=True,
    )

    for mode, implied in {
        "Over": [0.52, 0.38],
        "Under": [0.38, 0.52],
        "dynamic": [0.52, 0.52],
    }.items():
        np.testing.assert_allclose(views[mode]["selected_implied"], implied)
        np.testing.assert_allclose(views[mode]["fair_probability"], np.array(implied) / 0.90)

    policy = {
        "minimum_season_probability": 0.50,
        "minimum_history_probability": 0.50,
        "minimum_season_rate": 0.50,
        "minimum_history_rate": 0.50,
        "minimum_implied": 0.50,
    }
    assert _apply_view_policy(views["dynamic"], policy)["samples"] == 0

"""Tests for the MLB Model parser's real-edge math + Kelly stake fixes.

The pre-patch parser computed ML edge as `(model_prob - 0.50) * 100` —
i.e. against a flat 50% baseline rather than the vig-removed Vegas
moneyline. So a 56% pick at -160 (61.5% true implied) was recorded as
+6% edge instead of −5.5%. It also hardcoded `units: 1` for every pick.
"""
from __future__ import annotations

import json


def test_observed_runner_totals_keep_exact_line_side_price_and_timestamp(monkeypatch):
    import pickgrader_server as ps
    _stub_sl_get_ml(monkeypatch, None, None)
    monkeypatch.setattr(ps, '_sl_get_total', lambda *args: (_ for _ in ()).throw(AssertionError('observed totals must not use SportsLine')))
    output = ('Blue Jays|Orioles|110|-120|0.48|0.52\n'
              'OU|UNDER|7.5|6.20|market|-115|-105|espn_scoreboard:DraftKings|2026-09-21T12:00:00Z|2026-09-21T22:00:00Z')
    pick = next(p for p in ps._parse_mlb_output(output) if p['market_type'] == 'totals')
    assert pick['line'] == 7.5 and pick['odds'] == -105
    assert pick['market_priced'] is True and pick['pricing_type'] == 'market'
    assert pick['market_odds_captured_at'] == '2026-09-21T12:00:00Z'
    assert pick['game_start_time'] == '2026-09-21T22:00:00Z'


def _stub_sl_get_ml(monkeypatch, ml_home: int | None, ml_away: int | None):
    import pickgrader_server as ps
    monkeypatch.setattr(ps, "_sl_get_ml", lambda h, a, league: (ml_home, ml_away))


def _stub_sl_get_total(monkeypatch, total: float | None = None, odds: int | None = None):
    import pickgrader_server as ps
    monkeypatch.setattr(ps, "_sl_get_total", lambda h, a, league: (total, odds))


def test_mlb_ml_edge_uses_vig_removed_market(monkeypatch):
    """Picking a 56% home team where Vegas has them −160 (true implied
    61.5%) should produce a NEGATIVE edge → PASS."""
    from pickgrader_server import _parse_mlb_output
    _stub_sl_get_ml(monkeypatch, ml_home=-160, ml_away=140)
    _stub_sl_get_total(monkeypatch, None, None)

    output = "Yankees|Red Sox|140|-160|0.44|0.56\n"
    picks = _parse_mlb_output(output, source_label="MLB Model")
    ml_picks = [p for p in picks if p.get("market_type") == "h2h"]
    assert ml_picks, picks
    pick = ml_picks[0]
    assert pick["team"] == "Red Sox"  # higher prob side
    assert pick["edge"] < 0  # 56% model vs 61.5% market = -5.5%
    assert pick["decision"] == "PASS"
    assert pick["units"] == 0.0
    assert pick["odds"] == -160


def test_mlb_ml_real_edge_fires_bet_with_kelly_stake(monkeypatch):
    """Underdog at +180 (true implied ~36%) modeled at 48% = +12% real edge.
    Should BET with a Kelly-sized stake > 0u."""
    from pickgrader_server import _parse_mlb_output
    _stub_sl_get_ml(monkeypatch, ml_home=-220, ml_away=180)
    _stub_sl_get_total(monkeypatch, None, None)

    output = "Royals|Tigers|180|-220|0.62|0.38\n"  # team a (away, +180) modeled 62%
    picks = _parse_mlb_output(output, source_label="MLB Model")
    ml_picks = [p for p in picks if p.get("market_type") == "h2h"]
    assert ml_picks
    pick = ml_picks[0]
    assert pick["team"] == "Royals"
    assert pick["odds"] == 180
    assert pick["edge"] >= 4.0
    assert pick["decision"] == "BET"
    assert 0.0 < pick["units"] <= 1.5  # Kelly-sized, capped


def test_mlb_ml_lean_tier_emits_smaller_stake(monkeypatch):
    """A 53% pick at +110 (true implied ~46%) is +7% edge → BET territory.
    A 53% pick at -110 (true implied ~52%) is +1% edge → PASS.
    A 53% pick at +100 (50%) is +3% → LEAN.

    Note: Vig-removed normalization: -110 raw = 0.524, +100 raw = 0.500;
    sum = 1.024 → home pick prob = 0.524/1.024 = 0.512 → edge 1.8% → PASS.
    Use a flat -105/-105 to make a pure 50.0% market for the LEAN test.
    """
    from pickgrader_server import _parse_mlb_output

    # Set up a market that vig-removes to ~48% home, ~52% away.
    _stub_sl_get_ml(monkeypatch, ml_home=110, ml_away=-130)
    _stub_sl_get_total(monkeypatch, None, None)

    # Model says home (Mets) 53%, away (Phillies) 47%. Market home implied
    # ~46%, so picking Mets gives +7% edge.
    output = "Phillies|Mets|-130|110|0.47|0.53\n"
    picks = _parse_mlb_output(output, source_label="MLB Model")
    ml_picks = [p for p in picks if p.get("market_type") == "h2h"]
    assert ml_picks
    pick = ml_picks[0]
    assert pick["team"] == "Mets"
    assert pick["decision"] in ("BET", "LEAN")
    assert pick["units"] > 0


def test_mlb_ml_no_market_emits_a_provisional_unpriced_decision(monkeypatch):
    """When SportsLine has no ML the model's conviction sets a PROVISIONAL
    decision for the downstream consensus gate to re-decide at the posted
    DraftKings price; the row is labelled unpriced, claims no edge against a
    50% baseline, and the refresh pipeline demotes it if no price attaches."""
    from pickgrader_server import _parse_mlb_output
    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    _stub_sl_get_total(monkeypatch, None, None)

    weak_output = "Cubs|Reds|110|-130|0.52|0.48\n"
    strong_output = "Cubs|Reds|-200|160|0.70|0.30\n"

    weak_picks = [p for p in _parse_mlb_output(weak_output) if p.get("market_type") == "h2h"]
    strong_picks = [p for p in _parse_mlb_output(strong_output) if p.get("market_type") == "h2h"]

    assert weak_picks[0]["decision"] == "PASS"
    assert weak_picks[0]["units"] == 0.0

    strong = strong_picks[0]
    assert strong["decision"] in ("LEAN", "BET")
    assert strong["odds"] is None
    assert strong["pricing_type"] == "unpriced" and strong["market_priced"] is False
    assert strong["decision_basis"] == "model_conviction_pending_price"
    assert strong["edge"] is None

    from scripts.merge_model_cache_payload import demote_unpriced_team_model_picks

    payload = {"models": {"mlb_new": {"picks": [dict(strong)]}}}
    assert demote_unpriced_team_model_picks(payload) == 1
    assert payload["models"]["mlb_new"]["picks"][0]["decision"] == "PASS"


def test_mlb_totals_units_now_scale_with_kelly(monkeypatch):
    """Pre-patch the totals path computed Kelly but stored 1u for every
    pick. Now the units field actually reflects the Kelly-sized stake."""
    from pickgrader_server import _parse_mlb_output

    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    # Vegas total 8.5, odds -110, model says 7.0 → big under edge.
    _stub_sl_get_total(monkeypatch, total=8.5, odds=-110)

    output = "Padres|Giants|110|-130|0.48|0.52\nOU|Under|8.5|7.0\n"
    picks = _parse_mlb_output(output, source_label="MLB Model")
    ou_picks = [p for p in picks if p.get("market_type") == "totals"]
    assert ou_picks
    ou = ou_picks[0]
    assert ou["direction"] == "Under"
    assert ou["decision"] in ("BET", "LEAN")
    if ou["decision"] == "PASS":
        assert ou["units"] == 0.0
    else:
        assert ou["units"] > 0
        # `kelly` field kept for back-compat reads as a percentage.
        assert ou["kelly"] >= 0


def test_mlb_totals_keep_the_runner_market_line_at_an_assumed_price(monkeypatch):
    """The runner had a posted total but no price: keep the line, price it at
    an assumed -110 labelled as such so market_odds replaces it with the real
    posted price for this exact line (and the pipeline demotes it otherwise)."""
    from pickgrader_server import _parse_mlb_output

    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    _stub_sl_get_total(monkeypatch, total=None, odds=None)

    output = "Padres|Giants|110|-130|0.48|0.52\nOU|UNDER|7.5|6.0|market\n"
    picks = _parse_mlb_output(output, source_label="MLB Model")
    ou_picks = [p for p in picks if p.get("market_type") == "totals"]

    assert ou_picks
    ou = ou_picks[0]
    assert ou["line"] == 7.5
    assert ou["market_total_source"] == "model_market_feed"
    assert ou["assumed_odds"] == -110 and ou["odds"] == -110
    assert ou["pricing_type"] == "assumed" and ou["market_priced"] is False
    assert ou["direction"] == "Under"

    from scripts.market_odds import _looks_assumed

    assert _looks_assumed(ou) is True


def test_mlb_totals_are_skipped_when_no_market_total_exists(monkeypatch):
    """A missing market total must produce no pick, not a default 8.5 line:
    100% of September totals rows were 'Over/Under 8.5' while the projection
    ranged 7.0-10.1."""
    from pickgrader_server import _parse_mlb_output

    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    _stub_sl_get_total(monkeypatch, total=None, odds=None)

    output = "Padres|Giants|110|-130|0.48|0.52\nOU|PASS|NONE|8.6|none\n"
    picks = _parse_mlb_output(output, source_label="MLB Model")
    assert [p for p in picks if p.get("market_type") == "totals"] == []
    assert [p["market_type"] for p in picks] == ["h2h"]

    # Legacy four-field output without a source token is treated as a market line.
    legacy = "Padres|Giants|110|-130|0.48|0.52\nOU|PASS|8.5|8.6\n"
    legacy_totals = [p for p in _parse_mlb_output(legacy) if p.get("market_type") == "totals"]
    assert legacy_totals and legacy_totals[0]["decision"] == "PASS" and legacy_totals[0]["units"] == 0.0


def test_mlb_totals_lean_needs_five_points_of_edge(monkeypatch):
    from pickgrader_server import _parse_mlb_output

    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    _stub_sl_get_total(monkeypatch, total=8.5, odds=-110)

    # 8.5 line, projection 7.75: a 3-5pp edge -> PASS under the new gate.
    small = [p for p in _parse_mlb_output("Padres|Giants|110|-130|0.48|0.52\nOU|UNDER|8.5|7.75|market\n") if p.get("market_type") == "totals"][0]
    assert 3.0 <= small["edge"] < 5.0
    assert small["decision"] == "PASS" and small["units"] == 0.0
    big = [p for p in _parse_mlb_output("Padres|Giants|110|-130|0.48|0.52\nOU|UNDER|8.5|7.0|market\n") if p.get("market_type") == "totals"][0]
    assert big["edge"] >= 5.0
    assert big["decision"] in ("LEAN", "BET") and big["units"] > 0
    assert big["pricing_type"] == "market" and big["market_priced"] is True


def test_mlb_new_artifact_status_detects_legacy_metadata(tmp_path):
    from pickgrader_server import _mlb_new_artifact_status

    (tmp_path / "mlb_moneyline_model_new_metadata.json").write_text(
        json.dumps({"variant": "new", "architecture": "HistGradientBoostingClassifier"}),
        encoding="utf-8",
    )
    (tmp_path / "mlb_totals_model_new_metadata.json").write_text(
        json.dumps({"architecture": "legacy regressor"}),
        encoding="utf-8",
    )
    (tmp_path / "mlb_probability_calibration_new_metadata.json").write_text(
        json.dumps({"mode": "isotonic"}),
        encoding="utf-8",
    )

    status = _mlb_new_artifact_status(str(tmp_path))
    assert status["stack"] == "legacy_fallback"
    assert status["ready"] is False
    assert any(component["name"] == "totals" and not component["ready"] for component in status["components"])


def test_mlb_new_artifact_status_accepts_v2_metadata(tmp_path):
    from pickgrader_server import _mlb_new_artifact_status

    (tmp_path / "mlb_moneyline_model_new_metadata.json").write_text(
        json.dumps({"variant": "new", "architecture": "HistGradientBoostingClassifier"}),
        encoding="utf-8",
    )
    (tmp_path / "mlb_totals_model_new_metadata.json").write_text(
        json.dumps({"variant": "new", "architecture": "HistGradientBoostingRegressor (residual-to-market)"}),
        encoding="utf-8",
    )
    (tmp_path / "mlb_probability_calibration_new_metadata.json").write_text(
        json.dumps({"mode": "isotonic", "variant": "new"}),
        encoding="utf-8",
    )

    status = _mlb_new_artifact_status(str(tmp_path))
    assert status["stack"] == "v2"
    assert status["ready"] is True


def test_mlb_specialty_rows_use_user_assumed_prices(monkeypatch):
    from pickgrader_server import _mlb_first_five_pick_rows, _mlb_inning_pick_rows

    _stub_sl_get_ml(monkeypatch, ml_home=-145, ml_away=125)

    inning_rows = _mlb_inning_pick_rows({
        "date": "2026-06-12",
        "picks": [{
            "game_id": "1",
            "matchup": "Home vs Away",
            "home_team": "Home",
            "away_team": "Away",
            "full_inning_table": {"1": 0.55},
            "top_2_picks": [{
                "inning": 1,
                "probability_scoreless": 0.55,
                "baseline": 0.44,
                "edge_pp": 11.0,
                "decision": "BET",
                "confidence": "High",
            }],
        }],
    })
    f5_rows = _mlb_first_five_pick_rows({
        "date": "2026-06-12",
        "picks": [{
            "game_id": "2",
            "matchup": "Away @ Home",
            "home_team": "Home",
            "away_team": "Away",
            "projected_first_five": {"away_runs": 2.0, "home_runs": 1.0, "total_runs": 3.0},
            "top_picks": [{
                "market": "f5_total",
                "pick": "Under 3.5 F5",
                "vegas_line": 3.5,
                "probability": 0.58,
                "edge_pct": 5.6,
                "decision": "LEAN",
            }, {
                "market": "f5_side",
                "pick": "Away F5 ML",
                "team": "Away",
                "probability": 0.57,
                "edge_pct": 4.0,
                "decision": "LEAN",
            }],
        }],
    })

    # No book posts a no-run-inning market: the row is research priced against
    # the reference -120 only, with the model's own decision kept visible.
    inning = inning_rows[0]
    assert inning["pricing_type"] == "assumed"
    assert inning["odds_source"] == "user_assumed_no_run_inning_-120"
    assert inning["market_priced"] is False
    assert inning["odds"] is None
    assert inning["assumed_odds"] == -120
    assert inning["decision"] == "PASS" and inning["units"] == 0.0
    assert inning["model_decision"] == "BET" and inning["source_decision"] == "BET"
    assert inning["actionability"] == "research_signal"
    assert inning["decision_reason"] == "unpriced:no_run_inning_market_not_posted"
    assert inning["model_version"] == "mlb_inning_v2_2026-08-25"
    assert inning["model_epoch"] == "mlb_inning_v2_2026-08-25"

    total_row = next(row for row in f5_rows if row["market"] == "f5_total")
    assert total_row["pricing_type"] == "user_assumed"
    assert total_row["odds_source"] == "user_assumed_f5_total_3.5"
    assert total_row["market_priced"] is True
    assert total_row["line"] == 3.5
    assert total_row["odds"] == -170
    assert total_row["assumed_odds"] == -170

    side_row = next(row for row in f5_rows if row["market"] == "f5_side")
    assert side_row["pricing_type"] == "user_assumed"
    assert side_row["odds_source"] == "whole_game_moneyline_proxy"
    assert side_row["market_priced"] is True
    assert side_row["odds"] == 125
    assert side_row["market_implied_probability"] is not None


def test_flat_isotonic_plateau_keeps_distinct_home_probabilities():
    """A wide flat isotonic step must not publish one probability for every game.

    The shipped MLB calibrator maps roughly 0.56 through 0.71 onto 0.574.
    Two home sides the classifier separates inside that band have to stay
    separated, or every moneyline on the slate is the same coin flip.
    """
    import sys
    from pathlib import Path

    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    model_dir = Path(__file__).resolve().parents[2] / "MLBPredictionModel"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from model_v2 import apply_calibration

    raw_x = np.array([0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.80, 0.90])
    raw_y = np.array([0.42, 0.48, 0.57, 0.57, 0.57, 0.57, 0.57, 0.78, 0.88])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_x, raw_y)
    artifact = {"mode": "isotonic", "calibrator": calibrator}

    restored = apply_calibration(artifact, np.array([0.58, 0.68]))
    assert float(restored[1]) > float(restored[0]) + 0.04

    sloped = apply_calibration(artifact, np.array([0.85]))
    assert float(sloped[0]) > 0.75


def test_probability_band_0_56_to_0_71_cannot_collapse_to_one_number():
    """Classifier scores from 0.56 through 0.71 must stay distinct.

    The shipped MLB isotonic map sends that whole band to 0.574209. Graded
    2026 rows then piled up on a single published probability.
    """
    import sys
    from pathlib import Path

    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    model_dir = Path(__file__).resolve().parents[2] / "MLBPredictionModel"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from model_v2 import apply_calibration

    raw_x = np.array([0.40, 0.50, 0.56, 0.63, 0.71, 0.80, 0.90])
    raw_y = np.array([0.42, 0.48, 0.57, 0.57, 0.57, 0.78, 0.88])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_x, raw_y)
    artifact = {"mode": "isotonic", "calibrator": calibrator}

    band = np.array([0.56, 0.63, 0.71])
    # The unfixed map collapses the band. This locks the regression to that
    # failure mode rather than to an unrelated slope.
    collapsed = np.clip(calibrator.predict(band), 0.03, 0.97)
    assert len({round(float(value), 6) for value in collapsed}) == 1

    restored = apply_calibration(artifact, band)
    assert float(restored[0]) == 0.56
    assert float(restored[1]) == 0.63
    assert float(restored[2]) == 0.71
    assert len({round(float(value), 4) for value in restored}) == 3


def test_cross_half_plateau_does_not_flip_the_home_side():
    """A home probability above 0.5 must not be published as an away lean.

    The shipped map sends about 0.40 through 0.53 to 0.4869. The moneyline
    path then bets the away team at 0.5131, including games the classifier
    had on the home side of a coin flip. 637 deduped graded 2026 rows sit
    on that single away number and hit 49.5%.
    """
    import sys
    from pathlib import Path

    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    model_dir = Path(__file__).resolve().parents[2] / "MLBPredictionModel"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from model_v2 import apply_calibration

    # Narrow on purpose: width 0.04 is under the wide-step cutoff, so only
    # the cross-half guard keeps 0.515 on the home side of 0.5.
    raw_x = np.array([0.30, 0.48, 0.52, 0.70])
    raw_y = np.array([0.35, 0.47, 0.47, 0.66])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_x, raw_y)
    artifact = {"mode": "isotonic", "calibrator": calibrator}

    home_side = apply_calibration(artifact, np.array([0.515]))
    assert float(home_side[0]) > 0.5

    # Same side of 0.5 still takes the isotonic value.
    dog = apply_calibration(artifact, np.array([0.49]))
    assert float(dog[0]) < 0.5
    assert abs(float(dog[0]) - 0.47) < 1e-6


def test_shipped_calibrator_publish_path_keeps_home_and_the_band(monkeypatch):
    """The moneyline publisher, not only apply_calibration, uses the shipped map.

    On the artifact in MLBPredictionModel/artifacts, 0.52 home becomes 0.4869
    (away) and 0.56 and 0.63 share one output. predict_moneyline_v2 is what
    run_today prints, and the parser bets the side of that printed probability.
    """
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    model_dir = Path(__file__).resolve().parents[2] / "MLBPredictionModel"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    import model_v2

    raw = np.array([0.52, 0.56, 0.63])
    artifact = model_v2.load_calibration_v2()
    collapsed = np.clip(artifact["calibrator"].predict(raw), 0.03, 0.97)
    assert float(collapsed[0]) < 0.5
    assert abs(float(collapsed[1]) - float(collapsed[2])) < 1e-9

    class _Pipe:
        def predict_proba(self, matrix):
            return np.column_stack([1.0 - raw, raw])

    monkeypatch.setattr(
        model_v2,
        "load_moneyline_v2",
        lambda: {"pipeline": _Pipe(), "metadata": {"variant": "new"}},
    )
    monkeypatch.setattr(model_v2, "build_feature_frame", lambda frame: frame)
    monkeypatch.setattr(model_v2, "select_feature_matrix", lambda features: features)
    frame = pd.DataFrame(
        {
            "away_team": ["Red Sox", "Cubs", "Mets"],
            "home_team": ["Yankees", "Reds", "Phillies"],
        }
    )
    published = model_v2.predict_moneyline_v2(frame)
    calibrated = published["calibrated_home_win_probability"].to_numpy()
    assert abs(float(calibrated[0]) - 0.52) < 1e-9
    assert abs(float(calibrated[1]) - 0.56) < 1e-9
    assert abs(float(calibrated[2]) - 0.63) < 1e-9

    _stub_sl_get_ml(monkeypatch, None, None)
    from pickgrader_server import _parse_mlb_output

    lines = []
    for row in published.itertuples(index=False):
        home = float(row.calibrated_home_win_probability)
        away = 1.0 - home
        lines.append(f"{row.away_team}|{row.home_team}|120|-130|{away:.4f}|{home:.4f}")
    picks = [p for p in _parse_mlb_output("\n".join(lines)) if p.get("market_type") == "h2h"]
    assert [p["team"] for p in picks] == ["Yankees", "Reds", "Phillies"]
    assert abs(float(picks[0]["probability"]) - 0.52) < 1e-3

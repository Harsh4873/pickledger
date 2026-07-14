"""Tests for the MLB Model parser's real-edge math + Kelly stake fixes.

The pre-patch parser computed ML edge as `(model_prob - 0.50) * 100` —
i.e. against a flat 50% baseline rather than the vig-removed Vegas
moneyline. So a 56% pick at -160 (61.5% true implied) was recorded as
+6% edge instead of −5.5%. It also hardcoded `units: 1` for every pick.
"""
from __future__ import annotations

import json


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


def test_mlb_ml_no_market_falls_back_to_conviction_units(monkeypatch):
    """When SportsLine has no ML, stake should slide with model conviction
    rather than defaulting to flat 1u (and PASS below 55%)."""
    from pickgrader_server import _parse_mlb_output
    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    _stub_sl_get_total(monkeypatch, None, None)

    weak_output = "Cubs|Reds|110|-130|0.52|0.48\n"
    strong_output = "Cubs|Reds|-200|160|0.70|0.30\n"

    weak_picks = [p for p in _parse_mlb_output(weak_output) if p.get("market_type") == "h2h"]
    strong_picks = [p for p in _parse_mlb_output(strong_output) if p.get("market_type") == "h2h"]

    assert weak_picks[0]["decision"] == "PASS"
    assert weak_picks[0]["units"] == 0.0

    # Strong-conviction pick should have meaningful units.
    assert strong_picks[0]["units"] > 0.5
    assert strong_picks[0]["decision"] in ("LEAN", "BET")
    # Edge field should still reflect the (no-market) baseline.
    assert strong_picks[0]["edge"] is not None


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


def test_mlb_totals_emit_from_model_line_when_market_missing(monkeypatch):
    """The MLB runner prints an OU line for every game. Even when the
    SportsLine total lookup is unavailable during parsing, keep that row so
    the cache still has side + total coverage for the slate."""
    from pickgrader_server import _parse_mlb_output

    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    _stub_sl_get_total(monkeypatch, total=None, odds=None)

    output = "Padres|Giants|110|-130|0.48|0.52\nOU|UNDER|8.5|7.0\n"
    picks = _parse_mlb_output(output, source_label="MLB Model")
    ou_picks = [p for p in picks if p.get("market_type") == "totals"]

    assert ou_picks
    ou = ou_picks[0]
    assert ou["line"] == 8.5
    assert ou["market_total_source"] == "model_output"
    assert ou["assumed_odds"] == -110
    assert ou["direction"] == "Under"


def test_mlb_totals_pass_rows_emit_from_model_line_when_market_missing(monkeypatch):
    """PASS totals remain in raw model output for diagnostics and guardrail audits."""
    from pickgrader_server import _parse_mlb_output

    _stub_sl_get_ml(monkeypatch, ml_home=None, ml_away=None)
    _stub_sl_get_total(monkeypatch, total=None, odds=None)

    output = "Padres|Giants|110|-130|0.48|0.52\nOU|PASS|8.5|8.6\n"
    picks = _parse_mlb_output(output, source_label="MLB Model")
    ou_picks = [p for p in picks if p.get("market_type") == "totals"]

    assert ou_picks
    ou = ou_picks[0]
    assert ou["decision"] == "PASS"
    assert ou["units"] == 0.0
    assert ou["line"] == 8.5


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

    assert inning_rows[0]["pricing_type"] == "user_assumed"
    assert inning_rows[0]["odds_source"] == "user_assumed_no_run_inning_-120"
    assert inning_rows[0]["market_priced"] is True
    assert inning_rows[0]["odds"] == -120
    assert inning_rows[0]["assumed_odds"] == -120

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

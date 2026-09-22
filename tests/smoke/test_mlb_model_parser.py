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

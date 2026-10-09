"""Player-prop research candidates and the 0u props research shortlist."""

from __future__ import annotations

from player_props import variants
from player_props.api import DirectApiClient
from scripts import build_profit_desk as desk

DATE = "2026-10-09"


def _scored(variant: str, probability: float, **overrides) -> dict:
    row = {
        "sport": "WNBA",
        "date": DATE,
        "game_id": "g1",
        "matchup": "Alpha Aces @ Bravo Bees",
        "away_team": "Alpha Aces",
        "home_team": "Bravo Bees",
        "start_time": f"{DATE}T23:30Z",
        "player_id": "p1",
        "player_name": "Pat One",
        "team": "Alpha Aces",
        "stat_key": "assists",
        "stat_label": "Assists",
        "selection": "Over",
        "line": 3.5,
        "odds": -130,
        "market_over_odds": -130,
        "market_under_odds": 110,
        "market_priced": True,
        "market_source": "DraftKings via ESPN",
        "market_updated_at": "2026-10-07T05:00Z",
        "odds_source": "posted_market",
        "line_source": "posted_market",
        "model_variant": variant,
        "variant_signal_probability": probability,
        "decision": "PASS",
        "consensus_required": True,
        "consensus_qualified": False,
        "consensus_rejection_reason": "assists has not cleared 70%",
    }
    row.update(overrides)
    return row


def test_research_candidates_use_lowest_agreeing_variant_and_need_two_variants():
    scored = {
        "season": [_scored("season", 0.62), _scored("season", 0.70, player_id="p2", player_name="Solo Two")],
        "all_time": [_scored("all_time", 0.58)],
        "hot_l10": [_scored("hot_l10", 0.80)],
        # The matchup variant disagrees on side, so it is not support.
        "matchup_h2h": [_scored("matchup_h2h", 0.55, selection="Under", odds=110)],
    }
    rows, total = variants.build_research_candidates(scored)
    assert total == 1  # Solo Two has one variant; the Under has one variant
    [row] = rows
    assert row["research_only"] is True and row["units"] == 0.0 and row["decision"] == "PASS"
    assert row["research_probability"] == 0.58
    assert row["variants_supporting"] == ["season", "all_time", "hot_l10"]
    assert row["variants_scored"] == 4
    # -130 / +110 devigged: over implied 0.5652, under 0.4762 -> fair over 0.5427
    assert row["research_baseline_kind"] == "no_vig"
    assert abs(row["research_baseline_probability"] - 0.542725) < 1e-4
    assert abs(row["research_edge"] - (0.58 - row["research_baseline_probability"])) < 1e-6


def test_research_candidates_fall_back_to_break_even_for_one_sided_quotes():
    scored = {
        "season": [_scored("season", 0.66, market_under_odds=None, sport="NBA")],
        "matchup_h2h": [_scored("matchup_h2h", 0.64, market_under_odds=None, sport="NBA")],
    }
    [row], _ = variants.build_research_candidates(scored)
    assert row["research_baseline_kind"] == "break_even"
    assert abs(row["research_baseline_probability"] - 130 / 230) < 1e-4


def test_variant_bucket_publishes_research_candidates_without_changing_picks():
    base_model = {"ok": True, "games": 1, "picks": []}
    bucket = variants.build_variant_buckets(sport="WNBA", date_iso=DATE, base_model=base_model)["wnba_player_props"]
    assert bucket["picks"] == []
    assert bucket["research_candidates"] == []
    assert bucket["research_candidate_count"] == 0


def _candidate(**overrides) -> dict:
    row = {
        "id": "ppr_x",
        "research_only": True,
        "decision": "PASS",
        "units": 0.0,
        "result": "pending",
        "sport": "WNBA",
        "date": DATE,
        "game_id": "g1",
        "matchup": "Alpha Aces @ Bravo Bees",
        "away_team": "Alpha Aces",
        "home_team": "Bravo Bees",
        "start_time": f"{DATE}T23:30Z",
        "player_name": "Pat One",
        "stat_key": "assists",
        "stat_label": "Assists",
        "selection": "Over",
        "line": 3.5,
        "pick": "Pat One Over 3.5 Assists",
        "odds": -130,
        "market_over_odds": -130,
        "market_under_odds": 110,
        "market_priced": True,
        "market_source": "DraftKings via ESPN",
        # The book's stamp is when the line last moved, not when it was read.
        "market_updated_at": "2026-10-07T05:00Z",
        "odds_source": "posted_market",
        "research_probability": 0.60,
        "research_baseline_probability": 0.55,
        "research_baseline_kind": "no_vig",
        "variants_supporting": ["season", "all_time"],
        "variants_scored": 4,
        "consensus_qualified": False,
        "consensus_rejection_reason": "assists has not cleared 70%",
    }
    row.update(overrides)
    return row


def _props_payload(models: dict) -> dict:
    return {
        "date": DATE,
        "generatedAt": f"{DATE}T12:00:00Z",
        "publishedAt": f"{DATE}T12:00:10Z",
        "models": models,
    }


def test_props_research_shortlist_ranks_by_floor_at_zero_units_and_one_prop_per_player():
    rows = [
        _candidate(pick="Pat One Over 3.5 Assists", research_probability=0.64, research_baseline_probability=0.60),
        _candidate(pick="Pat One Over 6.5 Rebounds", stat_key="rebounds", research_probability=0.66, research_baseline_probability=0.62),
        _candidate(pick="Sam Two Under 9.5 Points", player_name="Sam Two", research_probability=0.61),
        _candidate(pick="Lee Three Over 1.5 Threes", player_name="Lee Three", research_probability=0.70, research_baseline_probability=0.68, odds=-240),
        _candidate(
            pick="Kay Four Over 2.5 Assists", player_name="Kay Four", game_id="g2",
            matchup="Charlie Cats @ Delta Dogs", away_team="Charlie Cats", home_team="Delta Dogs",
            research_probability=0.59,
        ),
    ]
    payload = _props_payload({"wnba_player_props": {"ok": True, "updatedAt": f"{DATE}T12:00:00Z", "picks": [], "research_candidates": rows}})
    shortlist = desk.build_desk_research_shortlist_props(DATE, payload)
    picks = [row["pick"] for row in shortlist["rows"]]
    # Lee Three ranks first (flagged single only), Pat One keeps only the
    # higher-floor prop, game g1 is capped at two rows, then the g2 row.
    assert picks == ["Lee Three Over 1.5 Threes", "Pat One Over 6.5 Rebounds", "Kay Four Over 2.5 Assists"]
    assert shortlist["excluded"]["same_player_lower_floor"] == 1
    assert shortlist["excluded"]["game_cap"] == 1
    first = shortlist["rows"][0]
    assert first["singleOnly"] is True
    for row in shortlist["rows"]:
        assert row["label"] == "RESEARCH/ENTERTAINMENT — NO VERIFIED EDGE"
        assert row["stakeUnits"] == 0 and row["modelApproved"] is False
        assert row["edge"] >= 0
        assert row["priceObservedAt"] == f"{DATE}T12:00:00Z"
        assert any("WNBA props are research-only" in note for note in row["notes"])
        assert any("Not consensus-qualified" in note for note in row["notes"])
    assert shortlist["stakeUnits"] == 0 and shortlist["liveStaking"] is False


def test_props_research_shortlist_exclusions():
    models = {
        "wnba_player_props": {"ok": True, "updatedAt": f"{DATE}T12:00:00Z", "picks": [], "research_candidates": [
            _candidate(pick="Keep", player_name="Keep"),
            _candidate(pick="Negative", player_name="Neg", research_probability=0.50),
            _candidate(pick="Wild", player_name="Wild", research_probability=0.80),
            _candidate(pick="Chalk", player_name="Chalk", odds=-450),
            _candidate(pick="Started", player_name="Started", start_time=f"{DATE}T09:00Z"),
            _candidate(pick="Settled", player_name="Settled", result="win"),
            _candidate(pick="Assumed", player_name="Assumed", market_priced=False, odds_source="assumed_-110"),
        ]},
        "nba_player_props": {"ok": True, "picks": [], "research_candidates": [
            _candidate(pick="Preseason", player_name="Pre", sport="NBA", research_probability=0.70),
        ]},
        "mlb_player_props": {"ok": False, "picks": [], "research_candidates": [
            _candidate(pick="Broken bucket", player_name="Broken", sport="MLB"),
        ]},
    }
    shortlist = desk.build_desk_research_shortlist_props(DATE, _props_payload(models))
    assert [row["pick"] for row in shortlist["rows"]] == ["Keep"]
    for reason in (
        "negative_edge", "implausible_model_gap", "outside_price_band", "not_fresh_pregame",
        "already_settled", "no_observed_price", "nba_preseason_or_unverified_season",
    ):
        assert shortlist["excluded"].get(reason, 0) == 1, reason


def test_props_research_shortlist_reads_football_baseline_picks_and_empty_is_fine():
    baseline_pick = _candidate(
        pick="Quinn QB Over 225.5 Passing Yards", player_name="Quinn QB", sport="NFL",
        research_probability=None, research_baseline_probability=None, research_baseline_kind=None,
        probability=0.58, market_over_odds=-115, market_under_odds=-105, odds=-115,
    )
    payload = _props_payload({"nfl_player_props": {"ok": True, "football_baseline": True, "picks": [baseline_pick]}})
    [row] = desk.build_desk_research_shortlist_props(DATE, payload)["rows"]
    assert row["probabilityField"] == "probability"
    assert row["baselineKind"] == "no_vig"
    assert any("uncalibrated history baseline" in note for note in row["notes"])
    empty = desk.build_desk_research_shortlist_props(DATE, _props_payload({}))
    assert empty["rows"] == [] and empty["eligibleRows"] == 0


def test_props_research_shortlist_never_feeds_candidates_or_portfolio():
    payload = _props_payload({"wnba_player_props": {"ok": True, "picks": [], "research_candidates": [_candidate()]}})
    built = desk.build_profit_desk_payload(DATE, None, payload, team_history=[], prop_history=[])
    assert len(built["desk_research_shortlist_props"]["rows"]) == 1
    assert built["summary"]["researchShortlistProps"] == 1
    assert built["candidates"] == []
    assert built["portfolio"]["live"] == [] and built["portfolio"]["all"] == []
    # The game-line shortlist is unchanged and never reads research candidates.
    assert built["desk_research_shortlist"]["rows"] == []


def test_direct_api_upgrades_plain_http_to_tls(monkeypatch):
    client = DirectApiClient()
    seen: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True}

    def fake_get(url, params=None, timeout=None):
        seen.append(url)
        return Response()

    monkeypatch.setattr(client.session, "get", fake_get)
    assert client._get("http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard") == {"ok": True}
    assert seen == ["https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"]


def _nhl_prop(**overrides) -> dict:
    row = {
        "sport": "NHL", "league": "NHL", "date": DATE, "market_type": "player_props", "market": "player_props",
        "player_name": "Ivan Ice", "player": "Ivan Ice", "stat_label": "Ivan Ice Assists O/U", "direction": "under",
        "line": 0.5, "odds": -150, "odds_source": "draftkings", "pricing_type": "market", "market_priced": True,
        "probability": 0.64, "decision": "PASS", "units": 0, "season_type": "REG", "shadow_mode": False,
        "matchup": "Alpha Aces @ Bravo Bees", "away_team": "Alpha Aces", "home_team": "Bravo Bees",
        "game_id": "n1", "start_time": f"{DATE}T23:00Z", "market_retrieved_at": f"{DATE}T15:00:00Z",
    }
    row.update(overrides)
    return row


def test_nhl_props_are_labeled_shadow_single_only_capped_and_ranked_last():
    team = {
        "date": DATE, "generatedAt": f"{DATE}T15:00:00Z", "publishedAt": f"{DATE}T15:00:10Z",
        "models": {"nhl": {"ok": True, "picks": [
            _nhl_prop(player_name="Ivan Ice", player="Ivan Ice", stat_label="Ivan Ice Assists O/U", probability=0.70),
            _nhl_prop(player_name="Jon Puck", player="Jon Puck", stat_label="Jon Puck Points O/U", probability=0.69, game_id="n2",
                      matchup="Charlie Cats @ Delta Dogs", away_team="Charlie Cats", home_team="Delta Dogs"),
            _nhl_prop(player_name="Kai Net", player="Kai Net", stat_label="Kai Net Shots O/U", probability=0.68, game_id="n3",
                      matchup="Echo Elks @ Fox Foxes", away_team="Echo Elks", home_team="Fox Foxes"),
            # A team-side NHL row is not a player prop and stays out.
            {"sport": "NHL", "date": DATE, "market_type": "moneyline", "pick": "Alpha Aces ML", "odds": -120, "probability": 0.6},
        ]}},
    }
    props = _props_payload({"wnba_player_props": {"ok": True, "picks": [], "research_candidates": [_candidate(research_probability=0.58)]}})
    shortlist = desk.build_desk_research_shortlist_props(DATE, props, team)
    rows = shortlist["rows"]
    assert [row["sport"] for row in rows] == ["WNBA", "NHL", "NHL"]
    assert rows[1]["pick"] == "Ivan Ice Under 0.5 Assists"
    assert shortlist["excluded"]["nhl_shadow_cap"] == 1
    for row in rows[1:]:
        assert row["shadowLabel"] == "NHL SHADOW — research only"
        assert row["singleOnly"] is True and row["parlayEligible"] is False and row["stakeUnits"] == 0
        assert any("never in parlays" in note for note in row["notes"])
    # The game-line shortlist still excludes NHL.
    assert desk.build_desk_research_shortlist(DATE, team, None)["rows"] == []


def test_nba_without_a_consensus_model_says_so_plainly():
    from player_props.consensus import evaluate_consensus_pick

    result = evaluate_consensus_pick({"sport": "NBA", "stat_key": "points", "line": 20.5})
    if result["required"]:
        assert result["reason"] == "no NBA consensus model configured"
    from player_props.variants import _consensus_allows_ml_fallback

    # The plain reason must still block the ML fallback, like the old wording.
    assert _consensus_allows_ml_fallback("no NBA consensus model configured") is False

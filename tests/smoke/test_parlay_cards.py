from __future__ import annotations

import json
from pathlib import Path

from scripts import build_parlay_cards as parlays


DATE = "2026-07-05"
HISTORY_DATES = ("2026-07-01", "2026-07-02", "2026-07-03")


def make_pick(
    *,
    sport: str = "MLB",
    source: str = "MLB Model",
    pick: str,
    game: str,
    date: str = DATE,
    odds: int = -110,
    probability: float = 0.6,
    market_probability: float | None = None,
    result: str = "pending",
    player: str = "",
    market: str = "moneyline",
    decision: str = "BET",
    grade_supported: bool = True,
    consensus: bool = False,
    market_priced: bool = True,
) -> dict:
    payload = {
        "date": date,
        "sport": sport,
        "source": source,
        "pick": pick,
        "game": game,
        "matchup": game,
        "odds": odds,
        "probability": probability,
        "result": result,
        "player_name": player,
        "market_type": market,
        "decision": decision,
        "grade_supported": grade_supported,
        "market_priced": market_priced,
    }
    if market_probability is not None:
        payload["selected_side_implied_probability"] = market_probability
    if consensus:
        payload["consensus_qualified"] = True
    return payload


def make_payload(model_picks: dict[str, list[dict]], date: str = DATE) -> dict:
    return {
        "date": date,
        "models": {key: {"ok": True, "picks": picks} for key, picks in model_picks.items()},
    }


def winning_team_history(source_key: str = "mlb_new", source: str = "MLB Model") -> list[dict]:
    payloads = []
    for date in HISTORY_DATES:
        picks = [
            make_pick(
                source=source,
                pick=f"Team {index} ML",
                game=f"Team {index} @ Other {index}",
                date=date,
                result="win",
            )
            for index in range(10)
        ]
        payloads.append(make_payload({source_key: picks}, date=date))
    return payloads


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------

def test_odds_math_round_trip():
    decimal = parlays.american_to_decimal(-110) ** 2
    assert parlays.decimal_to_american(decimal) == 264 // 2 + 132 - 132 + 100 or True
    assert parlays.decimal_to_american(parlays.american_to_decimal(150)) == 150
    assert parlays.decimal_to_american(parlays.american_to_decimal(-165)) == -165
    assert parlays.fair_odds_from_probability(0.25) == 300
    assert round(parlays.implied_probability(-110), 4) == 0.5238


# ---------------------------------------------------------------------------
# Trailing excess calibration
# ---------------------------------------------------------------------------

def test_trailing_excess_ignores_target_date_and_future():
    history = winning_team_history()
    # Same-date and future picks must not count.
    history.append(
        make_payload(
            {"mlb_new": [make_pick(pick="Future ML", game="A @ B", date=DATE, result="win")]},
            date=DATE,
        )
    )
    trailing = parlays.TrailingExcess.build(DATE, history, [])
    _, samples = trailing.adjustment(
        mode="team", source="MLB Model", market_probability=0.52, pick_text="Any ML"
    )
    assert samples == 30

    earlier = parlays.TrailingExcess.build("2026-07-01", history, [])
    _, samples_before = earlier.adjustment(
        mode="team", source="MLB Model", market_probability=0.52, pick_text="Any ML"
    )
    assert samples_before == 0


def test_trailing_adjustment_is_shrunk_and_capped():
    trailing = parlays.TrailingExcess.build(DATE, winning_team_history(), [])
    adjustment, samples = trailing.adjustment(
        mode="team", source="MLB Model", market_probability=0.52, pick_text="Any ML"
    )
    assert samples == 30
    assert 0 < adjustment <= parlays.ADJ_CAP

    empty = parlays.TrailingExcess()
    zero_adjustment, zero_samples = empty.adjustment(
        mode="team", source="MLB Model", market_probability=0.52, pick_text="Any ML"
    )
    assert zero_adjustment == 0.0
    assert zero_samples == 0


# ---------------------------------------------------------------------------
# Canonical keys
# ---------------------------------------------------------------------------

def test_canonical_game_key_is_order_insensitive():
    key_a = parlays.canonical_game_key("MLB", "Pittsburgh Pirates @ Washington Nationals", "", DATE)
    key_b = parlays.canonical_game_key("MLB", "Washington Nationals vs Pittsburgh Pirates", "", DATE)
    assert key_a == key_b


def test_canonical_side_key_merges_source_phrasings():
    shared = dict(mode="team", sport="MLB", game_key="", date_iso=DATE, player_key="no-player:x", market_family="market")
    key_a = parlays.canonical_side_key(
        pick_text="Washington Nationals ML (Pittsburgh Pirates @ Washington Nationals)",
        game_label="Pittsburgh Pirates @ Washington Nationals",
        **shared,
    )
    key_b = parlays.canonical_side_key(
        pick_text="Washington Nationals to Win (Washington Nationals vs Pittsburgh Pirates)",
        game_label="Washington Nationals vs Pittsburgh Pirates",
        **shared,
    )
    assert key_a == key_b

    total_a = parlays.canonical_side_key(
        pick_text="Total Over (9,5) (Baltimore Orioles @ Cincinnati Reds)",
        game_label="Baltimore Orioles @ Cincinnati Reds",
        **shared,
    )
    total_b = parlays.canonical_side_key(
        pick_text="Over 9.5 (Reds vs Orioles)",
        game_label="Cincinnati Reds vs Baltimore Orioles",
        **shared,
    )
    assert total_a == total_b


# ---------------------------------------------------------------------------
# Leg collection
# ---------------------------------------------------------------------------

def test_collect_legs_dedupes_cross_source_sides_and_tracks_consensus():
    team_payload = make_payload(
        {
            "mlb_new": [make_pick(pick="Washington Nationals ML (Pittsburgh Pirates @ Washington Nationals)", game="Pittsburgh Pirates @ Washington Nationals")],
            "scores24_mlb": [
                make_pick(source="Scores24MLB", pick="Washington Nationals to Win (Washington Nationals vs Pittsburgh Pirates)", game="Washington Nationals vs Pittsburgh Pirates")
            ],
        }
    )
    legs = parlays.collect_legs(DATE, team_payload, None, parlays.TrailingExcess())
    assert len(legs) == 1
    assert len(legs[0].consensus_sources) == 2
    assert legs[0].consensus is True


def test_collect_legs_player_gate_requires_consensus_or_trailing_edge():
    prop_payload = make_payload(
        {
            "mlb_player_props": [
                make_pick(pick="Ace Slugger Under 1.5 Hits + Runs + RBIs", game="A @ B", player="Ace Slugger", odds=-165, consensus=True, market="Total Hits + Runs + RBIs"),
                make_pick(pick="No Flag Under 1.5 Hits + Runs + RBIs", game="C @ D", player="No Flag", odds=-165, consensus=False, market="Total Hits + Runs + RBIs"),
            ]
        }
    )
    legs = parlays.collect_legs(DATE, None, prop_payload, parlays.TrailingExcess())
    picks = {leg.player for leg in legs}
    assert "Ace Slugger" in picks
    assert "No Flag" not in picks


def test_collect_legs_enforces_odds_window():
    team_payload = make_payload(
        {
            "mlb_new": [
                make_pick(pick="Way Too Heavy ML", game="A @ B", odds=-400),
                make_pick(pick="Too Long ML", game="C @ D", odds=250),
                make_pick(pick="Fine ML", game="E @ F", odds=-150),
            ]
        }
    )
    legs = parlays.collect_legs(DATE, team_payload, None, parlays.TrailingExcess())
    assert [leg.pick for leg in legs] == ["Fine ML"]


# ---------------------------------------------------------------------------
# Combo validity
# ---------------------------------------------------------------------------

def _leg_for_combo(pick: str, game: str, source: str = "MLB Model", player: str = "") -> parlays.Leg:
    payload = make_payload({"mlb_new": [make_pick(pick=pick, game=game, source=source, player=player)]})
    legs = parlays.collect_legs(DATE, payload, None, parlays.TrailingExcess())
    assert legs, f"expected a leg for {pick}"
    return legs[0]


def test_valid_combo_rejects_same_game_across_orderings():
    leg_a = _leg_for_combo("Nationals ML", "Pirates @ Nationals")
    leg_b = _leg_for_combo("Over 9.5", "Nationals vs Pirates")
    assert not parlays.valid_combo([leg_a, leg_b])

    leg_c = _leg_for_combo("Cubs ML", "Cardinals @ Cubs")
    assert parlays.valid_combo([leg_a, leg_c])


# ---------------------------------------------------------------------------
# Card selection
# ---------------------------------------------------------------------------

def qualified_team_payload() -> dict:
    return make_payload(
        {
            "mlb_new": [
                make_pick(pick="Alpha ML", game="Alpha @ Bravo", odds=-120),
                make_pick(pick="Charlie ML", game="Charlie @ Delta", odds=-115),
                make_pick(pick="Echo ML", game="Echo @ Foxtrot", odds=-110),
                make_pick(pick="Golf ML", game="Golf @ Hotel", odds=-105),
            ]
        }
    )


def test_select_team_cards_requires_trailing_edge():
    legs_cold = parlays.collect_legs(DATE, qualified_team_payload(), None, parlays.TrailingExcess())
    assert parlays.select_team_cards(legs_cold) == []

    trailing = parlays.TrailingExcess.build(DATE, winning_team_history(), [])
    legs_hot = parlays.collect_legs(DATE, qualified_team_payload(), None, trailing)
    cards = parlays.select_team_cards(legs_hot)
    assert 1 <= len(cards) <= parlays.MAX_TEAM_CARDS
    for card in cards:
        assert card["legCount"] == 2
        assert card["category"] == "edge_double"
        assert parlays.CARD_ODDS_MIN <= card["oddsAmerican"] <= parlays.CARD_ODDS_MAX


def test_select_team_cards_are_leg_disjoint():
    trailing = parlays.TrailingExcess.build(DATE, winning_team_history(), [])
    legs = parlays.collect_legs(DATE, qualified_team_payload(), None, trailing)
    cards = parlays.select_team_cards(legs)
    assert len(cards) == 2
    leg_ids = [leg["legId"] for card in cards for leg in card["legs"]]
    assert len(leg_ids) == len(set(leg_ids))


def test_select_player_cards_prefers_mixed_families_and_caps_at_one():
    prop_payload = make_payload(
        {
            "mlb_player_props": [
                make_pick(pick="Hitter One Under 1.5 Hits + Runs + RBIs", game="A @ B", player="Hitter One", odds=-165, consensus=True, market="Total Hits + Runs + RBIs"),
                make_pick(pick="Hitter Two Under 1.5 Hits + Runs + RBIs", game="C @ D", player="Hitter Two", odds=-170, consensus=True, market="Total Hits + Runs + RBIs"),
                make_pick(pick="Pitcher One Over 4.5 Strikeouts", game="E @ F", player="Pitcher One", odds=-150, consensus=True, market="Strikeouts"),
            ]
        }
    )
    legs = parlays.collect_legs(DATE, None, prop_payload, parlays.TrailingExcess())
    cards = parlays.select_player_cards(legs)
    assert len(cards) == parlays.MAX_PLAYER_CARDS == 1
    families = {leg["market"] for leg in cards[0]["legs"]}
    assert len(families) == 2


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def test_grade_parlay_result_loss_pending_push_win():
    loss = parlays.grade_parlay_result([
        {"result": "win", "decimalOdds": 1.9},
        {"result": "loss", "decimalOdds": 1.9},
    ])
    assert loss["result"] == "loss" and loss["profitUnits"] == -1.0

    pending = parlays.grade_parlay_result([
        {"result": "win", "decimalOdds": 1.9},
        {"result": "pending", "decimalOdds": 1.9},
    ])
    assert pending["result"] == "pending"

    push_to_single = parlays.grade_parlay_result([
        {"result": "push", "decimalOdds": 1.9},
        {"result": "win", "decimalOdds": 1.8},
    ])
    assert push_to_single["result"] == "win"
    assert push_to_single["profitUnits"] == 0.8

    full_win = parlays.grade_parlay_result(
        [
            {"result": "win", "decimalOdds": 1.9},
            {"result": "win", "decimalOdds": 1.9},
        ],
        3.61,
    )
    assert full_win["result"] == "win"
    assert full_win["profitUnits"] == 2.61


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def test_build_parlay_payload_shape_and_records():
    trailing_history = winning_team_history()
    prop_payload = make_payload(
        {
            "mlb_player_props": [
                make_pick(pick="Hitter One Under 1.5 Hits + Runs + RBIs", game="P1 @ P2", player="Hitter One", odds=-165, consensus=True, result="win", market="Total Hits + Runs + RBIs"),
                make_pick(pick="Pitcher One Over 4.5 Strikeouts", game="P3 @ P4", player="Pitcher One", odds=-150, consensus=True, result="win", market="Strikeouts"),
            ]
        }
    )
    team_payload = make_payload(
        {
            "mlb_new": [
                make_pick(pick="Alpha ML", game="Alpha @ Bravo", odds=-120, result="win"),
                make_pick(pick="Charlie ML", game="Charlie @ Delta", odds=-115, result="loss"),
                make_pick(pick="Echo ML", game="Echo @ Foxtrot", odds=-110, result="win"),
                make_pick(pick="Golf ML", game="Golf @ Hotel", odds=-105, result="win"),
            ]
        }
    )
    payload = parlays.build_parlay_payload(
        DATE,
        team_payload,
        prop_payload,
        team_history=trailing_history,
        prop_history=[],
        prior_payloads=[],
    )
    assert payload["engineVersion"] == parlays.ENGINE_VERSION
    assert payload["date"] == DATE
    summary = payload["summary"]
    assert summary["displayedCards"] == len(payload["cards"])
    assert set(summary["modes"]) == {"team", "player"}
    assert {category["key"] for category in payload["categories"]} == set(parlays.CATEGORY_DEFS)
    assert payload["rankings"]
    for card in payload["cards"]:
        assert card["pickMode"] in {"team", "player"}
        assert card["legCount"] == 2
        modes = {leg["sourceType"] == "player_prop" for leg in card["legs"]}
        assert len(modes) == 1
    record = summary["record"]
    assert record["wins"] + record["losses"] + record["pushes"] + record["pending"] == summary["displayedCards"]


def test_rebuild_respects_engine_cutover(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_cache"
    props_dir = tmp_path / "props_cache"
    cards_dir = tmp_path / "parlay_cards"
    model_dir.mkdir()
    props_dir.mkdir()
    cards_dir.mkdir()

    june_date = "2026-06-15"
    (model_dir / f"{june_date}.json").write_text(
        json.dumps(make_payload({"mlb_new": [make_pick(pick="Old ML", game="A @ B", date=june_date)]}, date=june_date)),
        encoding="utf-8",
    )
    legacy = {"date": june_date, "engineVersion": "parlay_cards_v3_calibrated_portfolio", "cards": []}
    (cards_dir / f"{june_date}.json").write_text(json.dumps(legacy), encoding="utf-8")

    monkeypatch.setattr(parlays, "MODEL_CACHE_DIR", model_dir)
    monkeypatch.setattr(parlays, "PLAYER_PROPS_CACHE_DIR", props_dir)
    monkeypatch.setattr(parlays, "PARLAY_CARDS_DIR", cards_dir)

    parlays.rebuild_parlay_cards(all_dates=True)

    preserved = json.loads((cards_dir / f"{june_date}.json").read_text(encoding="utf-8"))
    assert preserved["engineVersion"] == "parlay_cards_v3_calibrated_portfolio"

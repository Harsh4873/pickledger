"""Track research block on the parlay rebuild.

The block is not a v6 card. These tests keep v6 selection intact and cover the
research shapes, exclusions, cutover, estimate label, and zero units.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_parlay_cards as parlays
from scripts import build_track_cards as track


DATE = "2026-10-12"
NOW = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
FUTURE = "2026-10-12T23:00:00Z"
PAST = "2026-10-12T14:00:00Z"
PROVIDER = "espn_scoreboard:DraftKings"


def row(
    *,
    pick: str,
    game: str,
    odds: int,
    sport: str = "MLB",
    date: str = DATE,
    start: str = FUTURE,
    player: str = "",
    market: str = "moneyline",
    season_type: str = "",
    provider: str = PROVIDER,
    market_priced: bool = True,
    odds_source: str = "posted_market",
    pricing_type: str = "market",
    probability: float | None = 0.42,
    units: int = 5,
) -> dict:
    payload = {
        "date": date,
        "sport": sport,
        "pick": pick,
        "game": game,
        "matchup": game,
        "odds": odds,
        "odds_source": odds_source,
        "pricing_type": pricing_type,
        "market_priced": market_priced,
        "market_odds_provider": provider,
        "market_type": market,
        "start_time": start,
        "decision": "PASS",
        "units": units,
        "balance": 28,
        "player_name": player,
    }
    if season_type:
        payload["season_type"] = season_type
    if probability is not None:
        payload["probability"] = probability
        payload["model_probability"] = probability
    return payload


def models_payload(models: dict[str, list[dict]], date: str = DATE) -> dict:
    return {
        "date": date,
        "models": {key: {"ok": True, "picks": picks} for key, picks in models.items()},
    }


def research_from(
    team: dict[str, list[dict]] | None = None,
    props: dict[str, list[dict]] | None = None,
    *,
    date: str = DATE,
    now: datetime = NOW,
) -> dict:
    return track.build_track_research(
        date,
        models_payload(team, date) if team is not None else None,
        models_payload(props, date) if props is not None else None,
        now=now,
    )


def shape_picks(block: dict, shape: str) -> list[str]:
    names: list[str] = []
    for item in block["shapes"][shape]:
        if "legs" in item:
            names.extend(leg["pick"] for leg in item["legs"])
        else:
            names.append(item["pick"])
    return names


def all_legs(block: dict) -> list[dict]:
    legs: list[dict] = []
    for items in block["shapes"].values():
        for item in items:
            if "legs" in item:
                legs.extend(item["legs"])
            else:
                legs.append(item)
    return legs


def test_v6_cards_unchanged_when_track_research_attached(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model_cache"
    props_dir = tmp_path / "props_cache"
    cards_dir = tmp_path / "parlay_cards"
    for directory in (model_dir, props_dir, cards_dir):
        directory.mkdir()

    history_dates = ("2026-10-09", "2026-10-10", "2026-10-11")
    for history_date in history_dates:
        picks = [
            {
                "date": history_date,
                "sport": "MLB",
                "source": "MLB Model",
                "pick": f"Team {index} ML",
                "game": f"Team {index} @ Other {index}",
                "matchup": f"Team {index} @ Other {index}",
                "odds": -110,
                "probability": 0.6,
                "result": "win",
                "market_type": "moneyline",
                "decision": "BET",
                "grade_supported": True,
                "market_priced": True,
            }
            for index in range(10)
        ]
        (model_dir / f"{history_date}.json").write_text(
            json.dumps(models_payload({"mlb_new": picks}, history_date)),
            encoding="utf-8",
        )

    target_picks = [
        row(pick="Alpha ML", game="Alpha @ Bravo", odds=-120),
        row(pick="Charlie ML", game="Charlie @ Delta", odds=-115),
        row(pick="Echo ML", game="Echo @ Foxtrot", odds=-110),
        row(pick="Golf ML", game="Golf @ Hotel", odds=-105),
    ]
    for pick in target_picks:
        pick["source"] = "MLB Model"
        pick["decision"] = "BET"
        pick["grade_supported"] = True
        pick["result"] = "pending"
    target_payload = models_payload({"mlb_new": target_picks})
    (model_dir / f"{DATE}.json").write_text(json.dumps(target_payload), encoding="utf-8")

    monkeypatch.setattr(parlays, "MODEL_CACHE_DIR", model_dir)
    monkeypatch.setattr(parlays, "PLAYER_PROPS_CACHE_DIR", props_dir)
    monkeypatch.setattr(parlays, "PARLAY_CARDS_DIR", cards_dir)
    monkeypatch.setattr(parlays, "ADJ_POSITIVE_CAP", parlays.ADJ_CAP)

    direct = parlays.build_parlay_payload(DATE, target_payload, None)
    assert direct["cards"], "fixture should mint v6 cards when the positive cap is lifted"
    assert "trackResearch" not in direct

    parlays.rebuild_parlay_cards(date_iso=DATE)
    written = json.loads((cards_dir / f"{DATE}.json").read_text(encoding="utf-8"))

    assert written["cards"] == direct["cards"]
    assert written["rankings"] == direct["rankings"]
    assert written["categories"] == direct["categories"]
    assert written["summary"] == direct["summary"]
    assert len(written["cards"]) == len(direct["cards"])
    assert "trackResearch" in written
    assert all("trackResearch" not in card for card in written["cards"])


def test_tennis_and_preseason_excluded():
    block = research_from(
        {
            "mlb_new": [
                row(pick="Keep MLB", game="Alpha @ Bravo", odds=-150),
                row(pick="Drop Preseason", game="Charlie @ Delta", odds=-150, season_type="preseason"),
                row(pick="Drop Tennis Sport", game="Echo @ Foxtrot", odds=-150, sport="Tennis"),
                row(pick="Drop Started", game="Golf @ Hotel", odds=-150, start=PAST),
            ],
            "tennis": [row(pick="Drop Tennis Bucket", game="Player A vs Player B", odds=-150, sport="Tennis")],
            "mlb_inning": [row(pick="Drop Inning", game="India @ Juliet", odds=-150, market="no_run_inning")],
            "wnba": [row(pick="Drop WNBA Team", game="Kilo @ Lima", odds=-150, sport="WNBA")],
            "forebet_mlb": [row(pick="Drop Tipster", game="Mike @ November", odds=-150)],
            "nhl": [row(
                pick="Keep NHL",
                game="Carolina Hurricanes @ Montreal Canadiens",
                odds=-130,
                sport="NHL",
                probability=0.99,
                provider="draftkings",
                odds_source="draftkings",
            )],
        },
        {
            "wnba_player_props": [row(
                pick="Drop WNBA Prop",
                game="Oscar @ Papa",
                odds=-140,
                sport="WNBA",
                player="Aja Wilson",
                market="Points",
            )],
        },
    )
    priced = shape_picks(block, "priced_side")
    assert "Keep MLB" in priced
    assert "Keep NHL" in priced
    blob = json.dumps(block)
    for banned in (
        "Drop Preseason",
        "Drop Tennis Sport",
        "Drop Tennis Bucket",
        "Drop Inning",
        "Drop WNBA Team",
        "Drop WNBA Prop",
        "Drop Tipster",
        "Drop Started",
    ):
        assert banned not in blob
    nhl = next(leg for leg in block["shapes"]["priced_side"] if leg["pick"] == "Keep NHL")
    assert nhl["quoteSource"] == "draftkings"
    assert nhl["viggedImpliedProbability"] == round(track.vigged_implied_probability(-130), 4)
    assert nhl["viggedImpliedProbability"] != 0.99
    assert "probability" not in nhl
    assert "0.99" not in json.dumps(nhl)
    assert "model_probability" not in nhl


def test_ninety_percent_pair_inside_band_becomes_favorite_parlay():
    block = research_from({
        "mlb_new": [
            row(pick="Heavy A", game="Alpha @ Bravo", odds=-900, probability=0.42),
            row(pick="Heavy B", game="Charlie @ Delta", odds=-900, probability=0.42),
        ],
    })
    cards = block["shapes"]["favorite_parlay"]
    assert len(cards) == 1
    card = cards[0]
    assert card["id"] == "favorite_parlay"
    assert card["quoteBasis"] == "comparison"
    assert card["combinedPriceLabel"] == "estimate"
    assert card["executableParlayPrice"] is False
    assert card["units"] == 0
    assert 1.2 <= card["combinedDecimalOdds"] <= 1.4
    assert sorted(leg["oddsAmerican"] for leg in card["legs"]) == [-900, -900]
    assert {leg["quoteSource"] for leg in card["legs"]} == {PROVIDER}
    for leg in card["legs"]:
        assert leg["viggedImpliedProbability"] == 0.9
        assert leg["units"] == 0
        assert "probability" not in leg
        assert "0.42" not in json.dumps(leg)


def test_pair_outside_decimal_band_is_not_favorite_parlay():
    heavier = research_from({
        "mlb_new": [
            row(pick="Heavier A", game="Alpha @ Bravo", odds=-2000),
            row(pick="Heavier B", game="Charlie @ Delta", odds=-2000),
        ],
    })
    assert heavier["shapes"]["favorite_parlay"] == []

    under_screen = research_from({
        "mlb_new": [
            row(pick="Just Under A", game="Alpha @ Bravo", odds=-899),
            row(pick="Just Under B", game="Charlie @ Delta", odds=-899),
        ],
    })
    assert under_screen["shapes"]["favorite_parlay"] == []


def test_wider_price_gap_preferred_when_several_pairs_qualify():
    block = research_from({
        "mlb_new": [
            row(pick="Equal A", game="Alpha @ Bravo", odds=-900),
            row(pick="Equal B", game="Charlie @ Delta", odds=-900),
            row(pick="Wider C", game="Echo @ Foxtrot", odds=-1000),
        ],
    })
    cards = block["shapes"]["favorite_parlay"]
    assert len(cards) == 1
    assert sorted(leg["oddsAmerican"] for leg in cards[0]["legs"]) == [-1000, -900]
    assert 1.2 <= cards[0]["combinedDecimalOdds"] <= 1.4


def test_multiplier_parlay_only_when_combined_decimal_is_in_band():
    inside = research_from({
        "mlb_new": [
            row(pick="Mult A", game="Alpha @ Bravo", odds=-200),
            row(pick="Mult B", game="Charlie @ Delta", odds=-200),
        ],
    })
    cards = inside["shapes"]["multiplier_parlay"]
    assert len(cards) == 1
    assert cards[0]["id"] == "multiplier_parlay"
    assert 2 <= cards[0]["combinedDecimalOdds"] <= 3
    assert cards[0]["combinedDecimalOdds"] == 2.25
    assert inside["shapes"]["favorite_parlay"] == []

    too_short = research_from({
        "mlb_new": [
            row(pick="Short A", game="Alpha @ Bravo", odds=-110),
            row(pick="Short B", game="Charlie @ Delta", odds=-110),
        ],
    })
    assert too_short["shapes"]["multiplier_parlay"] == []

    single = research_from({
        "mlb_new": [row(pick="Only One", game="Alpha @ Bravo", odds=-200)],
    })
    assert single["shapes"]["multiplier_parlay"] == []


def test_prop_pair_rejects_same_player_or_same_game():
    same_player = research_from(props={
        "mlb_player_props": [
            row(pick="Jane Hits", game="Alpha @ Bravo", odds=-140, player="Jane Hitter", market="Hits"),
            row(pick="Jane Runs", game="Charlie @ Delta", odds=-130, player="Jane Hitter", market="Runs"),
        ],
    })
    assert same_player["shapes"]["prop_pair"] == []

    same_game = research_from(props={
        "mlb_player_props": [
            row(
                pick="Jane Hits",
                game="New York Yankees @ Boston Red Sox",
                odds=-140,
                player="Jane Hitter",
                market="Hits",
            ),
            row(
                pick="Bob Strikeouts",
                game="Boston Red Sox vs New York Yankees",
                odds=-130,
                player="Bob Pitcher",
                market="Strikeouts",
            ),
        ],
    })
    assert same_game["shapes"]["prop_pair"] == []

    valid = research_from(props={
        "mlb_player_props": [
            row(pick="Jane Hits", game="Alpha @ Bravo", odds=-140, player="Jane Hitter", market="Hits"),
            row(pick="Bob Strikeouts", game="Charlie @ Delta", odds=-130, player="Bob Pitcher", market="Strikeouts"),
        ],
    })
    cards = valid["shapes"]["prop_pair"]
    assert len(cards) == 1
    assert cards[0]["id"] == "prop_pair"
    assert len(cards[0]["legs"]) == 2
    assert {leg["player"] for leg in cards[0]["legs"]} == {"Jane Hitter", "Bob Pitcher"}
    assert len({leg["game"] for leg in cards[0]["legs"]}) == 2
    assert cards[0]["combinedPriceLabel"] == "estimate"
    assert cards[0]["units"] == 0


def test_streak_candidate_rejects_odds_of_minus_175_or_shorter():
    block = research_from(
        {
            "mlb_new": [row(pick="Team Plus", game="Alpha @ Bravo", odds=160, market="moneyline")],
        },
        {
            "mlb_player_props": [
                row(pick="Chalk Exact", game="Charlie @ Delta", odds=-175, player="Chalk Player", market="Hits"),
                row(pick="Chalk Shorter", game="Echo @ Foxtrot", odds=-200, player="Short Player", market="Hits"),
                row(pick="Longer Minus", game="Golf @ Hotel", odds=-174, player="Long Player", market="Hits"),
                row(pick="Plus Price", game="India @ Juliet", odds=150, player="Plus Player", market="Strikeouts"),
            ],
        },
    )
    streak = block["shapes"]["streak_candidate"]
    assert sorted(leg["oddsAmerican"] for leg in streak) == [-174, 150]
    assert {leg["pick"] for leg in streak} == {"Longer Minus", "Plus Price"}
    assert all(leg["units"] == 0 for leg in streak)
    assert "Team Plus" not in shape_picks(block, "streak_candidate")
    assert "Team Plus" in shape_picks(block, "priced_side")
    assert "Longer Minus" not in shape_picks(block, "priced_side")


def test_empty_input_yields_empty_shapes():
    block = track.build_track_research(DATE, None, None, now=NOW)
    assert set(block["shapes"]) == set(track.SHAPE_IDS)
    assert all(block["shapes"][shape] == [] for shape in track.SHAPE_IDS)
    assert block["units"] == 0
    assert block["quoteBasis"] == "comparison"
    assert block["label"] == "research"

    empty_models = track.build_track_research(
        DATE,
        models_payload({}),
        models_payload({}),
        now=NOW,
    )
    assert all(empty_models["shapes"][shape] == [] for shape in track.SHAPE_IDS)


def test_files_before_cutover_are_not_rewritten_with_track_research(tmp_path: Path, monkeypatch):
    assert track.TRACK_RESEARCH_CUTOVER_DATE == "2026-10-12"
    model_dir = tmp_path / "model_cache"
    props_dir = tmp_path / "props_cache"
    cards_dir = tmp_path / "parlay_cards"
    for directory in (model_dir, props_dir, cards_dir):
        directory.mkdir()

    old_date = "2026-10-11"
    old_pick = row(pick="Old Side", game="Alpha @ Bravo", odds=-150, date=old_date)
    old_payload = models_payload({"mlb_new": [old_pick]}, old_date)
    (model_dir / f"{old_date}.json").write_text(json.dumps(old_payload), encoding="utf-8")
    (cards_dir / f"{old_date}.json").write_text(
        json.dumps({"date": old_date, "cards": [], "sentinel": "history"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(parlays, "MODEL_CACHE_DIR", model_dir)
    monkeypatch.setattr(parlays, "PLAYER_PROPS_CACHE_DIR", props_dir)
    monkeypatch.setattr(parlays, "PARLAY_CARDS_DIR", cards_dir)

    direct = parlays.build_parlay_payload(old_date, old_payload, None)
    parlays.rebuild_parlay_cards(date_iso=old_date)
    written = json.loads((cards_dir / f"{old_date}.json").read_text(encoding="utf-8"))
    assert "trackResearch" not in written
    assert written["cards"] == direct["cards"]
    assert written["rankings"] == direct["rankings"]


def test_combined_price_is_labeled_an_estimate():
    block = research_from({
        "mlb_new": [
            row(pick="Heavy A", game="Alpha @ Bravo", odds=-900),
            row(pick="Heavy B", game="Charlie @ Delta", odds=-900),
        ],
    })
    card = block["shapes"]["favorite_parlay"][0]
    assert card["combinedPriceLabel"] == "estimate"
    assert card["executableParlayPrice"] is False
    assert card["quoteBasis"] == "comparison"
    assert block["quoteBasis"] == "comparison"
    assert "estimate" in json.dumps(card["combinedPriceLabel"])


def test_units_are_zero():
    block = research_from(
        {
            "mlb_new": [
                row(pick="Heavy A", game="Alpha @ Bravo", odds=-900, units=5),
                row(pick="Heavy B", game="Charlie @ Delta", odds=-900, units=5),
            ],
        },
        {
            "mlb_player_props": [
                row(pick="Jane Hits", game="Echo @ Foxtrot", odds=-140, player="Jane Hitter", market="Hits", units=5),
                row(pick="Bob Ks", game="Golf @ Hotel", odds=-130, player="Bob Pitcher", market="Strikeouts", units=5),
            ],
        },
    )
    assert block["units"] == 0
    blob = json.dumps(block)
    assert "balance" not in blob.lower()
    assert "bankroll" not in blob.lower()
    assert "$" not in blob
    assert "onyx" not in blob.lower()
    for card in (
        *block["shapes"]["favorite_parlay"],
        *block["shapes"]["prop_pair"],
    ):
        assert card["units"] == 0
        assert all(leg["units"] == 0 for leg in card["legs"])
    assert all(leg["units"] == 0 for leg in all_legs(block))


def test_parlay_script_runs_as_workflows_invoke_it(tmp_path: Path):
    # Workflows run `python scripts/build_parlay_cards.py`, which puts
    # scripts/ rather than the repo root on sys.path. The lazy track-research
    # import must still resolve (regression: ModuleNotFoundError in CI).
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    source_dir = Path(parlays.__file__).parent
    for name in ("build_parlay_cards.py", "build_track_cards.py"):
        shutil.copy(source_dir / name, scripts_dir / name)
    result = subprocess.run(
        [sys.executable, "scripts/build_parlay_cards.py", "--date", "2026-10-11"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr

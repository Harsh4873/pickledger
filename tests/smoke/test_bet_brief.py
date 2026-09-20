import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.generate_bet_brief import (  # noqa: E402
    brief_markdown,
    generate,
    hard_exclusion_reason,
)


def cand(**over):
    base = {
        "id": over.get("id", "profit-test-1"),
        "sport": "NFL",
        "source": "NFL Model",
        "sourceKey": "nfl_model",
        "market": "totals",
        "pick": "Under 43.5 (CAR @ ATL)",
        "game": "CAR @ ATL",
        "line": 43.5,
        "oddsAmerican": -110,
        "decision": "BET",
        "tier": "watch",
        "lane": None,
        "stakeUnits": 0,
        "price": {
            "source": "nflverse_posted_lines",
            "timestamp": "2026-09-20T11:45:00Z",
            "tier": "A",
            "freshPregame": True,
            "observedExecutable": True,
            "breakEvenProbability": 0.52381,
            "startTime": "2026-09-20T17:00Z",
        },
        "estimate": {
            "expectedValue": 0.01,
            "conservativeExpectedValue": -0.05,
            "probabilityPositiveEv": 0.4,
            "value": {
                "expectedValue": 0.01,
                "conservativeExpectedValue": -0.05,
                "probabilityPositiveEv": 0.4,
            },
        },
        "blockers": [],
    }
    base.update(over)
    return base


def profit_doc(candidates, sources=None, date="2026-09-20"):
    return {
        "date": date,
        "generatedAt": "2026-09-20T18:53:00Z",
        "engineVersion": "profit_desk_v2_live",
        "candidates": candidates,
        "sources": sources or [],
        "summary": {
            "candidateCount": len(candidates),
            "edgeQualified": 0,
            "valueQualified": 0,
            "liveQualified": 0,
            "liveRecordToDate": {"wins": 1, "losses": 1, "pushes": 0, "netUnits": 0.0, "roi": 0.0, "settled": 2},
        },
        "notices": [],
    }


ASOF_AM = dt.datetime(2026, 9, 20, 14, 0, tzinfo=dt.timezone.utc)
MISSING_LEDGER = ROOT / "data" / "does-not-exist-ledger.json"


def test_no_tennis_ever():
    t = cand(id="profit-tennis", sport="Tennis", source="Tennis Model", pick="Player A ML", oddsAmerican=-110)
    assert hard_exclusion_reason(t) is not None
    assert "tennis" in hard_exclusion_reason(t).lower()
    brief = generate(profit_doc([t]), MISSING_LEDGER, "am", ASOF_AM)
    assert brief["betThis"] == []
    assert len(brief["pass"]) == 1


def test_juice_cap_blocks_without_huge_ev():
    weak = cand(id="profit-juice", oddsAmerican=-259)
    assert hard_exclusion_reason(weak) is not None
    assert "-200" in hard_exclusion_reason(weak)
    huge = cand(
        id="profit-huge",
        oddsAmerican=-259,
        estimate={
            "expectedValue": 0.08,
            "conservativeExpectedValue": 0.06,
            "probabilityPositiveEv": 0.85,
            "value": {"expectedValue": 0.08, "conservativeExpectedValue": 0.06, "probabilityPositiveEv": 0.85},
        },
    )
    assert hard_exclusion_reason(huge) is None


def test_avoid_tier_and_missing_price_blocked():
    avoided = cand(id="profit-avoid", tier="avoid")
    assert hard_exclusion_reason(avoided) is not None
    stale = cand(id="profit-stale")
    stale["price"] = dict(stale["price"], freshPregame=False)
    assert hard_exclusion_reason(stale) is not None
    no_odds = cand(id="profit-noodds", oddsAmerican=None)
    assert hard_exclusion_reason(no_odds) is not None


def test_started_games_off_board():
    late = dt.datetime(2026, 9, 20, 19, 0, tzinfo=dt.timezone.utc)
    brief = generate(profit_doc([cand()]), MISSING_LEDGER, "pm", late)
    assert brief["betThis"] == []
    assert "Off the board" in brief["pass"][0]["reason"]


def test_qualified_path_caps_at_two():
    quals = [
        cand(id=f"profit-q{i}", lane="value", stakeUnits=0.5, decision="LEAN",
             pick=f"Under 0.5 RBIs ({i})", sport="MLB", source="MLBPlayerProps")
        for i in range(3)
    ]
    brief = generate(profit_doc(quals), MISSING_LEDGER, "am", ASOF_AM)
    assert len(brief["betThis"]) == 2
    assert all(b["qualified"] for b in brief["betThis"])


def test_climb_fallback_prefers_nfl_over_mls_and_gates_proven_negative():
    nfl = cand(id="profit-nfl")
    mls = cand(id="profit-mls", sport="MLS", source="MLS Model", pick="Under 4.5 (SD @ MIA)",
               game="SD @ MIA", line=4.5, oddsAmerican=-145)
    sources = [
        {"source": "NFL Model", "sourceKey": "nfl_model", "wins": 0, "losses": 6,
         "flatRoi": -1.0, "samples": 6, "distinctDates": 1, "probabilityPositiveEv": 0.15},
        {"source": "MLS Model", "sourceKey": "mls_model", "wins": 62, "losses": 60,
         "flatRoi": -0.134, "samples": 122, "distinctDates": 17, "probabilityPositiveEv": 0.2},
    ]
    brief = generate(profit_doc([nfl, mls], sources), MISSING_LEDGER, "am", ASOF_AM)
    assert len(brief["betThis"]) == 1
    assert "CAR @ ATL" in brief["betThis"][0]["pick"]
    assert brief["betThis"][0]["qualified"] is False
    mls_only = generate(profit_doc([mls], sources), MISSING_LEDGER, "am", ASOF_AM)
    assert mls_only["betThis"] == []
    assert "proven negative" in mls_only["pass"][0]["reason"]


def test_lean_only_slate_sits_out():
    lean = cand(id="profit-lean", decision="LEAN", pick="Under 51.5 (WAS @ DAL)")
    brief = generate(profit_doc([lean]), MISSING_LEDGER, "am", ASOF_AM)
    assert brief["betThis"] == []


def test_today_brief_matches_morning_context():
    profit = json.loads((ROOT / "data" / "profit_desk" / "latest.json").read_text())
    assert profit.get("date") == "2026-09-20"
    brief = generate(profit, ROOT / "data" / "personal_ledger.json", "am", ASOF_AM)
    assert len(brief["betThis"]) == 1
    top = brief["betThis"][0]
    assert top["pick"] == "Under 43.5 (CAR @ ATL)"
    assert top["oddsAmerican"] == -110
    assert top["qualified"] is False
    picks = [p["pick"] for p in brief["pass"]]
    assert "Under 51.5 (WAS @ DAL)" in picks
    assert len(brief["betThis"]) <= 2
    for b in brief["betThis"]:
        assert "tennis" not in str(b.get("sport", "")).lower()
    md = brief_markdown(brief)
    assert "Under 43.5 (CAR @ ATL)" in md
    assert "\u2014" not in md
    assert "\u2014" not in json.dumps(brief)

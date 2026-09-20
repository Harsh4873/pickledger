import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.personal_ledger import (  # noqa: E402
    american_profit,
    load_ledger,
    main,
    summarize,
)


def test_american_profit_math():
    assert american_profit(2.0, -110, "win") == 1.82
    assert american_profit(2.0, 120, "win") == 2.40
    assert american_profit(5.0, 259, "win") == 12.95
    assert american_profit(2.0, -110, "loss") == -2.0
    assert american_profit(2.0, 120, "loss") == -2.0
    assert american_profit(2.0, -110, "push") == 0.0
    assert american_profit(2.0, -110, "void") == 0.0


def test_seed_summary_matches_chat():
    ledger = load_ledger(ROOT / "data" / "personal_ledger.json")
    summary = summarize(ledger)
    by_book = {b["book"]: b for b in summary["books"]}
    assert by_book["Novig"]["runningBankroll"] == 50.00
    assert by_book["Novig"]["pendingRisk"] == 2.00
    assert by_book["Onyx"]["runningBankroll"] == 15.00
    assert by_book["Onyx"]["settledProfitSinceSnapshot"] == -5.00
    assert by_book["ReBet"]["runningBankroll"] == 20.00
    assert by_book["ReBet"]["status"] == "hold"
    assert by_book["Fliff"]["status"] == "out"
    assert by_book["Fliff"]["runningBankroll"] is None
    assert summary["openCount"] == 1
    assert summary["settledCount"] == 2
    assert summary["open"][0]["oddsAmerican"] is None
    assert "\u2014" not in json.dumps(summary)


def test_add_blocked_for_out_book(tmp_path):
    dest = tmp_path / "ledger.json"
    dest.write_text((ROOT / "data" / "personal_ledger.json").read_text())
    code = main(["--ledger", str(dest), "add", "--date", "2026-09-21",
                 "--book", "Fliff", "--sport", "NFL", "--selection", "X",
                 "--odds", "-110", "--stake", "1.0"])
    assert code == 1


def test_add_gated_for_hold_book(tmp_path):
    dest = tmp_path / "ledger.json"
    dest.write_text((ROOT / "data" / "personal_ledger.json").read_text())
    code = main(["--ledger", str(dest), "add", "--date", "2026-09-21",
                 "--book", "ReBet", "--sport", "NFL", "--selection", "X",
                 "--odds", "-110", "--stake", "1.0"])
    assert code == 1
    code = main(["--ledger", str(dest), "add", "--date", "2026-09-21",
                 "--book", "ReBet", "--sport", "NFL", "--selection", "X",
                 "--odds", "-110", "--stake", "1.0", "--allow-hold"])
    assert code == 0


def test_settle_requires_odds_for_win(tmp_path):
    dest = tmp_path / "ledger.json"
    dest.write_text((ROOT / "data" / "personal_ledger.json").read_text())
    code = main(["--ledger", str(dest), "settle", "--id", "pl-20260920-003",
                 "--result", "win"])
    assert code == 1
    code = main(["--ledger", str(dest), "settle", "--id", "pl-20260920-003",
                 "--result", "win", "--odds", "-110"])
    assert code == 0
    ledger = load_ledger(dest)
    row = next(b for b in ledger["bets"] if b["id"] == "pl-20260920-003")
    assert row["status"] == "win"
    assert row["profitDollars"] == 1.82
    summary = summarize(ledger)
    by_book = {b["book"]: b for b in summary["books"]}
    assert by_book["Novig"]["runningBankroll"] == 51.82

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
    assert american_profit(1.0, -110, "win") == 0.91
    assert american_profit(1.0, 100, "win") == 1.0
    assert american_profit(2.0, -110, "loss") == -2.0
    assert american_profit(2.0, 120, "loss") == -2.0
    assert american_profit(2.0, -110, "push") == 0.0
    assert american_profit(2.0, -110, "void") == 0.0


def test_post_reset_summary_with_inferred_novig():
    ledger = load_ledger(ROOT / "data" / "personal_ledger.json")
    summary = summarize(ledger)
    by_book = {b["book"]: b for b in summary["books"]}
    assert by_book["Novig"]["snapshotBankroll"] == 30.00
    assert by_book["Novig"]["snapshotDate"] == "2026-09-19"
    assert by_book["Novig"]["settledProfitSinceSnapshot"] == -0.09
    assert by_book["Novig"]["pendingRisk"] == 2.00
    assert by_book["Novig"]["runningBankroll"] == 29.91
    assert by_book["Onyx"]["snapshotBankroll"] == 5.00
    assert by_book["Onyx"]["settledProfitSinceSnapshot"] == 0.00
    assert by_book["Onyx"]["runningBankroll"] == 5.00
    assert by_book["Onyx"]["lifetimeSettledProfit"] == -5.00
    assert by_book["ReBet"]["status"] == "active"
    assert by_book["ReBet"]["snapshotBankroll"] == 1.00
    assert by_book["ReBet"]["runningBankroll"] == 1.00
    assert by_book["Fliff"]["status"] == "active"
    assert by_book["Fliff"]["snapshotBankroll"] == 2.00
    assert by_book["Fliff"]["runningBankroll"] == 2.00
    assert by_book["Fliff"]["lifetimeSettledProfit"] == -2.00
    assert summary["openCount"] == 1
    assert summary["settledCount"] == 4
    assert summary["lifetimeSettledProfit"] == -7.09
    assert summary["open"][0]["id"] == "pl-20260919-003"
    assert summary["open"][0]["oddsAmerican"] == 221
    ids = {b["id"] for b in ledger["bets"]}
    assert "pl-20260919-001" in ids
    assert "pl-20260919-002" in ids
    assert "pl-20260919-003" in ids
    assert "pl-20260920-004" in ids
    assert "pl-20260920-005" in ids
    pre = {b["id"]: b.get("preSnapshot") for b in ledger["bets"]}
    assert pre["pl-20260919-001"] is True
    assert pre["pl-20260919-002"] is True
    assert "\u2014" not in json.dumps(summary)
    assert "\u2014" not in json.dumps(ledger)


def test_presnapshot_excluded_from_running():
    ledger = load_ledger(ROOT / "data" / "personal_ledger.json")
    summary = summarize(ledger)
    by_book = {b["book"]: b for b in summary["books"]}
    assert by_book["Onyx"]["settledProfitSinceSnapshot"] == 0.00
    assert by_book["Onyx"]["runningBankroll"] == 5.00
    assert by_book["Fliff"]["settledProfitSinceSnapshot"] == 0.00
    assert by_book["Fliff"]["runningBankroll"] == 2.00


def test_daily_refresh_books_accept_tickets(tmp_path):
    dest = tmp_path / "ledger.json"
    dest.write_text((ROOT / "data" / "personal_ledger.json").read_text())
    for book in ("ReBet", "Fliff"):
        code = main(["--ledger", str(dest), "add", "--date", "2026-09-21",
                     "--book", book, "--sport", "NFL", "--selection", "X",
                     "--odds", "-110", "--stake", "1.0"])
        assert code == 0


def test_out_and_hold_gates_still_work(tmp_path):
    dest = tmp_path / "ledger.json"
    ledger = json.loads((ROOT / "data" / "personal_ledger.json").read_text())
    for entry in ledger["books"]:
        if entry["book"] == "ReBet":
            entry["status"] = "hold"
        if entry["book"] == "Fliff":
            entry["status"] = "out"
    dest.write_text(json.dumps(ledger))
    code = main(["--ledger", str(dest), "add", "--date", "2026-09-21",
                 "--book", "Fliff", "--sport", "NFL", "--selection", "X",
                 "--odds", "-110", "--stake", "1.0"])
    assert code == 1
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
    code = main(["--ledger", str(dest), "add", "--date", "2026-09-21",
                 "--book", "Novig", "--sport", "NFL",
                 "--selection", "Under 99.5 (TEST)", "--stake", "1.0",
                 "--odds-unknown"])
    assert code == 0
    ledger = load_ledger(dest)
    new_id = sorted(b["id"] for b in ledger["bets"] if b["date"] == "2026-09-21")[-1]
    code = main(["--ledger", str(dest), "settle", "--id", new_id,
                 "--result", "win"])
    assert code == 1
    code = main(["--ledger", str(dest), "settle", "--id", new_id,
                 "--result", "win", "--odds", "-110"])
    assert code == 0
    ledger = load_ledger(dest)
    row = next(b for b in ledger["bets"] if b["id"] == new_id)
    assert row["status"] == "win"
    assert row["profitDollars"] == 0.91
    summary = summarize(ledger)
    by_book = {b["book"]: b for b in summary["books"]}
    assert by_book["Novig"]["runningBankroll"] == 30.82

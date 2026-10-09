"""Replay the October 8 listing containing real October 10 CFB previews."""

import copy
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import subprocess
from types import SimpleNamespace

from bs4 import BeautifulSoup
import pytest

from scripts.scrapers import sportytrader_scraper as st


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/provider_listings/sportytrader-cfb-later-dates.html"
THURSDAY_MATCHUP = "Sam Houston Bearkats @ Liberty Flames"


@pytest.fixture
def listing():
    html = FIXTURE.read_text()
    soup = BeautifulSoup(html, "html.parser")
    matchups = [" @ ".join(node.get_text(" ", strip=True) for node in card.select("span.font-semibold")[:2])
                for card in soup.select(".card")]
    body = soup.get_text("\n", strip=True)
    cards = st._extract_nfl_us_text_cards(body, st.SPORT_CONFIG["cfb"]["url"], matchups,
                                         expected_league=st.SPORT_CONFIG["cfb"]["league"])
    assert len(cards) == 3
    return html, body, cards, matchups


def replay(monkeypatch, capsys, listing, *, target="2026-10-08", matchups=None, failed_fallback=False):
    html, body, cards, _ = listing
    config = st.SPORT_CONFIG["cfb"]
    fallback = config["fallback_urls"][-1]
    monkeypatch.setitem(st.SPORT_CONFIG, "cfb", {**config, "fallback_urls": (fallback,) if failed_fallback else ()})
    page = SimpleNamespace(url=config["url"], content=lambda: html, close=lambda: None)
    browser = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(st, "sync_playwright", lambda: nullcontext(None))
    monkeypatch.setattr(st, "_launch_browser", lambda _: browser)
    monkeypatch.setattr(st, "_new_context", lambda _: SimpleNamespace(new_page=lambda: page))
    def load(_page, url):
        if url == fallback:
            raise RuntimeError("HTTP 503")
        return copy.deepcopy(cards), body
    monkeypatch.setattr(st, "_load_cards", load)
    args = ["scraper", "--sport", "cfb", "--date", target]
    for matchup in matchups or [THURSDAY_MATCHUP]:
        args.extend(["--expected-matchup", matchup])
    monkeypatch.setattr(st.sys, "argv", args)
    st.main()
    return capsys.readouterr().out


def test_later_previews_explain_healthy_empty_through_publication(monkeypatch, capsys, listing):
    import pickgrader_server as server
    from scripts.refresh_external_feeds import _record_feed_attempt, _split_provider_result
    from scripts.source_health import source_issues

    output = replay(monkeypatch, capsys, listing)
    reason = "no previews for 2026-10-08; 3 previews for later dates (next 2026-10-10)"
    assert reason in output
    assert "picks parsed" not in output
    assert server._provider_no_preview_evidence(output, "cfb", "2026-10-08")[0]["reason"] == reason
    monkeypatch.setattr(server, "_known_external_slate_matchups", lambda *_: [THURSDAY_MATCHUP])
    monkeypatch.setattr(server, "_save_external_feed_admin_docs", lambda *_: None)
    monkeypatch.setattr(server, "_subprocess_run", lambda command, **_: subprocess.CompletedProcess(command, 0, output, ""))
    result = server.run_sportytrader_scraper("2026-10-08", ["cfb"])
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["picks"] == []
    assert result["meta"]["zeroSlateSports"] == []
    assert result["meta"]["officialMatchupCounts"] == {"cfb": 1}
    now = "2026-10-08T23:39:00Z"
    bucket = _split_provider_result("sportytrader", result, "2026-10-08", ["cfb"], now)["sportytrader_cfb"]
    published = _record_feed_attempt({"ok": False, "lastError": "empty parse", "date": "2026-10-08"}, bucket, "2026-10-08", now)
    assert published["refreshStatus"] == "ok"
    assert reason in published["note"]
    assert published["meta"]["providerStatus"] == "no_previews_published"
    assert source_issues("sportytrader_cfb", published, "2026-10-08") == []


def test_listing_diagnostics_deduplicate_dom_and_text_cards(listing):
    cards = listing[2]
    assert st._other_date_preview_reason(cards * 2, datetime(2026, 10, 8), "cfb") == (
        "no previews for 2026-10-08; 3 previews for later dates (next 2026-10-10)"
    )
    assert "3 previews for earlier dates (latest 2026-10-10)" in st._other_date_preview_reason(cards, datetime(2026, 10, 11), "cfb")


@pytest.mark.parametrize("damage", ["datetime", "tip"])
def test_unparsed_cards_cannot_certify_absence(monkeypatch, capsys, listing, damage):
    import pickgrader_server as server
    listing[2][0][damage] = ""
    output = replay(monkeypatch, capsys, listing)
    assert not server._provider_no_preview_evidence(output, "cfb", "2026-10-08")
    assert "No SportyTrader CFB picks parsed" in output


def test_current_date_nonmatching_cards_stay_a_failure(monkeypatch, capsys, listing):
    import pickgrader_server as server
    from scripts.refresh_external_feeds import _split_provider_result
    output = replay(monkeypatch, capsys, listing, target="2026-10-10")
    assert not server._provider_no_preview_evidence(output, "cfb", "2026-10-10")
    monkeypatch.setattr(server, "_known_external_slate_matchups", lambda *_: [THURSDAY_MATCHUP])
    monkeypatch.setattr(server, "_save_external_feed_admin_docs", lambda *_: None)
    monkeypatch.setattr(server, "_subprocess_run", lambda command, **_: subprocess.CompletedProcess(command, 0, output, ""))
    result = server.run_sportytrader_scraper("2026-10-10", ["cfb"])
    bucket = _split_provider_result("sportytrader", result, "2026-10-10", ["cfb"], "2026-10-10T12:00:00Z")["sportytrader_cfb"]
    assert bucket["ok"] is False
    assert "empty parse" in bucket["error"]


def test_failed_fallback_cannot_certify_absence(monkeypatch, capsys, listing):
    import pickgrader_server as server
    output = replay(monkeypatch, capsys, listing, failed_fallback=True)
    assert "3 previews for later dates" in output
    assert not server._provider_no_preview_evidence(output, "cfb", "2026-10-08")


def test_saturday_picks_are_unchanged(monkeypatch, capsys, listing):
    cards, matchups = listing[2:]
    assert st._other_date_preview_reason(cards, datetime(2026, 10, 10), "cfb") == ""
    assert st._extract_rows(cards, datetime(2026, 10, 10), "cfb", matchups) == cards
    output = replay(monkeypatch, capsys, listing, target="2026-10-10", matchups=matchups)
    assert output.count("Match:") == 3
    assert "Provider status:" not in output
    for card in cards:
        assert card["tip"] in output

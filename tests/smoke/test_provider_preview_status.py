from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from scripts.scrapers import sportsgambler_scraper as sg
from scripts.scrapers import sportytrader_scraper as st


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "provider_listings"
DAY = "2026-10-08"
NBA_MATCHUP = "Boston Celtics @ Cleveland Cavaliers"
CFB_MATCHUP = "Sam Houston Bearkats @ Liberty Flames"


def _html(name):
    return (FIXTURES / f"{name}.html").read_text()


@pytest.mark.parametrize("sport,fixture,url", [
    ("nba", "sportytrader-nba-0", st.SPORT_CONFIG["nba"]["url"]),
    ("nba", "sportytrader-nba-1", st.SPORT_CONFIG["nba"]["fallback_urls"][0]),
    ("cfb", "sportytrader-cfb-0", st.SPORT_CONFIG["cfb"]["url"]),
])
def test_sportytrader_live_empty_notice_requires_dedicated_listing(sport, fixture, url):
    html = _html(fixture)
    assert st._unpublished_listing_reason(html, url, sport)
    assert not st._unpublished_listing_reason(html, st.BASKETBALL_LISTING_URL, sport)
    soup = BeautifulSoup(html, "html.parser")
    soup.select_one('[data-trans="tips.no.tips.variable"]').decompose()
    assert not st._unpublished_listing_reason(str(soup), url, sport)


def test_sportytrader_http_error_is_not_a_loaded_empty_listing():
    page = SimpleNamespace(goto=lambda *_args, **_kwargs: SimpleNamespace(status=404))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        st._load_cards(page, st.SPORT_CONFIG["cfb"]["url"])
    assert not any("/en/betting-tips/american-football/usa/ncaa/" in url
                   for url in st.SPORT_CONFIG["cfb"]["fallback_urls"])


@pytest.mark.parametrize("sport,matchup", [("nba", NBA_MATCHUP), ("cfb", CFB_MATCHUP)])
@pytest.mark.parametrize("failed_fallback", [False, True])
def test_sportytrader_cli_emits_evidence_only_after_successful_coverage(
    monkeypatch, capsys, sport, matchup, failed_fallback,
):
    html = _html(f"sportytrader-{sport}-0")
    config = st.SPORT_CONFIG[sport]
    fallback = config["fallback_urls"][-1]
    monkeypatch.setitem(st.SPORT_CONFIG, sport, {
        **config, "fallback_urls": (fallback,) if failed_fallback else (),
    })
    page = SimpleNamespace(url=config["url"], content=lambda: html, close=lambda: None)
    browser = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(st, "sync_playwright", lambda: nullcontext(None))
    monkeypatch.setattr(st, "_launch_browser", lambda _pw: browser)
    monkeypatch.setattr(st, "_new_context", lambda _browser: SimpleNamespace(new_page=lambda: page))

    def load(_page, url):
        if url == fallback:
            raise RuntimeError("HTTP 503")
        return [], BeautifulSoup(html, "html.parser").get_text("\n", strip=True)

    monkeypatch.setattr(st, "_load_cards", load)
    monkeypatch.setattr(st.sys, "argv", ["scraper", "--sport", sport, "--date", DAY,
                                        "--expected-matchup", matchup])
    st.main()
    output = capsys.readouterr().out
    import pickgrader_server as server
    evidence = server._provider_no_preview_evidence(output, sport, DAY)
    assert bool(evidence) is not failed_fallback
    assert "officialMatchupCards=0" in output


def _replay_sportsgambler(monkeypatch, *, nba_html=None, fallback_status=200):
    calls = []
    pages = dict(zip(sg.NBA_URLS, [_html("sportsgambler-nba-0") if nba_html is None else nba_html,
                                 _html("sportsgambler-nba-1")]))

    def get(url, **_kwargs):
        calls.append(url)
        assert url in pages, f"unexpected detail fetch: {url}"
        return SimpleNamespace(text=pages[url], status_code=200 if url == sg.NBA_URL else fallback_status)

    monkeypatch.setattr(sg.requests, "get", get)
    return calls


def test_sportsgambler_live_expired_previews_certify_no_current_nba_previews(monkeypatch):
    calls = _replay_sportsgambler(monkeypatch)
    diagnostics = {}
    assert sg.scrape_nba(date.fromisoformat(DAY), [NBA_MATCHUP], diagnostics=diagnostics) == []
    assert calls == list(sg.NBA_URLS)
    assert diagnostics["status"] == "no_previews_published"
    assert diagnostics["date"] == DAY
    assert "All 10 NBA previews are marked Expired" in diagnostics["evidence"][0]["reason"]


def test_sportsgambler_expired_preview_cannot_match_a_later_weekday():
    soup = BeautifulSoup(_html("sportsgambler-nba-0"), "html.parser")
    expected = sg._expected_matchup_whitelist(["New York Knicks @ San Antonio Spurs"])
    # The captured first preview says Saturday; a later Saturday must not reuse it.
    assert sg._collect_pml_articles(soup, "NBA", expected, None, set(), date(2026, 10, 10)) == []


@pytest.mark.parametrize("damage", ["missing_control", "missing_rows", "blocked_fallback"])
def test_sportsgambler_unexplained_or_incomplete_listing_stays_unverified(monkeypatch, damage):
    soup = BeautifulSoup(_html("sportsgambler-nba-0"), "html.parser")
    if damage == "missing_control":
        soup.select_one("a.read_tip").decompose()
    elif damage == "missing_rows":
        for row in soup.select(".pml-row"):
            row.decompose()
    _replay_sportsgambler(monkeypatch, nba_html=str(soup), fallback_status=503 if damage == "blocked_fallback" else 200)
    diagnostics = {}
    assert sg.scrape_nba(date.fromisoformat(DAY), [NBA_MATCHUP], diagnostics=diagnostics) == []
    assert diagnostics == {}


def test_sportsgambler_cli_preserves_live_listing_evidence(monkeypatch, capsys):
    _replay_sportsgambler(monkeypatch)
    monkeypatch.setattr(sg.sys, "argv", ["scraper", "--sport", "nba", "--date", DAY,
                                        "--expected-matchup", NBA_MATCHUP])
    with pytest.raises(SystemExit) as exc:
        sg.main()
    assert exc.value.code == 0
    import pickgrader_server as server
    assert server._provider_no_preview_evidence(capsys.readouterr().out, "nba", DAY)


@pytest.mark.parametrize("provider,sport,matchup", [
    ("sportytrader", "nba", NBA_MATCHUP),
    ("sportytrader", "cfb", CFB_MATCHUP),
    ("sportsgambler", "nba", NBA_MATCHUP),
])
def test_verified_no_preview_status_survives_runner_publication_and_health(monkeypatch, provider, sport, matchup):
    import pickgrader_server as server
    from scripts.refresh_external_feeds import _record_feed_attempt, _split_provider_result
    from scripts.source_health import source_issues

    if provider == "sportytrader":
        url = st.SPORT_CONFIG[sport]["url"]
        reason = st._unpublished_listing_reason(_html(f"sportytrader-{sport}-0"), url, sport)
        status = dict(status="no_previews_published", sport=sport, date=DAY,
                      evidence=[dict(url=url, reason=reason)])
    else:
        _replay_sportsgambler(monkeypatch)
        status = {}
        sg.scrape_nba(date.fromisoformat(DAY), [matchup], diagnostics=status)
    monkeypatch.setattr(server, "_known_external_slate_matchups", lambda *_args: [matchup])
    monkeypatch.setattr(server, "_save_external_feed_admin_docs", lambda *_args: None)
    monkeypatch.setattr(server, "_subprocess_run", lambda command, **_kwargs: subprocess.CompletedProcess(
        command, 0, stdout="No picks found.\nProvider status: " + json.dumps(status), stderr="",
    ))
    result = getattr(server, f"run_{provider}_scraper")(DAY, [sport])
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["picks"] == []
    assert result["meta"]["officialMatchupCounts"][sport] == 1
    assert result["meta"]["zeroSlateSports"] == []
    key = f"{provider}_{sport}"
    now = f"{DAY}T12:00:00Z"
    bucket = _split_provider_result(provider, result, DAY, [sport], now)[key]
    prior = {"ok": False, "refreshStatus": "error", "lastError": "empty parse", "date": DAY}
    published = _record_feed_attempt(prior, bucket, DAY, now)
    assert published["refreshStatus"] == "ok"
    assert published["meta"]["providerStatus"] == "no_previews_published"
    assert "No previews published" in published["note"]
    assert source_issues(key, published, DAY) == []


@pytest.mark.parametrize("change", [{"date": "2026-10-07"}, {"sport": "cfb"}, {"evidence": []}])
def test_provider_status_requires_matching_sport_date_and_evidence(change):
    import pickgrader_server as server
    status = dict(status="no_previews_published", sport="nba", date=DAY,
                  evidence=[dict(url=sg.NBA_URL, reason="All previews are marked Expired.")])
    status.update(change)
    assert not server._provider_no_preview_evidence("Provider status: " + json.dumps(status), "nba", DAY)

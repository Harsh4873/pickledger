from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _forebet_row(
    home: str,
    away: str,
    sign: str,
    probs: str,
    odds: tuple[str, str, str],
) -> str:
    prob_spans = " ".join(f"<span>{p}</span>" for p in probs.split())
    odds_spans = "".join(f"<span>{o}</span>" for o in (*odds, "no", "no", "no"))
    return f"""
    <div class="rcnt tr_0">
      <div class="stcn"><a>Us1</a></div>
      <div class="tnms"><div itemscope itemtype="http://schema.org/SportsEvent">
        <a class="tnmscn" href="/en/football/matches/{home.lower().replace(' ', '-')}-{away.lower().replace(' ', '-')}-2465432">
          <span class="homeTeam" itemprop="homeTeam"><span itemprop="name">{home}</span></span>
          <span class="awayTeam" itemprop="awayTeam"><span itemprop="name">{away}</span></span>
          <time datetime="2026-07-23" itemprop="startDate"><span class="date_bah">23/07/2026 01:30</span></time>
        </a>
      </div></div>
      <div class="fprc">{prob_spans}</div>
      <div class="predict"><span class="forepr"><span>{sign}</span></span>
        <span class="scrmobpred ex_sc">1 <span class="scrmobpreddash">-</span> 2</span></div>
      <div class="ex_sc tabonly">1 - 2</div>
      <div class="avg_sc tabonly">2.69</div>
      <div class="bigOnly prmod"><span class="lscrsp">+220</span><div class="haodd">{odds_spans}</div></div>
      <div class="lscr_td lResTdSmall"><a>PRE VIEW</a></div>
      <div class="lmin_td smallFontTd lbord"></div>
    </div>
    """


FIXTURE_HTML = "<html><body><div class='schema'>" + "".join(
    (
        # Away-win tip; Forebet name drifts from the ESPN slate name.
        _forebet_row("Philadelphia Union", "New York Red Bulls", "2", "18 30 52", ("-143", "+310", "+320")),
        # Draw tip.
        _forebet_row("Columbus Crew", "New York City FC", "X", "32 37 31", ("-105", "+270", "+260")),
        # Tip with no posted odds.
        _forebet_row("Charlotte FC", "Atlanta United", "1", "61 22 17", ("no", "no", "no")),
        # Alias cases: ESPN says "LA Galaxy" / "St. Louis CITY SC".
        _forebet_row("Los Angeles Galaxy", "Saint Louis City", "2", "24 37 39", ("-105", "+270", "+250")),
        # Not on the official slate; must be excluded.
        _forebet_row("Volta Redonda", "Sao Goncalo RJ", "1", "44 34 22", ("+100", "+220", "+230")),
    )
) + "</div></body></html>"

SLATE = [
    {"away": "Red Bull New York", "home": "Philadelphia Union", "start_time": "2026-07-23T00:30Z"},
    {"away": "New York City FC", "home": "Columbus Crew", "start_time": "2026-07-23T00:30Z"},
    {"away": "Atlanta United FC", "home": "Charlotte FC", "start_time": "2026-07-23T01:15Z"},
    {"away": "St. Louis CITY SC", "home": "LA Galaxy", "start_time": "2026-07-23T03:30Z"},
    {"away": "Seattle Sounders FC", "home": "Austin FC", "start_time": "2026-07-23T01:30Z"},
]


def _module():
    return _load_module("forebet_scraper_test", ROOT / "scripts" / "scrapers" / "forebet_scraper.py")


def test_forebet_mls_config():
    module = _module()
    config = module.SPORT_CONFIG["mls"]
    assert config["espn_league"] == "usa.1"
    assert config["listing_url"].endswith("/en/football-tips-and-predictions-for-usa/mls")
    assert config["source"] == "ForebetMLS"
    assert config["label"] == "MLS"


def test_parse_forebet_rows_decodes_div_layout():
    module = _module()
    rows = module.parse_forebet_rows(FIXTURE_HTML)
    assert len(rows) == 5
    first = rows[0]
    assert (first["home"], first["away"], first["sign"]) == ("Philadelphia Union", "New York Red Bulls", "2")
    assert (first["prob_home"], first["prob_draw"], first["prob_away"]) == (18, 30, 52)
    assert (first["odds_home"], first["odds_draw"], first["odds_away"]) == (-143, 310, 320)
    assert first["predicted_score"] == "1-2"
    assert first["avg_goals"] == 2.69
    # "PRE VIEW" score-cell text and the date cell must never read as scores.
    no_odds = rows[2]
    assert no_odds["odds_home"] is None and no_odds["odds_draw"] is None and no_odds["odds_away"] is None


def test_scrape_forebet_matches_official_slate_only(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "fetch_daily_matchups", lambda sport, date_iso, config=None: (SLATE, True))
    result = module.scrape_forebet("mls", "2026-07-22", html=FIXTURE_HTML)
    assert result["ok"] is True
    picks = {pick["tip"]: pick for pick in result["picks"]}

    away_ml = picks["Red Bull New York ML"]
    assert away_ml["source"] == "ForebetMLS"
    assert away_ml["sport"] == "MLS"
    assert away_ml["odds"] == 320
    assert away_ml["probability"] == 0.52
    assert away_ml["decision"] == "BET"
    assert away_ml["units"] == 1
    assert away_ml["market_type"] == "soccer_moneyline"
    assert away_ml["grade_supported"] is True
    assert away_ml["calibration_excluded"] is True
    assert away_ml["matchup"] == "Red Bull New York @ Philadelphia Union"
    assert away_ml["start_time"] == "2026-07-23T00:30Z"
    assert (away_ml["forebet_prob_home"], away_ml["forebet_prob_draw"], away_ml["forebet_prob_away"]) == (18, 30, 52)

    draw = picks["Draw"]
    assert draw["pick"] == "Draw (New York City FC @ Columbus Crew)"
    assert draw["odds"] == 270
    assert draw["market_type"] == "soccer_standard"
    assert draw["grade_supported"] is True

    no_odds = picks["Charlotte FC ML"]
    assert no_odds["odds"] is None
    assert no_odds["probability"] == 0.61

    # Team-name drift resolves through the shared alias table.
    assert picks["St. Louis CITY SC ML"]["odds"] == 250

    # The off-slate Brazilian match is excluded; the unmatched slate game is reported.
    assert result["meta"]["matchedPicks"] == 4
    assert result["meta"]["officialMatchups"] == 5
    assert result["meta"]["unpublishedMatchups"] == ["Seattle Sounders FC @ Austin FC"]
    assert all("Volta Redonda" not in pick["pick"] for pick in result["picks"])


def test_scrape_forebet_fails_closed_when_slate_unresolved(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "fetch_daily_matchups", lambda sport, date_iso, config=None: ([], False))
    result = module.scrape_forebet("mls", "2026-07-22", html=FIXTURE_HTML)
    assert result["ok"] is False
    assert result["picks"] == []
    assert "could not resolve" in result["error"]


def test_scrape_forebet_confirmed_off_day_skips_listing_fetch(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "fetch_daily_matchups", lambda sport, date_iso, config=None: ([], True))

    def _no_fetch(url):
        raise AssertionError("listing fetch must not run on a confirmed off-day")

    monkeypatch.setattr(module, "_fetch_listing_html", _no_fetch)
    result = module.scrape_forebet("mls", "2026-07-20")
    assert result["ok"] is True
    assert result["picks"] == []
    assert result["meta"]["officialMatchups"] == 0


def test_forebet_mls_is_registered_across_the_pipeline():
    refresh = _load_module("refresh_external_feeds_test", ROOT / "scripts" / "refresh_external_feeds.py")
    assert "forebet_mls" in refresh.FEED_RUNNERS
    assert "forebet_mls" not in refresh.SPLIT_PROVIDER_FEEDS

    merge = _load_module("merge_external_feed_test", ROOT / "scripts" / "merge_external_feed_cache_payload.py")
    assert "forebet_mls" in merge.EXTERNAL_FEED_MODEL_KEYS

    calibration = _load_module("pick_calibration_test", ROOT / "scripts" / "pick_calibration.py")
    assert "forebet_mls" in calibration.CALIBRATION_EXCLUDED_MODEL_KEYS

    assert "forebet_mls: 'ForebetMLS'" in (ROOT / "src" / "data.ts").read_text(encoding="utf-8")

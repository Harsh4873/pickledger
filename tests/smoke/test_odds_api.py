from scripts import odds_api


def test_cfb_sharp_fetch_requests_all_three_markets(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    captured = {}

    def fetch(url, params):
        captured.update({"url": url, "params": params})
        return []

    assert odds_api.fetch_sharp_events("CFB", fetch_json=fetch) == []
    assert captured["url"].endswith("/sports/americanfootball_ncaaf/odds")
    assert captured["params"]["markets"] == "h2h,spreads,totals"


def test_sharp_spread_price_matches_team_and_exact_line():
    pick = {
        "sport": "CFB",
        "market": "spread",
        "team": "Away Tech Owls",
        "pick": "Away Tech Owls +3.5",
        "line": 3.5,
    }
    event = {
        "bookmakers": [{
            "markets": [{
                "key": "spreads",
                "outcomes": [
                    {"name": "Home State Wildcats", "point": -3.5, "price": -112},
                    {"name": "Away Tech Owls", "point": 3.5, "price": -108},
                ],
            }],
        }],
    }
    priced = odds_api._sharp_price(pick, event)
    assert priced is not None
    odds, no_vig = priced
    assert odds == -108
    assert no_vig is not None
    assert 0.49 < no_vig < 0.51

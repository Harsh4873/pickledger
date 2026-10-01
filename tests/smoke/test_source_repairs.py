from datetime import date, datetime, timezone

import pytest

from MLBPredictionModel.observed_odds import fetch_mlb_market_odds_for_date
from scripts.source_health import source_issues
from scripts.model_versions import stamp_prediction_versions


def event(state="pre", start="2026-09-21T22:00Z", identity="1"):
    return {"id": identity, "date": start, "status": {"type": {"state": state}},
        "competitions": [{"competitors": [
            {"homeAway": side, "team": {"displayName": name}}
            for side, name in [("home", "Baltimore Orioles"), ("away", "Toronto Blue Jays")]],
            "odds": [{"provider": {"name": "DraftKings"}, "overUnder": 7.5,
                "total": {"over": {"close": {"odds": "-115"}}, "under": {"close": {"odds": "-105"}}}}]}]}


def test_mlb_observed_inputs_require_pregame_paired_prices_and_unambiguous_game():
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    def fetch(events):
        return fetch_mlb_market_odds_for_date(date(2026, 9, 21), fetch=lambda: {"events": events}, now=now)
    quote = fetch([event()])[('jays', 'orioles')]
    assert quote['total_line'] == 7.5 and quote['total_under_odds'] == -105
    assert quote['source'] == 'espn_scoreboard:DraftKings'
    assert not fetch([event('in')])
    assert not fetch([event(start='2026-09-21T11:00Z')])
    assert not fetch([event(), event('post', '2026-09-21T10:00Z', '2')])
    missing = event(); del missing['competitions'][0]['odds'][0]['total']['under']
    assert fetch([missing])[('jays', 'orioles')]['total_line'] is None


def test_source_health_distinguishes_partial_coverage_and_stale_features():
    assert source_issues('scores24_nfl', {'ok': True, 'meta': {'expectedMatchups': 14, 'matchedPicks': 8}}, '2026-09-20')
    issues = source_issues('tennis', {'ok': True, 'meta': {'ratingsThrough': '2026-07-20', 'unknownPlayers': 24, 'officialMatchups': 35}}, '2026-09-20')
    assert any('ratings stale' in s for s in issues)
    assert any('unrated' in s for s in issues)
    assert not source_issues('cfb', {'ok': True, 'games': 0, 'picks': []}, '2026-09-20')


def test_prediction_version_changes_with_artifact_and_not_probability(tmp_path):
    artifact = tmp_path / 'MLBPredictionModel/artifacts/model.json'
    artifact.parent.mkdir(parents=True); artifact.write_text('{"v":1}')
    payload = {'models': {'mlb_new': {'picks': [{'probability': .6}]}}}
    stamp_prediction_versions(payload, root=tmp_path)
    first = payload['models']['mlb_new']['prediction_model_version']
    payload['models']['mlb_new']['picks'][0]['probability'] = .7
    stamp_prediction_versions(payload, root=tmp_path)
    assert payload['models']['mlb_new']['prediction_model_version'] == first
    artifact.write_text('{"v":2}')
    stamp_prediction_versions(payload, root=tmp_path)
    assert payload['models']['mlb_new']['picks'][0]['prediction_model_version'] != first


def test_tennistonic_source_health_surfaces_paired_archive_lag():
    from scripts.source_health import source_issues

    green = source_issues(
        "tennistonic_tennis",
        {"ok": True, "date": "2026-09-20", "meta": {}, "picks": [{"id": "1"}]},
        "2026-09-20",
    )
    assert not any("archive stale" in issue for issue in green)

    lag = source_issues(
        "tennistonic_tennis",
        {
            "ok": True,
            "date": "2026-09-20",
            "meta": {"archiveThrough": "2026-07-20"},
            "picks": [{"id": "1"}],
        },
        "2026-09-20",
    )
    assert any("paired tennis archive stale" in issue for issue in lag)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_sportytrader_local_sync_calls_shared_refresh_and_restores_parser(monkeypatch, exit_code):
    import importlib.util
    from pathlib import Path
    from scripts import refresh_external_feeds as refresh

    monkeypatch.setenv("ENABLE_SPORTYTRADER_LOCALSYNC", "true")
    monkeypatch.setenv("SPORTYTRADER_DEBUG_ORPHAN", "false")
    path = Path(__file__).resolve().parents[2] / "scripts/scrapers/sportytrader_local_sync.py"
    spec = importlib.util.spec_from_file_location("sportytrader_sync_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_parser = refresh._parse_args

    class Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 1, 2, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(module, "datetime", Clock)
    calls = []
    def fake_refresh():
        calls.append(refresh._parse_args())
        return exit_code

    monkeypatch.setattr(refresh, "main", fake_refresh)
    assert module.main() == exit_code
    assert len(calls) == 1
    assert calls[0].date == "2026-09-30"
    assert calls[0].feeds == "sportytrader"
    assert "nfl" in calls[0].sports.split(",")
    assert calls[0].skip_firestore is True
    assert refresh._parse_args is original_parser

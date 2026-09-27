from copy import deepcopy
import json

from TennisPredictionModel.tennis_results import completed_matches, fetch_completed_matches


def payload():
    return {'events': [{'name': 'Known Open', 'groupings': [{'grouping': {'slug': 'womens-singles'},
        'competitions': [{'id': '1', 'date': '2026-09-18T18:00Z', 'status': {'type': {'name': 'STATUS_FINAL'}},
            'round': {'displayName': 'Final'}, 'competitors': [
                {'winner': won, 'athlete': {'displayName': name},
                 'linescores': [{'value': games, 'winner': won}, {'value': games, 'winner': won}]}
                for name, won, games in [('Player One', True, 6), ('Player Two', False, 3)]]}]}]}]}


def meta(*args):
    return {'surface': 'Hard', 'tier': 2, 'best_of': 3, 'court': 'Outdoor'}


def test_official_results_exclude_future_nonfinal_qualifying_and_guessed_surfaces():
    original = payload()
    rows = completed_matches(original, 'WTA', '2026-09-17', '2026-09-19', meta, lambda r: 7)
    match = next(iter(rows.values()))
    assert match.winner_games == 12 and match.loser_games == 6 and match.winner_sets == 2
    assert match.winner_rank is None and match.odds == {}
    assert not completed_matches(original, 'WTA', '2026-09-18', '2026-09-19', meta, lambda r: 7)
    assert not completed_matches(original, 'WTA', '2026-09-17', '2026-09-18', meta, lambda r: 7)
    for field, value in [('status', {'type': {'name': 'STATUS_RETIRED'}}), ('round', {'displayName': 'Qualifying Final'})]:
        broken = deepcopy(original); broken['events'][0]['groupings'][0]['competitions'][0][field] = value
        assert not completed_matches(broken, 'WTA', '2026-09-17', '2026-09-19', meta, lambda r: 7)
    assert not completed_matches(original, 'WTA', '2026-09-17', '2026-09-19', lambda *a: {'assumed': True}, lambda r: 7)


def test_evening_utc_rollover_results_use_the_chicago_match_day():
    evening = payload()
    evening['events'][0]['groupings'][0]['competitions'][0]['date'] = '2026-09-19T02:00:00Z'
    rows = completed_matches(evening, 'WTA', '2026-09-17', '2026-09-19', meta, lambda r: 7)
    assert len(rows) == 1
    assert next(iter(rows.values())).date == '2026-09-18'


def test_result_fallback_deduplicates_tournament_repeats_and_reports_failures(tmp_path):
    def fetch(url):
        if '/atp/' in url:
            raise RuntimeError('provider unavailable')
        return payload()
    rows, errors = fetch_completed_matches('2026-09-16', '2026-09-20', meta, lambda r: 7, fetch=fetch, cache_dir=tmp_path)
    assert len(rows) == 1 and len(errors) == 3


def test_incomplete_cached_scoreboard_is_refetched_and_failed_refresh_salvages_finals(tmp_path):
    stale = payload()
    stale['events'][0]['groupings'][0]['competitions'][0]['status'] = {'type': {'name': 'STATUS_IN_PROGRESS'}}
    cache_path = tmp_path / 'WTA-2026-09-18.json'
    cache_path.write_text(json.dumps(stale))
    calls = []

    def fetch(url):
        calls.append(url)
        return payload() if '/wta/' in url else {'events': []}

    rows, errors = fetch_completed_matches('2026-09-17', '2026-09-19', meta, lambda r: 7, fetch=fetch, cache_dir=tmp_path)
    assert len(rows) == 1 and errors == []
    assert any('/wta/' in url for url in calls)
    assert json.loads(cache_path.read_text())['events'][0]['groupings'][0]['competitions'][0]['status']['type']['name'] == 'STATUS_FINAL'

    calls.clear()
    fetch_completed_matches('2026-09-17', '2026-09-19', meta, lambda r: 7, fetch=fetch, cache_dir=tmp_path)
    assert not any('/wta/' in url for url in calls)

    partial = payload()
    ongoing = deepcopy(partial['events'][0]['groupings'][0]['competitions'][0])
    ongoing['id'] = '2'
    ongoing['status'] = {'type': {'name': 'STATUS_IN_PROGRESS'}}
    partial['events'][0]['groupings'][0]['competitions'].append(ongoing)
    cache_path.write_text(json.dumps(partial))

    def failed_fetch(url):
        if '/wta/' in url:
            raise TimeoutError('provider timed out')
        return {'events': []}

    rows, errors = fetch_completed_matches('2026-09-17', '2026-09-19', meta, lambda r: 7, fetch=failed_fetch, cache_dir=tmp_path)
    assert len(rows) == 1
    assert len(errors) == 1 and 'refresh failed' in errors[0]

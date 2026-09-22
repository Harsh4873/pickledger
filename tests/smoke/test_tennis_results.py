from copy import deepcopy

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


def test_result_fallback_deduplicates_tournament_repeats_and_reports_failures(tmp_path):
    def fetch(url):
        if '/atp/' in url:
            raise RuntimeError('provider unavailable')
        return payload()
    rows, errors = fetch_completed_matches('2026-09-16', '2026-09-20', meta, lambda r: 7, fetch=fetch, cache_dir=tmp_path)
    assert len(rows) == 1 and len(errors) == 3

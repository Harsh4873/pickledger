"""NBA team model across the season rollover: season labels, early-season priors, calibration."""
from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NBA_DIR = ROOT / "NBAPredictionModel"
if str(NBA_DIR) not in sys.path:
    sys.path.insert(0, str(NBA_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _record(games, wins, points_for, points_against):
    return {
        "gamesPlayed": float(games),
        "wins": float(wins),
        "losses": float(games - wins),
        "winPercent": wins / games if games else 0.0,
        "avgPointsFor": float(points_for),
        "avgPointsAgainst": float(points_against),
        "differential": float(points_for - points_against),
    }


def test_season_label_follows_the_modeled_date():
    from nba_season import espn_season_year, nba_season_label, previous_season_label

    assert nba_season_label("2026-11-07") == "2026-27"
    assert nba_season_label("2027-04-12") == "2026-27"
    assert nba_season_label("2027-08-01") == "2027-28"
    assert previous_season_label("2026-27") == "2025-26"
    assert espn_season_year("2026-27") == 2027


def test_early_season_record_leans_on_last_season():
    import live_data

    # Eight games into the season a strong team from last year is 2-6.
    prior = _record(82, 54, 120.0, 114.84)
    early = live_data.blend_prior_season(_record(8, 2, 110.0, 116.0), prior)
    weight = 8 / (8 + live_data.NBA_PRIOR_SEASON_GAMES)
    assert early["current_season_weight"] == pytest.approx(weight)
    expected_diff = weight * -6.0 + (1 - weight) * live_data.NBA_PRIOR_SEASON_CARRYOVER * 5.16
    assert early["point_diff"] == pytest.approx(expected_diff)
    assert -6.0 < early["point_diff"] < 5.16
    assert 0.25 < early["win_pct"] < 54 / 82
    # The scoring level stays this season's; only the margin is blended.
    assert early["points_for"] + early["points_against"] == pytest.approx(226.0)

    # Before opening night the prior carries everything.
    opener = live_data.blend_prior_season({"gamesPlayed": 0.0, "wins": 0.0, "losses": 0.0}, prior)
    assert opener["point_diff"] == pytest.approx(live_data.NBA_PRIOR_SEASON_CARRYOVER * 5.16)
    # Without a prior record the current season is used as is.
    alone = live_data.blend_prior_season(_record(8, 2, 110.0, 116.0), None)
    assert alone["point_diff"] == pytest.approx(-6.0)
    assert alone["win_pct"] == pytest.approx(0.25)


def test_espn_fallback_reads_this_and_last_season_for_the_modeled_date(monkeypatch):
    import live_data

    urls = []

    def fake_fetch(url):
        urls.append(url)
        if url.endswith("/teams"):
            return {"sports": [{"leagues": [{"teams": [{"team": {"id": "7", "abbreviation": "DEN", "name": "Nuggets", "displayName": "Denver Nuggets"}}]}]}]}
        if url.endswith("/statistics"):
            return {}
        season = "2026" if url.endswith("season=2026") else "2027"
        record = _record(82, 54, 120.0, 114.84) if season == "2026" else _record(7, 5, 118.0, 112.0)
        return {"team": {"record": {"items": [{"type": "total", "stats": [{"name": k, "value": v} for k, v in record.items()]}]}}}

    monkeypatch.setattr(live_data, "_fetch_espn_json", fake_fetch)
    stats = live_data.fetch_espn_team_stats_fallback([{"away_team": "Nuggets", "home_team": "Nuggets"}], season="2026-27")
    assert any(url.endswith("/den?season=2027") for url in urls)
    assert any(url.endswith("/den?season=2026") for url in urls)
    payload = stats["Nuggets"]
    weight = 7 / (7 + live_data.NBA_PRIOR_SEASON_GAMES)
    assert payload["current_season_weight"] == pytest.approx(weight)
    assert payload["net_rating"] == pytest.approx(weight * 6.0 + (1 - weight) * live_data.NBA_PRIOR_SEASON_CARRYOVER * 5.16)


def test_nba_api_failure_hands_the_modeled_season_to_espn(monkeypatch):
    import live_data

    seen = {}

    def blocked(**_kwargs):
        raise TimeoutError("stats.nba.com timed out")

    monkeypatch.setattr(live_data.leaguedashteamstats, "LeagueDashTeamStats", blocked)
    monkeypatch.setattr(live_data, "_pause", lambda: None)
    monkeypatch.setattr(live_data, "fetch_espn_team_stats_fallback", lambda games, season=None: seen.setdefault("season", season) and {"Nuggets": {}})
    live_data.fetch_all_team_stats(as_of_date="2026-11-07", upcoming_games=[{"away_team": "Nuggets", "home_team": "Pacers"}])
    assert seen["season"] == "2026-27"


def test_nba_api_payload_regresses_margins_and_records_toward_last_season():
    import live_data

    payload = {field: 12.0 for field in live_data._PRIOR_MARGIN_FIELDS}
    payload.update({field: 0.9 for field in live_data._PRIOR_WIN_FIELDS})
    payload.update({"off_rating": 124.0, "def_rating": 112.0})
    live_data.regress_payload_toward_prior(payload, 5.0, _record(82, 41, 115.0, 115.0))
    weight = 5 / (5 + live_data.NBA_PRIOR_SEASON_GAMES)
    assert payload["net_rating"] == pytest.approx(weight * 12.0)
    assert payload["recent_5_win_pct"] == pytest.approx(weight * 0.9 + (1 - weight) * 0.5)
    assert payload["off_rating"] - payload["def_rating"] == pytest.approx(weight * 12.0)
    assert payload["off_rating"] + payload["def_rating"] == pytest.approx(236.0)


def _team(name, is_home, net=0.0, win=0.5):
    from data_models import Team, TeamStats

    stats = TeamStats(
        net_rating=net, off_rating_10=112.0 + net / 2, def_rating_10=112.0 - net / 2, ts_pct=0.57,
        reb_pct=0.5, pace=99.0, last_10_win_pct=win, is_b2b_second_leg=False,
        is_3_in_4_nights=False, season_win_pct=win, rest_days=1.0, back_to_back_flag=False,
        recent_5_win_pct=win, recent_10_win_pct=win, weighted_win_pct=win,
        recent_5_point_diff=net, recent_10_point_diff=net, weighted_point_diff=net,
    )
    return Team(1 if is_home else 2, name, is_home, stats, [])


def test_relative_records_compare_both_teams():
    from probability_layers import calculate_layer1_base_rate

    home, away = _team("Nuggets", True, win=0.8), _team("Thunder", False, win=0.8)
    relative = calculate_layer1_base_rate(home, away, 0.5, relative_records=True)
    absolute = calculate_layer1_base_rate(home, away, 0.5)
    # Two .800 teams are an even matchup apart from home court.
    assert relative == pytest.approx(0.5 + 0.015)
    # The legacy form reads only the home record and calls it a big favourite.
    assert absolute > 0.65


def test_layer_sum_past_one_is_not_a_certainty():
    from probability_layers import LAYER_PROBABILITY_FLOOR, combine_home_win_probability

    home, away = _team("Warriors", True), _team("Magic", False)
    probability = combine_home_win_probability(1.12, 13.5, home, away)
    ceiling = 1 / (1 + math.exp(-(math.log((1 - LAYER_PROBABILITY_FLOOR) / LAYER_PROBABILITY_FLOOR) + 1.35 * 0.30)))
    assert probability == pytest.approx(ceiling)
    assert probability < 0.99


def _write_log(path, rows):
    import prediction_logging as pl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pl.PREDICTION_LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in pl.PREDICTION_LOG_FIELDS})


def test_platt_ignores_rows_from_the_overflowing_model(tmp_path):
    from calibration import PLATT_FIT_SINCE, fit_platt_scaler_from_log

    rng = random.Random(7)
    rows = []
    # 80 pre-fix rows with 99.9999% calls that lost about a third of the time.
    for i in range(80):
        rows.append({"game_date": "2026-10-30", "away_team": f"A{i}", "home_team": f"H{i}", "raw_probability": "0.999999", "actual_home_win": str(int(rng.random() < 0.66))})
    log_path = tmp_path / "predictions.csv"
    _write_log(log_path, rows)
    everything, _ = fit_platt_scaler_from_log(log_path=str(log_path), since="")
    assert everything.fitted and everything.calibrate(0.80) < 0.70
    scaler, diag = fit_platt_scaler_from_log(log_path=str(log_path))
    assert PLATT_FIT_SINCE > "2026-10-30"
    assert diag.sample_count == 0 and not scaler.fitted
    assert scaler.calibrate(0.80) == pytest.approx(0.80)


def test_platt_fit_stays_near_identity_on_a_small_calibrated_sample():
    from calibration import PlattScaler

    rng = random.Random(11)
    probabilities = [rng.uniform(0.2, 0.85) for _ in range(120)]
    outcomes = [int(rng.random() < p) for p in probabilities]
    scaler = PlattScaler().fit(probabilities, outcomes)
    assert 0.75 < scaler.a < 1.25
    # Home teams winning 57% of a sample does not get balanced away to 50%.
    balanced = PlattScaler().fit([0.57] * 400, [1] * 228 + [0] * 172)
    assert balanced.calibrate(0.57) == pytest.approx(0.57, abs=0.01)


def test_injury_absence_without_on_off_data_scales_with_minutes_and_rating_share(monkeypatch):
    import probability_layers as layers

    minutes = {
        "team_games": 8.0,
        "prior_team_games": 82.0,
        "current_weight": 8 / 23,
        "players": {
            "Paolo Banchero": {"gp": 2.0, "mpg": 34.0, "prior_gp": 72.0, "prior_mpg": 34.8},
            "Franz Wagner": {"gp": 8.0, "mpg": 36.0, "prior_gp": 34.0, "prior_mpg": 30.0},
            "Two Way": {"gp": 1.0, "mpg": 4.0, "prior_gp": 0.0, "prior_mpg": 0.0},
        },
    }
    monkeypatch.setattr(layers, "_fetch_team_on_off_data", lambda team, season: {})
    monkeypatch.setattr(layers, "_fetch_team_minutes", lambda team, season: minutes)
    monkeypatch.setattr(layers, "_build_roster_context", lambda team, season, data: ({}, {}))

    star, _ = layers.calculate_injury_adjustment("Magic", [{"name": "Wagner Franz Wagner", "status": "Out", "absence_probability": 1.0}], season="2026-27")
    long_out, _ = layers.calculate_injury_adjustment("Magic", [{"name": "Banchero Paolo Banchero", "status": "Out", "absence_probability": 1.0}], season="2026-27")
    bench, _ = layers.calculate_injury_adjustment("Magic", [{"name": "Two Way", "status": "Out", "absence_probability": 1.0}], season="2026-27")
    unknown, _ = layers.calculate_injury_adjustment("Magic", [{"name": "Nobody Here", "status": "Out", "absence_probability": 1.0}], season="2026-27")

    weight = 8 / 23
    assert star == pytest.approx(-0.05 * (weight * 1.0 + (1 - weight) * 34 / 82))
    # Out for six of eight games: most of this season's rating already lacks him.
    assert long_out == pytest.approx(-0.05 * (weight * 2 / 8 + (1 - weight) * 72 / 82))
    assert abs(bench) < 0.002
    assert unknown == 0.0

from __future__ import annotations

from copy import deepcopy


def _cfb_summary() -> dict:
    """A realistic, CATEGORY-GROUPED ESPN college-football game summary.

    ``boxscore.players`` is a list of two team blocks; each block's
    ``statistics`` is a list of category dicts. Each category has a ``keys``
    array and an ``athletes`` list whose ``stats`` strings are positionally
    aligned to ``keys``. Key order differs per category on purpose, so the
    extractor must resolve columns via ``keys.index(...)``.
    """
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "Texas A&M Aggies"},
                    "statistics": [
                        {
                            "name": "passing",
                            "keys": [
                                "completions/passingAttempts",
                                "passingYards",
                                "yardsPerPassAttempt",
                                "passingTouchdowns",
                                "interceptions",
                                "adjQBR",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"id": "111", "displayName": "Marcel Reed"},
                                    "stats": ["22/30", "312", "10.4", "3", "1", "88.1"],
                                }
                            ],
                        },
                        {
                            "name": "rushing",
                            "keys": [
                                "rushingAttempts",
                                "rushingYards",
                                "yardsPerRushAttempt",
                                "rushingTouchdowns",
                                "longRushing",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"id": "222", "displayName": "Le'Veon Moss"},
                                    "stats": ["18", "104", "5.8", "2", "31"],
                                }
                            ],
                        },
                        {
                            "name": "receiving",
                            "keys": [
                                "receptions",
                                "receivingYards",
                                "yardsPerReception",
                                "receivingTouchdowns",
                                "longReception",
                                "receivingTargets",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"id": "333", "displayName": "Le'Veon Moss"},
                                    "stats": ["3", "27", "9.0", "0", "12", "4"],
                                }
                            ],
                        },
                    ],
                },
                {
                    "team": {"displayName": "Notre Dame Fighting Irish"},
                    "statistics": [
                        {
                            "name": "passing",
                            "keys": [
                                "completions/passingAttempts",
                                "passingYards",
                                "yardsPerPassAttempt",
                                "passingTouchdowns",
                                "interceptions",
                                "adjQBR",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"id": "444", "displayName": "Riley Leonard"},
                                    "stats": ["15/24", "188", "7.8", "1", "0", "71.0"],
                                }
                            ],
                        }
                    ],
                },
            ]
        }
    }


def _prop_pick(**overrides) -> dict:
    pick = {
        "id": "cfb-prop-1",
        "sport": "CFB",
        "scope": "player",
        "player_name": "Marcel Reed",
        "stat_key": "passing_yards",
        "selection": "OVER",
        "line": 275.5,
    }
    pick.update(overrides)
    return pick


def test_cfb_passing_yards_prop_grades_win_and_loss():
    import pickgrader_server

    summary = _cfb_summary()

    over_win = pickgrader_server.grade_player_prop_pick(
        _prop_pick(selection="OVER", line=275.5), {}, deepcopy(summary)
    )
    assert over_win == "win"  # 312 > 275.5

    over_loss = pickgrader_server.grade_player_prop_pick(
        _prop_pick(selection="OVER", line=325.5), {}, deepcopy(summary)
    )
    assert over_loss == "loss"  # 312 < 325.5

    under_win = pickgrader_server.grade_player_prop_pick(
        _prop_pick(selection="UNDER", line=325.5), {}, deepcopy(summary)
    )
    assert under_win == "win"  # 312 < 325.5

    under_loss = pickgrader_server.grade_player_prop_pick(
        _prop_pick(selection="UNDER", line=275.5), {}, deepcopy(summary)
    )
    assert under_loss == "loss"  # 312 > 275.5


def test_cfb_exact_line_match_is_a_push():
    import pickgrader_server

    result = pickgrader_server.grade_player_prop_pick(
        _prop_pick(selection="OVER", line=312), {}, _cfb_summary()
    )
    assert result == "push"  # 312 == 312.0


def test_cfb_unknown_player_returns_pending():
    import pickgrader_server

    result = pickgrader_server.grade_player_prop_pick(
        _prop_pick(player_name="Nonexistent Quarterback"), {}, _cfb_summary()
    )
    assert result == "pending"


def test_cfb_rushing_and_receiving_stats_extract_by_category():
    import pickgrader_server

    summary = _cfb_summary()

    rush = pickgrader_server._extract_cfb_player_stat(summary, "Le'Veon Moss", "rushing_yards")
    assert rush == 104.0

    rec = pickgrader_server._extract_cfb_player_stat(summary, "Le'Veon Moss", "receiving_yards")
    assert rec == 27.0

    combo = pickgrader_server._extract_cfb_player_stat(
        summary, "Le'Veon Moss", "rushing_receiving_yards"
    )
    assert combo == 131.0  # 104 + 27

    pass_tds = pickgrader_server._extract_cfb_player_stat(
        summary, "Marcel Reed", "passing_touchdowns"
    )
    assert pass_tds == 3.0

    receptions = pickgrader_server._extract_cfb_player_stat(
        summary, "Le'Veon Moss", "receptions"
    )
    assert receptions == 3.0


def test_cfb_stat_keys_survive_parse_player_prop_pick():
    import pickgrader_server

    for raw_key, normalized in [
        ("passing_yards", "passing_yards"),
        ("Passing Yards", "passing_yards"),
        ("rushing_yards", "rushing_yards"),
        ("receiving_yards", "receiving_yards"),
        ("receptions", "receptions"),
        ("passing_touchdowns", "passing_touchdowns"),
        ("rushing_touchdowns", "rushing_touchdowns"),
        ("rushing_receiving_yards", "rushing_receiving_yards"),
        ("rush+rec yards", "rushing_receiving_yards"),
    ]:
        parsed = pickgrader_server.parse_player_prop_pick(
            {
                "player_name": "Marcel Reed",
                "stat_key": raw_key,
                "selection": "OVER",
                "line": 100.5,
                "sport": "CFB",
            }
        )
        assert parsed is not None, f"{raw_key!r} did not parse"
        assert parsed["stat_key"] == normalized, f"{raw_key!r} -> {parsed['stat_key']!r}"

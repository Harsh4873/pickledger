"""Tests for MLB First Five pitcher rest-days signal."""
from __future__ import annotations


def test_rest_days_normal_cycle():
    """4-6 days rest = modern MLB normal; rest modifier is 0."""
    from models.mlb_first_five.mlb_first_five_model import _pitcher_rest_runs_modifier

    for days in (4, 5, 6):
        adj, label = _pitcher_rest_runs_modifier(days)
        assert adj == 0.0
        assert "normal" in label.lower()


def test_rest_days_short_rest_bumps_runs_allowed():
    """3 days rest is "short rest" — pitcher gives up more F5 runs."""
    from models.mlb_first_five.mlb_first_five_model import _pitcher_rest_runs_modifier

    adj, label = _pitcher_rest_runs_modifier(3)
    assert adj == 0.20
    assert "short" in label.lower()


def test_rest_days_emergency_short_rest_is_largest():
    """0-2 days rest = bullpen day / opener; biggest runs penalty."""
    from models.mlb_first_five.mlb_first_five_model import _pitcher_rest_runs_modifier

    for days in (0, 1, 2):
        adj, _ = _pitcher_rest_runs_modifier(days)
        assert adj == 0.30


def test_rest_days_long_layoff_adds_rust():
    """Long layoffs add a small rust bump (7-9 days), bigger if 10+."""
    from models.mlb_first_five.mlb_first_five_model import _pitcher_rest_runs_modifier

    rust, _ = _pitcher_rest_runs_modifier(8)
    long_, _ = _pitcher_rest_runs_modifier(14)
    assert rust == 0.05
    assert long_ == 0.15


def test_rest_days_unknown_returns_zero():
    """No rest data (first start of season, etc.) is a no-signal default."""
    from models.mlb_first_five.mlb_first_five_model import _pitcher_rest_runs_modifier

    adj, label = _pitcher_rest_runs_modifier(None)
    assert adj == 0.0
    assert "unknown" in label.lower()


def test_rest_days_compute_from_records():
    """_pitcher_rest_days reads the most recent start date from the
    current-season record list."""
    from models.mlb_first_five.mlb_first_five_model import _pitcher_rest_days

    records = [
        {"date": "2026-05-04", "f5_allowed": 2.0},
        {"date": "2026-05-09", "f5_allowed": 1.0},
    ]
    # Model date 2026-05-15 → last start was 2026-05-09 → 5 days between =
    # 4 days rest under our convention.
    assert _pitcher_rest_days(records, "2026-05-15") == 5

    # No prior starts → None
    assert _pitcher_rest_days([], "2026-05-15") is None

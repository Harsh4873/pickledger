"""WNBA starter-quality and minutes-restriction signal.

The pre-patch WNBA model only had a single team-wide injury penalty
in [0, 0.45]. That folded "3 bench players questionable" and "the
team's star is questionable" into the same number. In reality:

- A "Questionable" star usually means a minutes restriction even
  if she ends up playing.
- 5/5 expected starters healthy is a different gating signal than
  3/5 starters healthy — bench depth doesn't replace the top of a
  WNBA rotation.
- "Day-To-Day" status on a star tonight after she sat last game is
  effectively a minutes restriction.

This module adds an explicit starter-availability summary that the
picks pipeline can use to:
- Downgrade BET → LEAN when the key player situation is murky.
- Add an extra "lineup uncertainty" margin penalty separate from
  the existing pts_share×status_weight penalty.

Inputs come from WNBAPredictionModel.wnba_injuries (status report),
WNBA_STAR_RATINGS (whose presence we know matters), and an optional
``live_stats`` map keyed by normalized name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from .wnba_injuries import (
        STATUS_WEIGHTS,
        WNBA_STAR_RATINGS,
        _normalize_abbr,
        _normalize_name,
    )
except ImportError:
    from wnba_injuries import (
        STATUS_WEIGHTS,
        WNBA_STAR_RATINGS,
        _normalize_abbr,
        _normalize_name,
    )


# A star with a Day-To-Day or Questionable tag almost always means a
# minutes restriction even when officially active. We add a small
# extra penalty per at-risk star, on top of the existing pts_share ×
# status_weight injury penalty, so the contextual layer slightly shifts
# margin in the opponent's favor.
MINUTES_RESTRICTION_PENALTY_PER_AT_RISK_STAR = 0.04
MINUTES_RESTRICTION_CAP = 0.12
LINEUP_UNCERTAINTY_PENALTY_PER_KEY_OUT = 0.06
LINEUP_UNCERTAINTY_CAP = 0.18


@dataclass
class LineupQuality:
    """Snapshot of one team's starter situation."""
    starters_total: int
    starters_healthy: int
    starters_questionable: list[str]
    starters_out: list[str]
    minutes_restriction_penalty: float
    lineup_uncertainty_penalty: float

    @property
    def quality_ratio(self) -> float:
        """Healthy starters as a fraction of the expected starter count."""
        if self.starters_total <= 0:
            return 1.0
        return round(self.starters_healthy / float(self.starters_total), 4)


def get_lineup_quality(
    team_abbr: str,
    injury_report: Optional[dict],
    live_stats: Optional[dict] = None,
) -> LineupQuality:
    """Build a LineupQuality summary for one team.

    "Starters" are inferred from WNBA_STAR_RATINGS plus any name in
    ``live_stats`` flagged as ``"is_starter": True``. Each starter is
    classified as healthy (no listed injury), questionable (Day-To-Day,
    Questionable), or out (Out, Doubtful).

    Returns a zero-penalty payload when the injury report is empty or
    the team has no recognized starters — never raises.
    """
    abbr = _normalize_abbr(team_abbr or "")
    starter_keys = _starter_keys_for_team(abbr, live_stats or {})

    if not abbr or not starter_keys:
        return LineupQuality(
            starters_total=0,
            starters_healthy=0,
            starters_questionable=[],
            starters_out=[],
            minutes_restriction_penalty=0.0,
            lineup_uncertainty_penalty=0.0,
        )

    report = injury_report or {}
    questionable: list[str] = []
    out: list[str] = []

    for key in starter_keys:
        info = report.get(key)
        status = (info or {}).get("status")
        if not status or status not in STATUS_WEIGHTS:
            continue  # treated as healthy
        weight = STATUS_WEIGHTS.get(status, 0.0)
        if weight >= 0.6:  # Out (1.0) or Doubtful (0.65)
            out.append(_display_name(info, key))
        elif weight > 0.0:  # Questionable (0.35) or Day-To-Day (0.2)
            questionable.append(_display_name(info, key))

    healthy = max(0, len(starter_keys) - len(questionable) - len(out))
    minutes_pen = min(
        MINUTES_RESTRICTION_CAP,
        len(questionable) * MINUTES_RESTRICTION_PENALTY_PER_AT_RISK_STAR,
    )
    uncertainty_pen = min(
        LINEUP_UNCERTAINTY_CAP,
        len(out) * LINEUP_UNCERTAINTY_PENALTY_PER_KEY_OUT,
    )

    return LineupQuality(
        starters_total=len(starter_keys),
        starters_healthy=healthy,
        starters_questionable=questionable,
        starters_out=out,
        minutes_restriction_penalty=round(minutes_pen, 4),
        lineup_uncertainty_penalty=round(uncertainty_pen, 4),
    )


def _starter_keys_for_team(abbr: str, live_stats: dict) -> list[str]:
    """Return normalized-name keys for the team's expected starters."""
    keys: list[str] = []

    # Live-stats entries marked as starters take priority.
    for name, payload in (live_stats or {}).items():
        if not isinstance(payload, dict):
            continue
        if not payload.get("is_starter"):
            continue
        team = _normalize_abbr(str(payload.get("team") or ""))
        if team != abbr:
            continue
        keys.append(_normalize_name(name))

    # Fall back to (and de-dup with) the hardcoded star list.
    for key, info in WNBA_STAR_RATINGS.items():
        if _normalize_abbr(str(info.get("team") or "")) != abbr:
            continue
        norm = _normalize_name(key)
        if norm not in keys:
            keys.append(norm)

    return keys


def _display_name(info: Optional[dict], key: str) -> str:
    if isinstance(info, dict):
        name = str(info.get("player_name") or "").strip()
        if name:
            return name
    return key.title()

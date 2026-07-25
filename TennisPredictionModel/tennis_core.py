"""Elo/WElo rating engine and the strictly as-of feature builder.

Three rating systems run side by side over one chronological pass:

* **Elo** — the classic update with the tennis scale factor from Kovalchik
  (2016), ``K_i(t) = scale / (N_i(t) + offset) ** shape``, the curve
  FiveThirtyEight's tennis model uses: a player's first matches move their
  rating a long way, a veteran's barely at all. The published constants
  (250/5/0.4) are the defaults here, but the shipped values are grid-searched
  on the development seasons — see `EloConfig`.
* **Surface Elo** — an independent rating per surface, updated only by matches
  on that surface, with its own match counter driving K. Serving predictions
  read a blend of overall and surface rating; the blend weight is a tuned
  hyper-parameter rather than a folk constant (FiveThirtyEight use 0.71/0.29 on
  hard, Sackmann 50/50).
* **WElo** — Angelini, Candila & De Angelis (EJOR 297(1), 2022): the same
  update scaled by the winner's share of games,
  ``f = games_won_by_winner / total_games``. A 6-0 6-0 win moves the rating by
  the full amount, a 7-6 7-6 win by about half. It is symmetric by
  construction, so the winner's gain is exactly the loser's loss.

Everything a feature row can see is state accumulated *before* that match: the
pass emits the row first and only then folds the result into both players'
state. That ordering is the single invariant this whole model rests on, and
``tests/smoke/test_tennis_model.py`` asserts it directly.

Feature rows are oriented by player key (lexicographically smaller player is
"p1") so the label is not systematically 1, and every player-dependent feature
is a p1-minus-p2 difference. Context features that do not depend on the
orientation (surface, round, best-of) stay as-is. Prediction then symmetrises
explicitly — see `symmetrise` — so ``P(a beats b) + P(b beats a) == 1`` holds
exactly rather than approximately.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from .tennis_data import Match, SURFACES

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
# Gzipped: the snapshot is a committed file the daily job re-reads every
# morning, and gzip is in the standard library, so there is no reason to carry
# several megabytes of whitespace-free JSON in the repository.
RATINGS_PATH = ARTIFACT_DIR / "tennis_ratings.json.gz"

INITIAL_RATING = 1500.0
_RECENT_WINDOW = 25
_FATIGUE_WINDOW_DAYS = 28


@dataclass(frozen=True)
class EloConfig:
    """Rating hyper-parameters, tuned by walk-forward search in training."""

    k_scale: float = 250.0
    k_offset: float = 5.0
    k_shape: float = 0.4
    surface_k_scale: float = 250.0
    surface_k_offset: float = 5.0
    surface_k_shape: float = 0.4
    surface_blend: float = 0.55
    # Grand Slams are best-of-five and far less noisy than a 250-level match,
    # so their results deserve a heavier update. 1.0 disables the adjustment.
    tier_k_weight: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "k_scale": self.k_scale,
            "k_offset": self.k_offset,
            "k_shape": self.k_shape,
            "surface_k_scale": self.surface_k_scale,
            "surface_k_offset": self.surface_k_offset,
            "surface_k_shape": self.surface_k_shape,
            "surface_blend": self.surface_blend,
            "tier_k_weight": self.tier_k_weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EloConfig":
        if not payload:
            return cls()
        known = {key: float(payload[key]) for key in cls().to_dict() if key in payload}
        return cls(**known)


def elo_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _ordinal(iso_date: str) -> int:
    try:
        return date.fromisoformat(iso_date).toordinal()
    except ValueError:
        return 0


class PlayerState:
    """Everything the feature builder knows about one player at a point in time."""

    __slots__ = (
        "elo", "welo", "surface_elo", "matches", "surface_matches", "wins",
        "surface_wins", "last_ordinal", "last_rank", "last_points", "recent",
        "elo_history", "activity", "last_retired_ordinal", "last_match_games",
        "event_key", "event_games", "event_matches", "tour",
    )

    def __init__(self) -> None:
        self.elo = INITIAL_RATING
        self.welo = INITIAL_RATING
        self.surface_elo = {surface: INITIAL_RATING for surface in SURFACES}
        self.matches = 0
        self.surface_matches = {surface: 0 for surface in SURFACES}
        self.wins = 0
        self.surface_wins = {surface: 0 for surface in SURFACES}
        self.last_ordinal = 0
        self.last_rank: int | None = None
        self.last_points: int | None = None
        self.recent: deque[int] = deque(maxlen=_RECENT_WINDOW)
        self.elo_history: deque[float] = deque(maxlen=11)
        # (ordinal, games played) for the fatigue windows.
        self.activity: deque[tuple[int, int]] = deque(maxlen=40)
        self.last_retired_ordinal = 0
        self.last_match_games = 0
        self.event_key = ""
        self.event_games = 0
        self.event_matches = 0
        self.tour = ""

    # -- read-side helpers -------------------------------------------------

    def blended_elo(self, surface: str, blend: float) -> float:
        return blend * self.elo + (1.0 - blend) * self.surface_elo.get(surface, INITIAL_RATING)

    def win_rate(self) -> float:
        return self.wins / self.matches if self.matches else 0.5

    def surface_win_rate(self, surface: str) -> float:
        played = self.surface_matches.get(surface, 0)
        return self.surface_wins.get(surface, 0) / played if played else 0.5

    def recent_win_rate(self, window: int) -> float:
        if not self.recent:
            return 0.5
        sample = list(self.recent)[-window:]
        return sum(sample) / len(sample)

    def elo_momentum(self) -> float:
        """Rating change across the stored history window."""
        if len(self.elo_history) < 2:
            return 0.0
        return self.elo_history[-1] - self.elo_history[0]

    def games_since(self, ordinal: int, days: int) -> int:
        cutoff = ordinal - days
        return sum(games for day, games in self.activity if day >= cutoff)

    def matches_since(self, ordinal: int, days: int) -> int:
        cutoff = ordinal - days
        return sum(1 for day, _ in self.activity if day >= cutoff)

    def rest_days(self, ordinal: int) -> int:
        if not self.last_ordinal:
            return 30
        return max(0, min(365, ordinal - self.last_ordinal))

    # -- write-side --------------------------------------------------------

    def note_event(self, event_key: str) -> None:
        if event_key != self.event_key:
            self.event_key = event_key
            self.event_games = 0
            self.event_matches = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "elo": round(self.elo, 3),
            "welo": round(self.welo, 3),
            "surface_elo": {key: round(value, 3) for key, value in self.surface_elo.items()},
            "matches": self.matches,
            "surface_matches": dict(self.surface_matches),
            "wins": self.wins,
            "surface_wins": dict(self.surface_wins),
            "last_ordinal": self.last_ordinal,
            "last_rank": self.last_rank,
            "last_points": self.last_points,
            "recent": list(self.recent),
            "elo_history": [round(value, 3) for value in self.elo_history],
            "activity": [list(entry) for entry in self.activity],
            "last_retired_ordinal": self.last_retired_ordinal,
            "last_match_games": self.last_match_games,
            "event_key": self.event_key,
            "event_games": self.event_games,
            "event_matches": self.event_matches,
            "tour": self.tour,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlayerState":
        state = cls()
        state.elo = float(payload.get("elo", INITIAL_RATING))
        state.welo = float(payload.get("welo", INITIAL_RATING))
        stored_surface = payload.get("surface_elo") or {}
        state.surface_elo = {
            surface: float(stored_surface.get(surface, INITIAL_RATING)) for surface in SURFACES
        }
        state.matches = int(payload.get("matches", 0))
        stored_played = payload.get("surface_matches") or {}
        state.surface_matches = {surface: int(stored_played.get(surface, 0)) for surface in SURFACES}
        state.wins = int(payload.get("wins", 0))
        stored_wins = payload.get("surface_wins") or {}
        state.surface_wins = {surface: int(stored_wins.get(surface, 0)) for surface in SURFACES}
        state.last_ordinal = int(payload.get("last_ordinal", 0))
        state.last_rank = payload.get("last_rank")
        state.last_points = payload.get("last_points")
        state.recent = deque(payload.get("recent") or [], maxlen=_RECENT_WINDOW)
        state.elo_history = deque(payload.get("elo_history") or [], maxlen=11)
        state.activity = deque(
            ((int(day), int(games)) for day, games in (payload.get("activity") or [])), maxlen=40
        )
        state.last_retired_ordinal = int(payload.get("last_retired_ordinal", 0))
        state.last_match_games = int(payload.get("last_match_games", 0))
        state.event_key = str(payload.get("event_key") or "")
        state.event_games = int(payload.get("event_games", 0))
        state.event_matches = int(payload.get("event_matches", 0))
        state.tour = str(payload.get("tour") or "")
        return state


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

# Differences (p1 - p2); these flip sign when the two players are swapped.
ANTISYMMETRIC_FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "blend_elo_diff",
    "welo_diff",
    "log_rank_diff",
    "log_points_diff",
    "experience_diff",
    "surface_experience_diff",
    "win_rate_diff",
    "surface_win_rate_diff",
    "form10_diff",
    "form25_diff",
    "elo_momentum_diff",
    "h2h_diff",
    "h2h_surface_diff",
    "rest_diff",
    "games_7d_diff",
    "games_14d_diff",
    "matches_28d_diff",
    "event_games_diff",
    "recent_retirement_diff",
    "last_match_games_diff",
    # Interactions. Each is an antisymmetric feature times a symmetric one, so
    # it flips sign under a swap exactly like the terms above — the property
    # `mirror_vector` relies on. A linear model cannot express "rating gaps
    # matter more over five sets" without them; trees get it for free, which is
    # part of why the two families end up so close.
    "elo_diff_bo5",
    "elo_diff_wta",
    "elo_diff_experience",
    "elo_diff_saturating",
    "surface_specialisation",
]

# Context; identical under a swap.
SYMMETRIC_FEATURES = [
    "best_of",
    "tier",
    "round_order",
    "surface_hard",
    "surface_clay",
    "surface_grass",
    "surface_carpet",
    "indoor",
    "is_wta",
    "min_experience",
    "h2h_matches",
]

FEATURE_NAMES = ANTISYMMETRIC_FEATURES + SYMMETRIC_FEATURES
_ANTISYMMETRIC_INDEX = [FEATURE_NAMES.index(name) for name in ANTISYMMETRIC_FEATURES]


def _log_rank(rank: int | None) -> float:
    """Rank on a log scale; unranked players sit past the bottom of the list."""
    if not rank or rank <= 0:
        return math.log(1500.0)
    return math.log(float(rank))


def _log_points(points: int | None) -> float:
    return math.log1p(float(points)) if points and points > 0 else 0.0


class RatingEngine:
    """Chronological replay producing ratings, state and as-of feature rows."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self.players: dict[str, PlayerState] = {}
        # (player_key, opponent_key) -> wins, and the same split by surface.
        self.h2h: dict[tuple[str, str], int] = {}
        self.h2h_surface: dict[tuple[str, str, str], int] = {}
        self.last_ordinal = 0
        self.last_date = ""

    # -- state -------------------------------------------------------------

    def state(self, key: str) -> PlayerState:
        found = self.players.get(key)
        if found is None:
            found = PlayerState()
            self.players[key] = found
        return found

    def _k(self, matches: int) -> float:
        config = self.config
        return config.k_scale / ((matches + config.k_offset) ** config.k_shape)

    def _surface_k(self, matches: int) -> float:
        config = self.config
        return config.surface_k_scale / ((matches + config.surface_k_offset) ** config.surface_k_shape)

    # -- features ----------------------------------------------------------

    def features(
        self,
        p1_key: str,
        p2_key: str,
        *,
        surface: str,
        best_of: int,
        tier: int,
        round_order: int,
        indoor: bool,
        tour: str,
        ordinal: int,
        event_key: str,
        p1_rank: int | None = None,
        p2_rank: int | None = None,
        p1_points: int | None = None,
        p2_points: int | None = None,
    ) -> dict[str, float]:
        blend = self.config.surface_blend
        p1 = self.state(p1_key)
        p2 = self.state(p2_key)
        p1.note_event(event_key)
        p2.note_event(event_key)

        rank1 = p1_rank if p1_rank is not None else p1.last_rank
        rank2 = p2_rank if p2_rank is not None else p2.last_rank
        points1 = p1_points if p1_points is not None else p1.last_points
        points2 = p2_points if p2_points is not None else p2.last_points

        h2h1 = self.h2h.get((p1_key, p2_key), 0)
        h2h2 = self.h2h.get((p2_key, p1_key), 0)
        h2h_surface1 = self.h2h_surface.get((p1_key, p2_key, surface), 0)
        h2h_surface2 = self.h2h_surface.get((p2_key, p1_key, surface), 0)

        retirement1 = 1.0 if p1.last_retired_ordinal and ordinal - p1.last_retired_ordinal <= 60 else 0.0
        retirement2 = 1.0 if p2.last_retired_ordinal and ordinal - p2.last_retired_ordinal <= 60 else 0.0

        elo_gap = p1.elo - p2.elo
        surface_gap = p1.surface_elo.get(surface, INITIAL_RATING) - p2.surface_elo.get(surface, INITIAL_RATING)
        blend_gap = p1.blended_elo(surface, blend) - p2.blended_elo(surface, blend)
        min_experience = math.log1p(min(p1.matches, p2.matches))
        is_wta = 1.0 if str(tour).upper() == "WTA" else 0.0

        return {
            "elo_diff": elo_gap,
            "surface_elo_diff": surface_gap,
            "blend_elo_diff": blend_gap,
            "welo_diff": p1.welo - p2.welo,
            # Lower rank number is better, so negate to keep "bigger is better".
            "log_rank_diff": -(_log_rank(rank1) - _log_rank(rank2)),
            "log_points_diff": _log_points(points1) - _log_points(points2),
            "experience_diff": math.log1p(p1.matches) - math.log1p(p2.matches),
            "surface_experience_diff": math.log1p(p1.surface_matches.get(surface, 0)) - math.log1p(p2.surface_matches.get(surface, 0)),
            "win_rate_diff": p1.win_rate() - p2.win_rate(),
            "surface_win_rate_diff": p1.surface_win_rate(surface) - p2.surface_win_rate(surface),
            "form10_diff": p1.recent_win_rate(10) - p2.recent_win_rate(10),
            "form25_diff": p1.recent_win_rate(25) - p2.recent_win_rate(25),
            "elo_momentum_diff": p1.elo_momentum() - p2.elo_momentum(),
            "h2h_diff": float(h2h1 - h2h2),
            "h2h_surface_diff": float(h2h_surface1 - h2h_surface2),
            "rest_diff": math.log1p(p1.rest_days(ordinal)) - math.log1p(p2.rest_days(ordinal)),
            "games_7d_diff": float(p1.games_since(ordinal, 7) - p2.games_since(ordinal, 7)),
            "games_14d_diff": float(p1.games_since(ordinal, 14) - p2.games_since(ordinal, 14)),
            "matches_28d_diff": float(p1.matches_since(ordinal, _FATIGUE_WINDOW_DAYS) - p2.matches_since(ordinal, _FATIGUE_WINDOW_DAYS)),
            "event_games_diff": float(p1.event_games - p2.event_games),
            "recent_retirement_diff": retirement1 - retirement2,
            "last_match_games_diff": float(p1.last_match_games - p2.last_match_games),
            "elo_diff_bo5": blend_gap * (1.0 if int(best_of) == 5 else 0.0),
            "elo_diff_wta": blend_gap * is_wta,
            "elo_diff_experience": blend_gap * min_experience,
            # Saturating transform: past a few hundred points of rating gap the
            # marginal certainty flattens, which a raw linear term overstates.
            "elo_diff_saturating": 400.0 * math.tanh(blend_gap / 400.0),
            # How much of the edge is surface-specific rather than general.
            "surface_specialisation": surface_gap - elo_gap,
            "best_of": float(best_of),
            "tier": float(tier),
            "round_order": float(round_order),
            "surface_hard": 1.0 if surface == "Hard" else 0.0,
            "surface_clay": 1.0 if surface == "Clay" else 0.0,
            "surface_grass": 1.0 if surface == "Grass" else 0.0,
            "surface_carpet": 1.0 if surface == "Carpet" else 0.0,
            "indoor": 1.0 if indoor else 0.0,
            "is_wta": is_wta,
            "min_experience": min_experience,
            "h2h_matches": float(h2h1 + h2h2),
        }

    def features_for_match(self, match: Match) -> tuple[dict[str, float], int]:
        """Feature row for a historical match, oriented by player key.

        Returns the row and the label (1 when the p1 orientation won).
        """
        p1_key, p2_key = sorted((match.winner_key, match.loser_key))
        p1_is_winner = p1_key == match.winner_key
        rank1, rank2 = (match.winner_rank, match.loser_rank) if p1_is_winner else (match.loser_rank, match.winner_rank)
        points1, points2 = (match.winner_points, match.loser_points) if p1_is_winner else (match.loser_points, match.winner_points)
        row = self.features(
            p1_key,
            p2_key,
            surface=match.surface,
            best_of=match.best_of,
            tier=match.tier,
            round_order=match.round_order,
            indoor=match.court.lower() == "indoor",
            tour=match.tour,
            ordinal=_ordinal(match.date),
            event_key=f"{match.season}:{match.tour}:{match.tournament}",
            p1_rank=rank1,
            p2_rank=rank2,
            p1_points=points1,
            p2_points=points2,
        )
        return row, (1 if p1_is_winner else 0)

    # -- updates -----------------------------------------------------------

    def update(self, match: Match) -> None:
        """Fold a finished match into both players' state.

        Walkovers and disqualifications are dropped entirely — no tennis was
        played, so there is nothing to learn. Retirements *are* rated (someone
        did win) but with the neutral margin weight ``f = 0.5``, because a
        truncated scoreline says nothing about dominance; they are excluded
        from the training set separately.
        """
        if match.status in {"walkover", "disqualified", "other"}:
            return
        winner = self.state(match.winner_key)
        loser = self.state(match.loser_key)
        surface = match.surface
        ordinal = _ordinal(match.date)
        event_key = f"{match.season}:{match.tour}:{match.tournament}"
        winner.note_event(event_key)
        loser.note_event(event_key)

        tier_weight = 1.0 + (self.config.tier_k_weight - 1.0) * ((match.tier - 2) / 3.0)
        tier_weight = max(0.25, tier_weight)

        expected_winner = elo_probability(winner.elo, loser.elo)
        k_winner = self._k(winner.matches) * tier_weight
        k_loser = self._k(loser.matches) * tier_weight
        winner.elo += k_winner * (1.0 - expected_winner)
        loser.elo -= k_loser * (1.0 - expected_winner)

        surface_expected = elo_probability(
            winner.surface_elo.get(surface, INITIAL_RATING),
            loser.surface_elo.get(surface, INITIAL_RATING),
        )
        k_surface_winner = self._surface_k(winner.surface_matches.get(surface, 0)) * tier_weight
        k_surface_loser = self._surface_k(loser.surface_matches.get(surface, 0)) * tier_weight
        winner.surface_elo[surface] = winner.surface_elo.get(surface, INITIAL_RATING) + k_surface_winner * (1.0 - surface_expected)
        loser.surface_elo[surface] = loser.surface_elo.get(surface, INITIAL_RATING) - k_surface_loser * (1.0 - surface_expected)

        total_games = match.total_games
        if match.status == "retired" or total_games <= 0:
            margin_weight = 0.5
        else:
            margin_weight = match.winner_games / total_games
        weighted_expected = elo_probability(winner.welo, loser.welo)
        winner.welo += k_winner * (1.0 - weighted_expected) * margin_weight
        loser.welo -= k_loser * (1.0 - weighted_expected) * margin_weight

        winner.matches += 1
        loser.matches += 1
        winner.wins += 1
        winner.surface_matches[surface] = winner.surface_matches.get(surface, 0) + 1
        loser.surface_matches[surface] = loser.surface_matches.get(surface, 0) + 1
        winner.surface_wins[surface] = winner.surface_wins.get(surface, 0) + 1
        winner.recent.append(1)
        loser.recent.append(0)
        winner.elo_history.append(winner.elo)
        loser.elo_history.append(loser.elo)
        winner.last_ordinal = ordinal
        loser.last_ordinal = ordinal
        winner.activity.append((ordinal, total_games))
        loser.activity.append((ordinal, total_games))
        winner.last_match_games = total_games
        loser.last_match_games = total_games
        winner.event_games += total_games
        loser.event_games += total_games
        winner.event_matches += 1
        loser.event_matches += 1
        winner.tour = match.tour
        loser.tour = match.tour
        if match.winner_rank:
            winner.last_rank = match.winner_rank
        if match.loser_rank:
            loser.last_rank = match.loser_rank
        if match.winner_points:
            winner.last_points = match.winner_points
        if match.loser_points:
            loser.last_points = match.loser_points
        if match.status == "retired":
            loser.last_retired_ordinal = ordinal

        self.h2h[(match.winner_key, match.loser_key)] = self.h2h.get((match.winner_key, match.loser_key), 0) + 1
        self.h2h_surface[(match.winner_key, match.loser_key, surface)] = (
            self.h2h_surface.get((match.winner_key, match.loser_key, surface), 0) + 1
        )
        self.last_ordinal = max(self.last_ordinal, ordinal)
        if match.date > self.last_date:
            self.last_date = match.date

    # -- replay ------------------------------------------------------------

    def replay(
        self,
        matches: Iterable[Match],
        *,
        emit_from: str | None = None,
        emit_to: str | None = None,
        training_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Single chronological pass: emit the row, *then* update the state.

        ``training_only`` keeps retirements out of the emitted rows (their
        outcome is contaminated by the injury, not the matchup) while still
        letting them move the ratings.
        """
        records: list[dict[str, Any]] = []
        for match in matches:
            if match.status in {"walkover", "disqualified", "other"}:
                continue
            in_window = (emit_from is None or match.date >= emit_from) and (
                emit_to is None or match.date <= emit_to
            )
            emit = in_window and (match.completed or not training_only)
            if emit:
                row, label = self.features_for_match(match)
                records.append({
                    "date": match.date,
                    "season": match.season,
                    "tour": match.tour,
                    "surface": match.surface,
                    "tier": match.tier,
                    "tournament": match.tournament,
                    "round": match.round,
                    "best_of": match.best_of,
                    "p1_key": min(match.winner_key, match.loser_key),
                    "p2_key": max(match.winner_key, match.loser_key),
                    "winner_key": match.winner_key,
                    "loser_key": match.loser_key,
                    "features": row,
                    "label": label,
                    "odds": match.odds,
                    "status": match.status,
                })
            self.update(match)
        return records

    # -- serialisation -----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "config": self.config.to_dict(),
            "through_date": self.last_date,
            "through_ordinal": self.last_ordinal,
            "players": {key: state.to_dict() for key, state in self.players.items()},
            "h2h": {f"{a}|{b}": wins for (a, b), wins in self.h2h.items() if wins},
            "h2h_surface": {
                f"{a}|{b}|{surface}": wins
                for (a, b, surface), wins in self.h2h_surface.items()
                if wins
            },
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "RatingEngine":
        engine = cls(EloConfig.from_dict(payload.get("config")))
        engine.players = {
            key: PlayerState.from_dict(value) for key, value in (payload.get("players") or {}).items()
        }
        for key, wins in (payload.get("h2h") or {}).items():
            a, _, b = key.partition("|")
            engine.h2h[(a, b)] = int(wins)
        for key, wins in (payload.get("h2h_surface") or {}).items():
            parts = key.split("|")
            if len(parts) == 3:
                engine.h2h_surface[(parts[0], parts[1], parts[2])] = int(wins)
        engine.last_date = str(payload.get("through_date") or "")
        engine.last_ordinal = int(payload.get("through_ordinal") or 0)
        return engine

    def prune(self, *, active_within_days: int = 1095, activity_days: int = 90) -> dict[str, int]:
        """Shrink the snapshot to what serving actually reads.

        The full state is ~3,000 players carrying years of activity history and
        every head-to-head pair ever played — several megabytes of JSON for a
        file that gets committed and re-read every morning. Serving only needs
        players who might appear on a slate, and fatigue windows never look back
        further than 28 days. Players are kept for three years so someone
        returning from a long injury still has their rating rather than
        restarting at 1500.
        """
        cutoff = self.last_ordinal - active_within_days
        activity_cutoff = self.last_ordinal - activity_days
        before = len(self.players)
        keep = {
            key: state
            for key, state in self.players.items()
            if state.last_ordinal >= cutoff and state.matches > 0
        }
        for state in keep.values():
            state.activity = deque(
                ((day, games) for day, games in state.activity if day >= activity_cutoff),
                maxlen=40,
            )
        self.players = keep
        h2h_before = len(self.h2h)
        self.h2h = {pair: wins for pair, wins in self.h2h.items() if pair[0] in keep and pair[1] in keep}
        self.h2h_surface = {
            triple: wins
            for triple, wins in self.h2h_surface.items()
            if triple[0] in keep and triple[1] in keep
        }
        return {
            "players_before": before,
            "players_after": len(self.players),
            "h2h_before": h2h_before,
            "h2h_after": len(self.h2h),
        }

    def save(self, path: Path = RATINGS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.snapshot(), separators=(",", ":")) + "\n"
        if path.suffix == ".gz":
            # mtime=0 keeps the bytes reproducible, so an unchanged snapshot
            # produces an unchanged file and no empty commit.
            with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as handle:
                handle.write(payload.encode("utf-8"))
        else:
            path.write_text(payload, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = RATINGS_PATH) -> "RatingEngine | None":
        candidates = [path]
        # Accept either form, so a hand-inspected plain-JSON snapshot still loads.
        if path.suffix == ".gz":
            candidates.append(path.with_suffix(""))
        else:
            candidates.append(path.with_suffix(path.suffix + ".gz"))
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                if candidate.suffix == ".gz":
                    with gzip.open(candidate, "rt", encoding="utf-8") as handle:
                        return cls.from_snapshot(json.load(handle))
                return cls.from_snapshot(json.loads(candidate.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
        return None

    def leaderboard(
        self,
        surface: str | None = None,
        limit: int = 25,
        min_matches: int = 20,
        tour: str | None = None,
        active_within_days: int | None = 365,
    ) -> list[dict[str, Any]]:
        """Diagnostic ranking. Elo never decays, so a retired champion keeps
        their final rating forever — the activity filter is what makes this
        read as a *current* leaderboard rather than an all-time one."""
        blend = self.config.surface_blend
        cutoff = (self.last_ordinal - active_within_days) if active_within_days else None
        rows = [
            {
                "player": key,
                "elo": round(state.elo, 1),
                "welo": round(state.welo, 1),
                "surface_elo": round(state.surface_elo.get(surface, INITIAL_RATING), 1) if surface else None,
                "blended": round(state.blended_elo(surface, blend), 1) if surface else round(state.elo, 1),
                "matches": state.matches,
            }
            for key, state in self.players.items()
            if state.matches >= min_matches
            and (tour is None or state.tour == tour)
            and (cutoff is None or state.last_ordinal >= cutoff)
        ]
        rows.sort(key=lambda row: row["blended"], reverse=True)
        return rows[:limit]


# --------------------------------------------------------------------------
# vectors and symmetry
# --------------------------------------------------------------------------


def to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def matrix(records: Sequence[dict[str, Any]]) -> list[list[float]]:
    return [to_vector(record["features"]) for record in records]


def mirror_vector(vector: Sequence[float]) -> list[float]:
    """The same match seen from the other player's side."""
    flipped = list(vector)
    for index in _ANTISYMMETRIC_INDEX:
        flipped[index] = -flipped[index]
    return flipped


def symmetrise(forward: float, mirrored: float) -> float:
    """Average a prediction with its mirror so the two sides sum to exactly 1."""
    return 0.5 * (forward + (1.0 - mirrored))

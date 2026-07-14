from __future__ import annotations

import sys
import traceback
from itertools import combinations
from datetime import datetime
from typing import Any

def _parse_date_arg(date_str: str | None) -> str:
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _parse_game_ids_arg(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {part.strip() for part in raw_value.split(",") if part.strip()}


def _run_for_games(date_arg: str | None, game_ids: set[str] | None = None) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    from live_data import load_props_slate
    from rf_model import build_prop_predictions

    (
        games,
        players,
        opponent_lookup,
        player_df,
        position_baselines,
        league_meta,
        _season,
    ) = load_props_slate(date_arg, game_ids=game_ids)

    if not games or not players:
        return games, players, []

    predictions = build_prop_predictions(
        players,
        opponent_lookup,
        player_df,
        position_baselines,
        league_meta,
    )
    return games, players, predictions


def _format_american_odds(probability: float) -> str:
    fair_american = (1.0 / probability - 1.0) * 100.0
    rounded = int(round(fair_american))
    if rounded > 0:
        return f"+{rounded}"
    return str(rounded)


def main() -> None:
    date_arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    selected_game_ids: set[str] = set()
    list_game_ids = False
    for arg in sys.argv[1:]:
        if arg.startswith("--game-ids="):
            selected_game_ids = _parse_game_ids_arg(arg.split("=", 1)[1])
            break
        if arg == "--list-game-ids":
            list_game_ids = True

    output_date = _parse_date_arg(date_arg)

    print(f"NBA PROPS MODEL - {output_date}")
    print("=" * 80)

    if list_game_ids:
        try:
            from live_data import fetch_todays_games

            for game in fetch_todays_games(date_arg):
                game_id = str(game.get("game_id", "")).strip()
                if game_id:
                    away_team = str(game.get("away_team", "")).strip()
                    home_team = str(game.get("home_team", "")).strip()
                    if away_team and home_team:
                        print(f"GAME_ID: {game_id} | {away_team} @ {home_team}")
                    else:
                        print(f"GAME_ID: {game_id}")
        except Exception as exc:
            print(f"Error loading NBA game IDs: {exc}")
            traceback.print_exc()
        print("=" * 80)
        return

    try:
        games, players, predictions = _run_for_games(date_arg, selected_game_ids or None)
    except Exception as exc:
        print(f"Error loading NBA props slate: {exc}")
        traceback.print_exc()
        return

    if not games:
        print("No NBA games found for today.")
        print("=" * 80)
        return

    if not players:
        print(f"No qualifying player props candidates found for {output_date}.")
        print("=" * 80)
        return

    predictions_by_player: dict[int, list] = {}
    for prediction in predictions:
        predictions_by_player.setdefault(prediction.player_id, []).append(prediction)

    players_by_game: dict[str, list] = {}
    for player in players:
        players_by_game.setdefault(player.game_id, []).append(player)

    for game in games:
        game_id = str(game["game_id"])
        game_players = players_by_game.get(game_id, [])
        if not game_players:
            continue

        print()
        print(f"GAME: {game['away_team']} @ {game['home_team']}")
        print("-" * 80)

        for player in game_players:
            print(
                f"PLAYER: {player.player_name} | {player.position} | "
                f"{player.team_abbreviation} | vs {player.opponent_team_abbreviation}"
            )
            print(
                "Season avg: "
                f"pts {player.points_per_game:.1f} | "
                f"reb {player.rebounds_per_game:.1f} | "
                f"ast {player.assists_per_game:.1f} | "
                f"mp {player.mp_per_game:.1f} | "
                f"usage {player.usage_rate:.1f}%"
            )

            for prediction in predictions_by_player.get(player.player_id, []):
                print()
                print(f"  PROP: {prediction.prop_label} - Line: {prediction.line:.1f}")
                print(
                    f"  RF Predicted: {prediction.predicted_value:.1f} | "
                    f"Direction: {prediction.direction} | Edge: {prediction.edge_pct:.1f}%"
                )
                print(
                    f"  Confidence: {prediction.confidence:.0f}% | "
                    f"Full Kelly: {prediction.full_kelly:.1f}% | "
                    f"1/4 Kelly: {prediction.quarter_kelly:.1f}% bankroll"
                )
                print(f"  True Prob: {prediction.true_prob * 100.0:.1f}%")
                print(f"  **Decision: {prediction.decision_text()}**")
                print(f"  Reason: {prediction.reason}")

            print()

    best_bets = sorted(
        [prediction for prediction in predictions if prediction.decision == "BET"],
        key=lambda prediction: prediction.edge_pct,
        reverse=True,
    )[:5]
    pass_count = sum(1 for prediction in predictions if prediction.decision == "PASS")

    print("=" * 80)
    print("BEST BETS SUMMARY")
    print("=" * 80)
    print("Rank | Player         | Prop | Line | Predicted | Edge  | 1/4 Kelly | Decision")
    if best_bets:
        for index, prediction in enumerate(best_bets, start=1):
            player_name = prediction.player_name[:13].ljust(13)
            print(
                f"{index}.   | {player_name} | "
                f"{prediction.prop_key:<4} | "
                f"{prediction.line:>4.1f} | "
                f"{prediction.predicted_value:>8.1f} | "
                f"{prediction.edge_pct:>5.1f}% | "
                f"{prediction.quarter_kelly:>8.1f}% | "
                f"{prediction.summary_decision()}"
            )
    else:
        print("No BET picks met the threshold.")

    print()
    print(f"PASS COUNT: {pass_count} props did not meet threshold")
    print("=" * 80)

    tier1_candidates = [
        prediction
        for prediction in predictions
        if prediction.decision == "BET"
        and prediction.confidence >= 68.0
        and prediction.edge_pct >= 9.0
        and prediction.prop_key in {"reb", "pts"}
        and 0.61 <= prediction.true_prob <= 0.74
    ]

    print("PARLAY SUGGESTIONS (Tier 1 Only — reb/pts, conf≥68, edge≥9%)")
    print("=" * 80)

    if len(tier1_candidates) < 3:
        print("⚠️  Not enough Tier 1 props today for a parlay. Bet straights only.")
        print()
        return

    combo_rows: list[tuple[tuple[Any, Any, Any], float, bool]] = []
    for combo in combinations(tier1_candidates, 3):
        game_counts: dict[str, int] = {}
        for leg in combo:
            game_counts[leg.game_id] = game_counts.get(leg.game_id, 0) + 1
        if max(game_counts.values()) > 2:
            continue
        combined_prob = combo[0].true_prob * combo[1].true_prob * combo[2].true_prob
        is_three_unique_games = len(game_counts) == 3
        combo_rows.append((combo, combined_prob, is_three_unique_games))

    sorted_combos = sorted(
        combo_rows,
        key=lambda row: (row[2], row[1]),
        reverse=True,
    )[:3]

    if not sorted_combos:
        print("⚠️  Not enough Tier 1 props today for a parlay. Bet straights only.")
        print()
        return

    for index, (combo, combined_prob, _is_three_unique_games) in enumerate(sorted_combos, start=1):
        print(
            f"COMBO {index} — Combined Prob: {combined_prob * 100.0:.1f}% | "
            f"Fair Odds: {_format_american_odds(combined_prob)}"
        )
        for leg_index, leg in enumerate(combo, start=1):
            print(
                f"  Leg {leg_index}: {leg.player_name} | {leg.prop_key} {leg.direction} {leg.line:.1f} | "
                f"Edge {leg.edge_pct:.1f}% | Conf {leg.confidence:.0f}%"
            )

        combo_game_counts: dict[str, int] = {}
        for leg in combo:
            combo_game_counts[leg.game_id] = combo_game_counts.get(leg.game_id, 0) + 1
        correlated_game_id = next((game_id for game_id, count in combo_game_counts.items() if count == 2), None)
        if correlated_game_id is not None:
            same_game_legs = [leg for leg in combo if leg.game_id == correlated_game_id]
            correlated_label = f"{same_game_legs[0].away_team_name}@{same_game_legs[0].home_team_name}"
            print(f"  ⚠️  2 legs from same game ({correlated_label}) — correlated risk")
        print()


if __name__ == "__main__":
    main()

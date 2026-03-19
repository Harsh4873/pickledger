from __future__ import annotations

import sys
import traceback
from datetime import datetime

from live_data import load_props_slate
from rf_model import build_prop_predictions


def _parse_date_arg(date_str: str | None) -> str:
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def main() -> None:
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    output_date = _parse_date_arg(date_arg)

    print(f"NBA PROPS MODEL - {output_date}")
    print("=" * 80)

    try:
        (
            games,
            players,
            opponent_lookup,
            player_df,
            position_baselines,
            league_meta,
            _season,
        ) = load_props_slate(date_arg)
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

    try:
        predictions = build_prop_predictions(
            players,
            opponent_lookup,
            player_df,
            position_baselines,
            league_meta,
        )
    except Exception as exc:
        print(f"Error building NBA props predictions: {exc}")
        traceback.print_exc()
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


if __name__ == "__main__":
    main()

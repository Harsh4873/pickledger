from __future__ import annotations

import argparse
import sys
from datetime import datetime

from calibration import apply_moneyline_calibration
from date_utils import get_mlb_slate_date
from live_data import build_live_dataframe
from market_mechanics import calculate_edge, check_minimum_threshold, remove_vig
from model_v2 import StaleV2Artifact, predict_moneyline_v2, predict_totals_v2
from model_variants import VALID_MLB_MODEL_VARIANTS
from moneyline_model import predict_home_win_probability
from prediction_logging import append_prediction_rows, build_prediction_log_rows
from probability_layers import predict_total_runs
from sportsline_odds import fetch_mlb_market_odds_for_date
from totals_model import predict_totals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLB model for a selected date and variant.")
    parser.add_argument(
        "legacy_date",
        nargs="?",
        default="",
        help="Optional legacy date arg in MM/DD/YYYY or YYYY-MM-DD format.",
    )
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    parser.add_argument(
        "--variant",
        choices=VALID_MLB_MODEL_VARIANTS,
        default="old",
        help="Model variant to run.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable local prediction logging.")
    return parser.parse_args(argv[1:] if argv else None)


def _parse_date(raw_value: str) -> datetime.date:
    raw = str(raw_value or "").strip()
    if not raw:
        return get_mlb_slate_date()

    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {raw}. Use YYYY-MM-DD or MM/DD/YYYY.")


def _prob_to_american(probability: float) -> int:
    probability = max(1e-6, min(1 - 1e-6, probability))
    if probability >= 0.5:
        return int(round(-100.0 * probability / (1.0 - probability)))
    return int(round(100.0 * (1.0 - probability) / probability))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv)
    try:
        target_date = _parse_date(args.date or args.legacy_date)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        # Fetch market odds once and reuse: they feed the totals model as an
        # input feature (market_total_line) AND gate ML/OU edge calculation.
        market_odds_map = fetch_mlb_market_odds_for_date(target_date)
        live_frame = build_live_dataframe(target_date, market_odds_map=market_odds_map)
        if live_frame.empty:
            print(f"No MLB games found for {target_date.isoformat()}.")
            return 0

        if args.variant == "new":
            # v2 stack: HistGradientBoosting + isotonic calibration + market-
            # residual totals. Inference returns `calibrated_home_win_probability`
            # and `predicted_total_runs` directly, so downstream code is shared.
            try:
                predictions = predict_moneyline_v2(live_frame)
                try:
                    predictions = predict_totals_v2(predictions)
                except FileNotFoundError:
                    predictions = predictions.copy()
                    predictions["predicted_total_runs"] = predictions.apply(
                        lambda row: predict_total_runs(row.to_dict()),
                        axis=1,
                    )
                except StaleV2Artifact as stale_exc:
                    # Totals v2 artifact is a pre-v2 file — lean on the legacy
                    # totals model under the same ``new`` variant slot so
                    # MLB New still produces a totals pick. Log clearly.
                    print(
                        f"[run_today] MLB NEW: stale v2 totals artifact "
                        f"({stale_exc}); falling back to legacy totals model "
                        f"under variant=new.",
                        file=sys.stderr,
                    )
                    try:
                        predictions = predict_totals(predictions, variant="new")
                    except FileNotFoundError:
                        predictions = predictions.copy()
                        predictions["predicted_total_runs"] = predictions.apply(
                            lambda row: predict_total_runs(row.to_dict()),
                            axis=1,
                        )
            except StaleV2Artifact as stale_exc:
                # Moneyline v2 artifact on disk is a pre-v2 file (no
                # ``variant="new"`` metadata). This commonly happens on a
                # fresh clone before the user has run ``train_model_v2.py``
                # locally. Fall back to the legacy stack for the ``new``
                # variant so MLB New is fully standalone — no MLB Old
                # bootstrap required. Same mechanics MLB Old uses, just
                # loading the ``_new.joblib`` artifacts instead.
                print(
                    f"[run_today] MLB NEW: stale v2 moneyline artifact "
                    f"({stale_exc}); falling back to legacy pipeline under "
                    f"variant=new. Run `python train_model_v2.py` inside "
                    f"MLBPredictionModel (or trigger the mlb-train GitHub "
                    f"Actions workflow) to enable the HistGradientBoosting "
                    f"stack.",
                    file=sys.stderr,
                )
                predictions = predict_home_win_probability(live_frame, variant="new")
                predictions = apply_moneyline_calibration(predictions, variant="new")
                try:
                    predictions = predict_totals(predictions, variant="new")
                except FileNotFoundError:
                    predictions = predictions.copy()
                    predictions["predicted_total_runs"] = predictions.apply(
                        lambda row: predict_total_runs(row.to_dict()),
                        axis=1,
                    )
        else:
            predictions = predict_home_win_probability(live_frame, variant=args.variant)
            predictions = apply_moneyline_calibration(predictions, variant=args.variant)
            try:
                predictions = predict_totals(predictions, variant=args.variant)
            except FileNotFoundError:
                predictions = predictions.copy()
                predictions["predicted_total_runs"] = predictions.apply(
                    lambda row: predict_total_runs(row.to_dict()),
                    axis=1,
                )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"MLB live inference failed: {exc}", file=sys.stderr)
        return 1

    variant_label = "MLBOLD" if args.variant == "old" else "MLBNEW"
    model_name = "MLB OLD" if args.variant == "old" else "MLB NEW"

    print(f"MLB Prediction Model - {variant_label} - Games for {target_date.isoformat()}")
    print("=" * 60)
    print(f"Found {len(predictions)} games.\n")

    prediction_rows = predictions.to_dict("records")
    for row in prediction_rows:
        away_key = str(row.get("away_team", "")).strip().split()[-1].lower() if row.get("away_team") else ""
        home_key = str(row.get("home_team", "")).strip().split()[-1].lower() if row.get("home_team") else ""
        mo = market_odds_map.get((away_key, home_key), {})
        row["market_ml_away"] = mo.get("ml_away")
        row["market_ml_home"] = mo.get("ml_home")
        row["market_total_line"] = mo.get("total_line")
        row["totals_line"] = mo.get("total_line")

    if not args.no_log:
        append_prediction_rows(
            build_prediction_log_rows(
                prediction_rows,
                model_name=model_name,
                model_variant=args.variant,
            )
        )

    for row in prediction_rows:
        home_prob = float(row.get("calibrated_home_win_probability", row["raw_home_win_probability"]))
        away_prob = 1.0 - home_prob
        home_odds = _prob_to_american(home_prob)
        away_odds = _prob_to_american(away_prob)

        print("---")
        print(
            f"{row['away_team']}|{row['home_team']}|"
            f"{away_odds}|{home_odds}|{away_prob:.4f}|{home_prob:.4f}"
        )

        ml_away = row.get("market_ml_away")
        ml_home = row.get("market_ml_home")
        market_total = row.get("market_total_line")

        if ml_away is not None and ml_home is not None:
            true_away, true_home = remove_vig(int(ml_away), int(ml_home))
            home_ml_edge = calculate_edge(home_prob, true_home)
            away_ml_edge = calculate_edge(away_prob, true_away)
            best_edge = max(home_ml_edge, away_ml_edge)
            best_side = row["home_team"] if home_ml_edge >= away_ml_edge else row["away_team"]
            ml_bet = check_minimum_threshold(best_edge, "moneyline")
            print(
                f"ML market: {row['away_team']} {int(ml_away):+d} | "
                f"{row['home_team']} {int(ml_home):+d}"
            )
            print(
                f"ML vig-free: {row['away_team']} {true_away:.1%} | "
                f"{row['home_team']} {true_home:.1%}"
            )
            print(
                f"ML edge: best={best_side} {best_edge:+.1%} | "
                f"BET: {'YES' if ml_bet else 'PASS'}"
            )
        else:
            print("ML market: unavailable (SportsLine scrape failed)")
            print(f"ML model only: {row['home_team']} {home_prob:.1%}")

        predicted_total = float(row.get("predicted_total_runs", predict_total_runs(row)))

        if market_total is not None:
            line = float(market_total)
            totals_edge = predicted_total - line
            direction = "OVER" if totals_edge > 0 else "UNDER"
            totals_bet = abs(totals_edge) >= 0.4 and check_minimum_threshold(
                abs(totals_edge) / 9.0, "total"
            )
            print(
                f"OU market: {line:.1f} | model: {predicted_total:.2f} | "
                f"edge: {totals_edge:+.2f} runs"
            )
            print(
                f"OU pick: {direction} {line:.1f} | "
                f"BET: {'YES' if totals_bet else 'PASS'}"
            )
            selection = direction if totals_bet else "PASS"
            ou_line = line
        else:
            ou_line = 8.5
            if predicted_total > ou_line + 0.5:
                selection = "OVER"
            elif predicted_total < ou_line - 0.5:
                selection = "UNDER"
            else:
                selection = "PASS"
            print(f"OU market: unavailable | model: {predicted_total:.2f}")

        print(f"OU|{selection}|{ou_line}|{predicted_total:.2f}")
        print("---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

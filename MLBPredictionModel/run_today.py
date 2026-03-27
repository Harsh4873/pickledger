from __future__ import annotations

import sys
from datetime import datetime

from calibration import apply_moneyline_calibration
from live_data import build_live_dataframe
from mlb_api import StatsAPIClient
from moneyline_model import predict_home_win_probability
from prediction_logging import append_prediction_rows, build_prediction_log_rows, compute_totals_confidence
from probability_layers import predict_total_runs
from totals_model import predict_totals


def _parse_date(argv: list[str]) -> datetime.date:
    if len(argv) <= 1:
        return datetime.now().date()

    raw = argv[1]
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
    argv = argv or sys.argv
    try:
        target_date = _parse_date(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        live_frame = build_live_dataframe(target_date)
        if live_frame.empty:
            print(f"No MLB games found for {target_date.isoformat()}.")
            return 0

        predictions = predict_home_win_probability(live_frame)
        predictions = apply_moneyline_calibration(predictions)
        try:
            predictions = predict_totals(predictions)
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

    print(f"MLB Prediction Model - Games for {target_date.isoformat()}")
    print("=" * 60)
    print(f"Found {len(predictions)} games.\n")

    client = StatsAPIClient()
    prediction_rows = predictions.to_dict("records")
    for row in prediction_rows:
        predicted_total = float(row.get("predicted_total_runs", predict_total_runs(row)))
        totals_line = client.get_game_total_line(int(row["game_pk"]))
        row["predicted_total_runs"] = predicted_total
        row["totals_line"] = totals_line
        row["totals_confidence"] = compute_totals_confidence(predicted_total, totals_line)

    append_prediction_rows(build_prediction_log_rows(prediction_rows))

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

        predicted_total = float(row["predicted_total_runs"])
        ou_line = row.get("totals_line")
        totals_confidence = row.get("totals_confidence")
        if ou_line is None:
            selection = "PASS"
            line_display = "N/A"
        elif predicted_total > ou_line + 0.5:
            selection = "OVER"
            line_display = f"{ou_line:.1f}"
        elif predicted_total < ou_line - 0.5:
            selection = "UNDER"
            line_display = f"{ou_line:.1f}"
        else:
            selection = "PASS"
            line_display = f"{ou_line:.1f}"
        confidence_display = "" if totals_confidence is None else f"{float(totals_confidence):.4f}"
        print(f"OU|{selection}|{line_display}|{predicted_total:.2f}|{confidence_display}")
        print("---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

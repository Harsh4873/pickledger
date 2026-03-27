from __future__ import annotations

import sys
from datetime import datetime

import statsapi

from live_data import build_live_dataframe
from totals_model import load_totals_model, predict_totals, select_totals_feature_frame


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


def _game_result(game_pk: int) -> tuple[str, int | None]:
    payload = statsapi.get("game", {"gamePk": int(game_pk)})
    status = (
        ((payload.get("gameData") or {}).get("status") or {}).get("detailedState")
        or ((payload.get("gameData") or {}).get("status") or {}).get("abstractGameState")
        or "Unknown"
    )
    teams = ((payload.get("liveData") or {}).get("linescore") or {}).get("teams") or {}
    away_runs = (teams.get("away") or {}).get("runs")
    home_runs = (teams.get("home") or {}).get("runs")
    if away_runs is None or home_runs is None or "final" not in status.lower():
        return status, None
    return status, int(away_runs) + int(home_runs)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    try:
        target_date = _parse_date(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    live_frame = build_live_dataframe(target_date)
    if live_frame.empty:
        print(f"No MLB games found for {target_date.isoformat()}.")
        return 0

    artifact = load_totals_model()
    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]
    blend_weight_model = float(metadata.get("blend_weight_model", 1.0))

    predicted = predict_totals(live_frame.copy())
    features = select_totals_feature_frame(predicted)
    predicted["diagnostic_raw_model_total_runs"] = pipeline.predict(features)

    print(f"MLB totals bias diagnostic for {target_date.isoformat()}")
    print(
        "away_team|home_team|status|actual_total|heuristic_total|raw_model_total|"
        "blended_total|raw_error|heur_error|blend_error"
    )
    for _, row in predicted.iterrows():
        status, actual_total = _game_result(int(row["game_pk"]))
        heuristic_total = float(row["heuristic_total_runs"])
        raw_model_total = float(row["diagnostic_raw_model_total_runs"])
        blended_total = (
            blend_weight_model * raw_model_total + (1.0 - blend_weight_model) * heuristic_total
        )

        if actual_total is None:
            raw_error = ""
            heur_error = ""
            blend_error = ""
        else:
            raw_error = f"{raw_model_total - actual_total:+.3f}"
            heur_error = f"{heuristic_total - actual_total:+.3f}"
            blend_error = f"{blended_total - actual_total:+.3f}"

        actual_display = "" if actual_total is None else str(actual_total)
        print(
            f"{row['away_team']}|{row['home_team']}|{status}|{actual_display}|"
            f"{heuristic_total:.3f}|{raw_model_total:.3f}|{blended_total:.3f}|"
            f"{raw_error}|{heur_error}|{blend_error}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

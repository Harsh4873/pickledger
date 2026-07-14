#!/usr/bin/env python3
"""Backtest committed and candidate parlay-card engines without future leakage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_parlay_cards as builder


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _date_range(start: str, end: str) -> list[str]:
    dates = sorted(
        {
            path.stem
            for directory in (builder.MODEL_CACHE_DIR, builder.PLAYER_PROPS_CACHE_DIR)
            for path in directory.glob("20??-??-??.json")
            if start <= path.stem <= end
        }
    )
    return dates


def _committed_payloads(start: str, end: str, engine: str) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(builder.PARLAY_CARDS_DIR.glob("20??-??-??.json")):
        if not start <= path.stem <= end:
            continue
        payload = _read_json(path)
        if not payload:
            continue
        version = _clean(payload.get("engineVersion")) or "unknown"
        if engine in {"committed", "all"} or engine == version or engine == version.replace("parlay_cards_", ""):
            payloads.append((version, payload))
    return payloads


def _candidate_v3_payloads(start: str, end: str) -> list[tuple[str, dict[str, Any]]]:
    prior_v3: list[dict[str, Any]] = []
    payloads: list[tuple[str, dict[str, Any]]] = []
    for date_iso in _date_range(start, end):
        team_payload = _read_json(builder.MODEL_CACHE_DIR / f"{date_iso}.json")
        prop_payload = _read_json(builder.PLAYER_PROPS_CACHE_DIR / f"{date_iso}.json")
        if not team_payload and not prop_payload:
            continue
        payload = builder.build_parlay_payload(
            date_iso,
            team_payload,
            prop_payload,
            team_history=builder._payloads_before(builder.MODEL_CACHE_DIR, date_iso),
            prop_history=builder._payloads_before(builder.PLAYER_PROPS_CACHE_DIR, date_iso),
            prior_payloads=prior_v3,
        )
        payloads.append((builder.ENGINE_VERSION, payload))
        prior_v3.append(payload)
    return payloads


def _card_mode(card: dict[str, Any]) -> str:
    return builder._card_pick_mode(card)


def _card_result(card: dict[str, Any]) -> str:
    result = _clean(card.get("result")).lower()
    return result if result in {"win", "loss", "push", "pending"} else "pending"


def _dedupe(cards: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for card in cards:
        seen[
            (
                _clean(card.get("date")),
                _clean(card.get("category")),
                _clean(card.get("id")),
                _clean(card.get("comboKey")),
            )
        ] = card
    return list(seen.values())


def summarize(engine_payloads: list[tuple[str, dict[str, Any]]], *, settled_only: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for engine, payload in engine_payloads:
        for card in payload.get("cards") or []:
            if not isinstance(card, dict):
                continue
            result = _card_result(card)
            if settled_only and result == "pending":
                continue
            grouped[(engine, _card_mode(card), _clean(card.get("category")) or "uncategorized")].append(card)

    rows: list[dict[str, Any]] = []
    for (engine, mode, category), cards in sorted(grouped.items()):
        unique_cards = _dedupe(cards)
        result_counts = Counter(_card_result(card) for card in unique_cards)
        settled = result_counts["win"] + result_counts["loss"]
        net = round(sum(float(card.get("profitUnits") or 0.0) for card in unique_cards if _card_result(card) in {"win", "loss"}), 2)
        odds = [int(card.get("oddsAmerican") or 0) for card in unique_cards if card.get("oddsAmerican") is not None]
        leg_counts = Counter(int(card.get("legCount") or 0) for card in unique_cards)
        exposure = Counter(
            _clean(leg.get("legId"))
            for card in unique_cards
            for leg in card.get("legs") or []
            if isinstance(leg, dict) and _clean(leg.get("legId"))
        )
        non_consensus_player = sum(
            1
            for card in unique_cards
            if mode == "player" and int(card.get("consensusLegs") or 0) == 0
        )
        rows.append(
            {
                "engine": engine,
                "mode": mode,
                "category": category,
                "slips": len(unique_cards),
                "wins": result_counts["win"],
                "losses": result_counts["loss"],
                "pushes": result_counts["push"],
                "pending": result_counts["pending"],
                "settled": settled,
                "hitRate": round(result_counts["win"] / settled, 4) if settled else None,
                "units": net,
                "roi": round(net / settled, 4) if settled else None,
                "averageOdds": round(sum(odds) / len(odds), 1) if odds else None,
                "twoLegSlips": leg_counts[2],
                "threeLegSlips": leg_counts[3],
                "maxLegExposure": max(exposure.values(), default=0),
                "uniqueLegs": len(exposure),
                "nonConsensusPlayerSlips": non_consensus_player,
            }
        )
    return rows


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fieldnames = list(rows[0].keys()) if rows else [
            "engine",
            "mode",
            "category",
            "slips",
            "wins",
            "losses",
            "pushes",
            "pending",
            "settled",
            "hitRate",
            "units",
            "roi",
            "averageOdds",
            "twoLegSlips",
            "threeLegSlips",
            "maxLegExposure",
            "uniqueLegs",
            "nonConsensusPlayerSlips",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--engine",
        default="all",
        help="Engine to include: all, committed, v1, v2, v3, or an exact engineVersion.",
    )
    parser.add_argument("--settled-only", action="store_true", help="Exclude pending cards from summarized metrics.")
    parser.add_argument("--output", required=True, help="Output path ending in .json or .csv.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    engine = _clean(args.engine)
    engine_payloads: list[tuple[str, dict[str, Any]]] = []
    if engine in {"all", "committed", "v1", "v2"} or engine.startswith("parlay_cards_"):
        committed_filter = {
            "v1": "parlay_cards_v1",
            "v2": "parlay_cards_v2_quality_guard",
        }.get(engine, engine)
        engine_payloads.extend(_committed_payloads(args.start, args.end, committed_filter))
    if engine in {"all", "v3", "v5", builder.ENGINE_VERSION}:
        engine_payloads.extend(_candidate_v3_payloads(args.start, args.end))
    rows = summarize(engine_payloads, settled_only=args.settled_only)
    _write_output(Path(args.output), rows)
    print(f"[parlay-backtest] wrote {len(rows)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

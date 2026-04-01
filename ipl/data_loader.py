from __future__ import annotations

import csv
import os
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import requests


CRICSHEET_IPL_ZIP_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
REPO_ROOT = Path(__file__).resolve().parent.parent
IPL_DIR = REPO_ROOT / "ipl"
DATA_DIR = IPL_DIR / "data"
ZIP_PATH = DATA_DIR / "ipl_csv2.zip"
RAW_DIR = DATA_DIR / "raw"


def _default_db_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "pickledger.db",
        here / "pickledger.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def download_cricsheet_ipl() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Cricsheet IPL archive to {ZIP_PATH} ...")
    with requests.get(CRICSHEET_IPL_ZIP_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with ZIP_PATH.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    extracted = 0
    print(f"Extracting {ZIP_PATH} into {RAW_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        archive.extractall(RAW_DIR)
        extracted = len(members)

    print(f"Extracted {extracted} files")
    return extracted


def _normalize_int(value: str | None) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return 0


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_ball(ball_value: str | None) -> tuple[int, int]:
    text = str(ball_value or "").strip()
    if not text:
        return 0, 0
    if "." in text:
        over_str, ball_str = text.split(".", 1)
        return _normalize_int(over_str), _normalize_int(ball_str)
    return _normalize_int(text), 0


def _load_info_rows(path: Path) -> dict[str, list[str]]:
    info: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3 or row[0] != "info":
                continue
            key = row[1].strip()
            value = row[2].strip() if len(row) > 2 else ""
            info.setdefault(key, []).append(value)
    return info


def _build_match_record(match_id: str, info: dict[str, list[str]]) -> tuple:
    teams = info.get("team", [])
    return (
        match_id,
        (info.get("season") or [None])[0],
        _normalize_text((info.get("date") or [None])[0]),
        _normalize_text((info.get("venue") or [None])[0]),
        _normalize_text((info.get("city") or [None])[0]),
        _normalize_text(teams[0]) if len(teams) > 0 else None,
        _normalize_text(teams[1]) if len(teams) > 1 else None,
        _normalize_text((info.get("toss_winner") or [None])[0]),
        _normalize_text((info.get("toss_decision") or [None])[0]),
        _normalize_text((info.get("winner") or [None])[0]),
        _normalize_int((info.get("winner_runs") or [0])[0]),
        _normalize_int((info.get("winner_wickets") or [0])[0]),
        _normalize_text((info.get("player_of_match") or [None])[0]),
    )


def _iter_delivery_rows(path: Path) -> Iterable[tuple]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                match_id = str(row.get("match_id", "")).strip()
                if not match_id:
                    continue
                over, ball = _parse_ball(row.get("ball"))
                yield (
                    match_id,
                    _normalize_int(row.get("innings")),
                    over,
                    ball,
                    _normalize_text(row.get("batting_team")),
                    _normalize_text(row.get("bowling_team")),
                    _normalize_text(row.get("striker")),
                    _normalize_text(row.get("non_striker")),
                    _normalize_text(row.get("bowler")),
                    _normalize_int(row.get("runs_off_bat")),
                    _normalize_int(row.get("extras")),
                    _normalize_int(row.get("wides")),
                    _normalize_int(row.get("noballs")),
                    _normalize_int(row.get("byes")),
                    _normalize_int(row.get("legbyes")),
                    _normalize_int(row.get("penalty")),
                    _normalize_text(row.get("wicket_type")),
                    _normalize_text(row.get("player_dismissed")),
                    _normalize_text(row.get("other_wicket_type")),
                    _normalize_text(row.get("other_player_dismissed")),
                )
            except Exception as exc:
                print(f"[WARN] Skipping malformed delivery row in {path.name}: {exc}")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ipl_matches (
            match_id TEXT PRIMARY KEY,
            season TEXT,
            date TEXT,
            venue TEXT,
            city TEXT,
            team1 TEXT,
            team2 TEXT,
            toss_winner TEXT,
            toss_decision TEXT,
            winner TEXT,
            win_by_runs INTEGER,
            win_by_wickets INTEGER,
            player_of_match TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ipl_deliveries (
            match_id TEXT,
            innings INTEGER,
            over INTEGER,
            ball INTEGER,
            batting_team TEXT,
            bowling_team TEXT,
            striker TEXT,
            non_striker TEXT,
            bowler TEXT,
            runs_off_bat INTEGER,
            extras INTEGER,
            wides INTEGER,
            noballs INTEGER,
            byes INTEGER,
            legbyes INTEGER,
            penalty INTEGER,
            wicket_type TEXT,
            player_dismissed TEXT,
            other_wicket_type TEXT,
            other_player_dismissed TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ipl_deliveries_match_innings_over_ball
        ON ipl_deliveries (match_id, innings, over, ball)
        """
    )


def ingest_to_sqlite(db_path: str | os.PathLike[str]) -> None:
    db_file = Path(db_path)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    info_files = sorted(RAW_DIR.glob("*_info.csv"))

    if not info_files:
        raise FileNotFoundError(
            f"No extracted Cricsheet info files found in {RAW_DIR}. Run download_cricsheet_ipl() first."
        )

    print(f"Ingesting IPL data into SQLite DB at {db_file} ...")
    with sqlite3.connect(db_file) as conn:
        _ensure_schema(conn)

        for index, info_file in enumerate(info_files, start=1):
            match_id = info_file.stem.removesuffix("_info")
            deliveries_file = RAW_DIR / f"{match_id}.csv"

            if index % 100 == 0:
                print(f"Processed {index}/{len(info_files)} match files ...")

            try:
                info = _load_info_rows(info_file)
                match_record = _build_match_record(match_id, info)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO ipl_matches (
                        match_id,
                        season,
                        date,
                        venue,
                        city,
                        team1,
                        team2,
                        toss_winner,
                        toss_decision,
                        winner,
                        win_by_runs,
                        win_by_wickets,
                        player_of_match
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    match_record,
                )
            except Exception as exc:
                print(f"[WARN] Failed to ingest match metadata from {info_file.name}: {exc}")
                continue

            if not deliveries_file.exists():
                print(f"[WARN] Missing delivery file for {match_id}: {deliveries_file.name}")
                continue

            try:
                delivery_rows = list(_iter_delivery_rows(deliveries_file))
                if delivery_rows:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO ipl_deliveries (
                            match_id,
                            innings,
                            over,
                            ball,
                            batting_team,
                            bowling_team,
                            striker,
                            non_striker,
                            bowler,
                            runs_off_bat,
                            extras,
                            wides,
                            noballs,
                            byes,
                            legbyes,
                            penalty,
                            wicket_type,
                            player_dismissed,
                            other_wicket_type,
                            other_player_dismissed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        delivery_rows,
                    )
            except Exception as exc:
                print(f"[WARN] Failed to ingest deliveries from {deliveries_file.name}: {exc}")

            if index % 50 == 0:
                conn.commit()

        conn.commit()


def validate_data(db_path: str | os.PathLike[str] | None = None) -> None:
    db_file = Path(db_path) if db_path is not None else _default_db_path()
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()

        total_matches = cur.execute("SELECT COUNT(*) FROM ipl_matches").fetchone()[0]
        total_deliveries = cur.execute("SELECT COUNT(*) FROM ipl_deliveries").fetchone()[0]
        date_range = cur.execute(
            "SELECT MIN(date), MAX(date) FROM ipl_matches"
        ).fetchone()
        season_rows = cur.execute(
            """
            SELECT season, COUNT(*)
            FROM ipl_matches
            GROUP BY season
            ORDER BY season
            """
        ).fetchall()
        batter_count = cur.execute(
            "SELECT COUNT(DISTINCT striker) FROM ipl_deliveries WHERE striker IS NOT NULL AND striker != ''"
        ).fetchone()[0]
        bowler_count = cur.execute(
            "SELECT COUNT(DISTINCT bowler) FROM ipl_deliveries WHERE bowler IS NOT NULL AND bowler != ''"
        ).fetchone()[0]

    print("\nIPL data validation")
    print(f"Total matches loaded: {total_matches}")
    print(f"Total deliveries loaded: {total_deliveries}")
    print("Season breakdown:")
    for season, count in season_rows:
        print(f"  {season}: {count}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    print(f"Count of distinct players who have batted: {batter_count}")
    print(f"Count of distinct players who have bowled: {bowler_count}")


if __name__ == "__main__":
    try:
        download_cricsheet_ipl()
        db_path = _default_db_path()
        ingest_to_sqlite(db_path)
        validate_data(db_path)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)

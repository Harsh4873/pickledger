#!/usr/bin/env python3
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path("pickledger.db")
STATE_FILE_PATH = Path("pickledger_state.json")
LEDGER_STATE_KEY = "primary"
SOURCE = "NBA Props Model"
DATE = "Mar 19"


DESIRED_RESULTS = {
    "Tobias Harris points OVER 13.0 vs WAS (Pistons @ Wizards)": "loss",
    "Jalen Duren points OVER 19.5 vs WAS (Pistons @ Wizards)": "push",
    "Jalen Duren assists OVER 1.0 vs WAS (Pistons @ Wizards)": "push",
    "Duncan Robinson assists UNDER 2.5 vs WAS (Pistons @ Wizards)": "win",
    "Alex Sarr assists OVER 2.0 vs DET (Pistons @ Wizards)": "loss",
    "Tre Johnson rebounds UNDER 3.5 vs DET (Pistons @ Wizards)": "win",
    "Desmond Bane assists UNDER 4.5 vs CHA (Magic @ Hornets)": "win",
    "Tristan da Silva rebounds UNDER 4.5 vs CHA (Magic @ Hornets)": "win",
    "Tristan da Silva assists UNDER 2.0 vs CHA (Magic @ Hornets)": "win",
    "Miles Bridges assists OVER 3.0 vs ORL (Magic @ Hornets)": "loss",
    "LaMelo Ball assists OVER 6.5 vs ORL (Magic @ Hornets)": "loss",
    "Coby White rebounds OVER 3.0 vs ORL (Magic @ Hornets)": "push",
    "James Harden rebounds UNDER 5.5 vs CHI (Cavaliers @ Bulls)": "push",
    "Evan Mobley assists OVER 3.0 vs CHI (Cavaliers @ Bulls)": "loss",
    "Sam Merrill rebounds UNDER 3.0 vs CHI (Cavaliers @ Bulls)": "win",
    "Sam Merrill assists UNDER 3.0 vs CHI (Cavaliers @ Bulls)": "push",
    "Matas Buzelis assists OVER 2.0 vs CLE (Cavaliers @ Bulls)": "push",
    "Rui Hachimura rebounds OVER 3.0 vs MIA (Lakers @ Heat)": "loss",
    "Rui Hachimura assists OVER 0.5 vs MIA (Lakers @ Heat)": "loss",
    "Marcus Smart rebounds OVER 2.5 vs MIA (Lakers @ Heat)": "loss",
    "Davion Mitchell rebounds UNDER 3.0 vs LAL (Lakers @ Heat)": "push",
    "Kawhi Leonard assists OVER 3.0 vs NOP (Clippers @ Pelicans)": "push",
    "John Collins assists UNDER 1.5 vs NOP (Clippers @ Pelicans)": "win",
    "Kris Dunn assists OVER 3.5 vs NOP (Clippers @ Pelicans)": "loss",
    "Derrick Jones Jr. assists UNDER 2.0 vs NOP (Clippers @ Pelicans)": "push",
    "Trey Murphy III assists UNDER 4.5 vs LAC (Clippers @ Pelicans)": "win",
    "Herbert Jones rebounds UNDER 4.0 vs LAC (Clippers @ Pelicans)": "win",
    "Herbert Jones assists UNDER 3.0 vs LAC (Clippers @ Pelicans)": "win",
    "Devin Booker rebounds UNDER 4.5 vs SAS (Suns @ Spurs)": "win",
    "Grayson Allen rebounds OVER 2.5 vs SAS (Suns @ Spurs)": "push",
    "Collin Gillespie rebounds UNDER 4.5 vs SAS (Suns @ Spurs)": "win",
    "Jordan Goodwin assists OVER 1.5 vs SAS (Suns @ Spurs)": "push",
    "Devin Vassell assists UNDER 3.0 vs PHX (Suns @ Spurs)": "win",
    "Paul George assists OVER 3.5 vs SAC (76ers @ Kings)": "push",
    "Dominick Barlow points OVER 8.0 vs SAC (76ers @ Kings)": "push",
    "Dominick Barlow assists OVER 1.0 vs SAC (76ers @ Kings)": "push",
    "DeMar DeRozan rebounds OVER 2.5 vs PHI (76ers @ Kings)": "push",
    "DeMar DeRozan assists UNDER 4.5 vs PHI (76ers @ Kings)": "push",
    "Daeqwon Plowden rebounds OVER 3.0 vs PHI (76ers @ Kings)": "loss",
    "Daeqwon Plowden assists OVER 1.0 vs PHI (76ers @ Kings)": "push",
    "Precious Achiuwa points OVER 9.5 vs PHI (76ers @ Kings)": "push",
    "Precious Achiuwa assists UNDER 2.0 vs PHI (76ers @ Kings)": "push",
    "Kevin Porter Jr. rebounds OVER 5.0 vs UTA (Bucks @ Jazz)": "push",
    "Kevin Porter Jr. assists OVER 7.0 vs UTA (Bucks @ Jazz)": "push",
    "Ryan Rollins points OVER 16.5 vs UTA (Bucks @ Jazz)": "loss",
    "AJ Green points OVER 9.0 vs UTA (Bucks @ Jazz)": "loss",
    "AJ Green assists OVER 1.5 vs UTA (Bucks @ Jazz)": "loss",
    "Myles Turner points OVER 11.5 vs UTA (Bucks @ Jazz)": "loss",
    "Myles Turner rebounds OVER 5.0 vs UTA (Bucks @ Jazz)": "loss",
    "Myles Turner assists OVER 1.0 vs UTA (Bucks @ Jazz)": "push",
    "Kyle Kuzma points OVER 13.0 vs UTA (Bucks @ Jazz)": "loss",
    "Svi Mykhailiuk points OVER 9.5 vs MIL (Bucks @ Jazz)": "push",
    "Svi Mykhailiuk rebounds UNDER 3.0 vs MIL (Bucks @ Jazz)": "push",
    "Kyle Filipowski assists UNDER 3.0 vs MIL (Bucks @ Jazz)": "push",
    "Cody Williams points OVER 7.5 vs MIL (Bucks @ Jazz)": "push",
    "Cody Williams rebounds UNDER 3.5 vs MIL (Bucks @ Jazz)": "push",
    "Cody Williams assists UNDER 2.0 vs MIL (Bucks @ Jazz)": "push",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state_file() -> dict:
    return json.loads(STATE_FILE_PATH.read_text(encoding="utf-8"))


def save_state_file(state: dict) -> None:
    STATE_FILE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def apply_results_to_state(state: dict) -> tuple[list[tuple[int, str, str]], list[str]]:
    added_picks = state.get("addedPicks")
    if not isinstance(added_picks, list):
        raise ValueError("State missing addedPicks list")

    results = state.get("results")
    if not isinstance(results, dict):
        results = {}
        state["results"] = results

    applied: list[tuple[int, str, str]] = []
    missing: list[str] = []

    for pick_text, desired in DESIRED_RESULTS.items():
        matches = [
            pick for pick in added_picks
            if pick.get("source") == SOURCE
            and pick.get("date") == DATE
            and pick.get("pick") == pick_text
        ]
        if len(matches) != 1:
            missing.append(pick_text)
            continue
        pick = matches[0]
        pick_id = int(pick["id"])
        results[str(pick_id)] = desired
        pick["result"] = desired
        applied.append((pick_id, pick_text, desired))

    state["savedAt"] = now_iso()
    return applied, missing


def update_sqlite_rows(conn: sqlite3.Connection, applied: list[tuple[int, str, str]]) -> list[str]:
    cur = conn.cursor()
    missing: list[str] = []
    stamp = now_iso()
    for pick_id, pick_text, desired in applied:
        cur.execute(
            """
            UPDATE picks
            SET result = ?, updated_at = ?
            WHERE id = ? AND source = ? AND date = ? AND pick = ?
            """,
            (desired, stamp, pick_id, SOURCE, DATE, pick_text),
        )
        if cur.rowcount == 0:
            missing.append(f"{pick_id}: {pick_text}")
    return missing


def save_sql_state(conn: sqlite3.Connection, state: dict) -> None:
    stamp = now_iso()
    conn.execute(
        """
        INSERT INTO ledger_state (state_key, state_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(state_key) DO UPDATE SET
            state_json = excluded.state_json,
            updated_at = excluded.updated_at
        """,
        (LEDGER_STATE_KEY, json.dumps(state, separators=(",", ":"), ensure_ascii=True), stamp),
    )


def main() -> None:
    state = load_state_file()
    applied, state_missing = apply_results_to_state(state)

    with sqlite3.connect(DB_PATH) as conn:
        db_missing = update_sqlite_rows(conn, applied)
        save_sql_state(conn, state)
        conn.commit()

    save_state_file(state)

    print(f"Applied {len(applied)} Mar 19 NBA props result fixes.")
    if state_missing:
        print("State entries not found:")
        for item in state_missing:
            print(f"  - {item}")
    if db_missing:
        print("DB rows not found (usually deleted from active ledger view):")
        for item in db_missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()

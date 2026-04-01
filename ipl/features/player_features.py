import sqlite3
import time

import pandas as pd


def build_player_features(db_path):
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            DROP TABLE IF EXISTS ipl_player_match_features;
            CREATE TABLE ipl_player_match_features (
                match_id TEXT,
                player_name TEXT,
                season TEXT,
                venue TEXT,
                player_team TEXT,
                opponent_team TEXT,
                toss_winner TEXT,
                toss_decision TEXT,
                runs_scored INT DEFAULT 0,
                balls_faced INT DEFAULT 0,
                fours INT DEFAULT 0,
                sixes INT DEFAULT 0,
                dismissed INT DEFAULT 0,
                strike_rate REAL,
                is_duck INT DEFAULT 0,
                milestone_50 INT DEFAULT 0,
                milestone_100 INT DEFAULT 0,
                pp_runs INT DEFAULT 0,
                mid_runs INT DEFAULT 0,
                death_runs INT DEFAULT 0,
                pp_sr REAL,
                death_sr REAL,
                overs_bowled REAL DEFAULT 0,
                balls_bowled INT DEFAULT 0,
                runs_conceded INT DEFAULT 0,
                wickets_taken INT DEFAULT 0,
                economy_rate REAL,
                maidens INT DEFAULT 0,
                dot_balls INT DEFAULT 0,
                pp_wickets INT DEFAULT 0,
                death_wickets INT DEFAULT 0,
                death_economy REAL,
                catches INT,
                stumpings INT,
                run_outs INT,
                PRIMARY KEY (match_id, player_name)
            );

            WITH phased AS (
              SELECT *,
                CASE WHEN "over"<=5 THEN 'PP'
                     WHEN "over"<=14 THEN 'MID'
                     ELSE 'DEATH' END AS phase
              FROM ipl_deliveries
            ),
            batting AS (
              SELECT match_id, striker AS player_name, batting_team AS player_team,
                SUM(runs_off_bat) AS runs_scored,
                SUM(CASE WHEN wides=0 THEN 1 ELSE 0 END) AS balls_faced,
                SUM(CASE WHEN runs_off_bat=4 THEN 1 ELSE 0 END) AS fours,
                SUM(CASE WHEN runs_off_bat=6 THEN 1 ELSE 0 END) AS sixes,
                MAX(CASE WHEN player_dismissed=striker THEN 1 ELSE 0 END) AS dismissed,
                SUM(CASE WHEN phase='PP'    THEN runs_off_bat ELSE 0 END) AS pp_runs,
                SUM(CASE WHEN phase='MID'   THEN runs_off_bat ELSE 0 END) AS mid_runs,
                SUM(CASE WHEN phase='DEATH' THEN runs_off_bat ELSE 0 END) AS death_runs,
                SUM(CASE WHEN phase='PP'    AND wides=0 THEN 1 ELSE 0 END) AS pp_balls,
                SUM(CASE WHEN phase='DEATH' AND wides=0 THEN 1 ELSE 0 END) AS death_balls
              FROM phased GROUP BY match_id, striker, batting_team
            ),
            bowling AS (
              SELECT match_id, bowler AS player_name, bowling_team AS player_team,
                COUNT(DISTINCT innings||'-'||"over") AS overs_bowled,
                SUM(CASE WHEN wides=0 AND noballs=0 THEN 1 ELSE 0 END) AS balls_bowled,
                SUM(runs_off_bat+COALESCE(wides,0)+COALESCE(noballs,0)) AS runs_conceded,
                SUM(CASE WHEN wicket_type IS NOT NULL
                         AND wicket_type NOT IN ('run out','retired hurt',
                         'retired out','obstructing the field') THEN 1 ELSE 0 END) AS wickets_taken,
                SUM(CASE WHEN phase='PP' AND wicket_type IS NOT NULL
                         AND wicket_type NOT IN ('run out','retired hurt',
                         'retired out','obstructing the field') THEN 1 ELSE 0 END) AS pp_wickets,
                SUM(CASE WHEN phase='DEATH' AND wicket_type IS NOT NULL
                         AND wicket_type NOT IN ('run out','retired hurt',
                         'retired out','obstructing the field') THEN 1 ELSE 0 END) AS death_wickets,
                SUM(CASE WHEN phase='DEATH'
                         THEN runs_off_bat+COALESCE(wides,0)+COALESCE(noballs,0)
                         ELSE 0 END) AS death_runs_conceded,
                COUNT(DISTINCT CASE WHEN phase='DEATH' THEN innings||'-'||"over" END) AS death_overs
              FROM phased GROUP BY match_id, bowler, bowling_team
            ),
            maiden_overs AS (
              SELECT match_id, bowler AS player_name, COUNT(*) AS maidens
              FROM (
                SELECT match_id, bowler, innings, "over",
                  SUM(runs_off_bat+COALESCE(wides,0)+COALESCE(noballs,0)
                      +COALESCE(byes,0)+COALESCE(legbyes,0)) AS over_runs
                FROM ipl_deliveries GROUP BY match_id, bowler, innings, "over"
              ) WHERE over_runs=0
              GROUP BY match_id, bowler
            ),
            dot_balls AS (
              SELECT match_id, bowler AS player_name,
                SUM(CASE WHEN runs_off_bat=0 AND COALESCE(wides,0)=0
                         AND COALESCE(noballs,0)=0 THEN 1 ELSE 0 END) AS dot_balls
              FROM ipl_deliveries GROUP BY match_id, bowler
            ),
            all_players AS (
              SELECT match_id, player_name, player_team FROM batting
              UNION
              SELECT match_id, player_name, player_team FROM bowling
            )
            INSERT OR REPLACE INTO ipl_player_match_features
            SELECT
              ap.match_id, ap.player_name, m.season, m.venue, ap.player_team,
              CASE WHEN ap.player_team=m.team1 THEN m.team2 ELSE m.team1 END AS opponent_team,
              m.toss_winner, m.toss_decision,
              COALESCE(b.runs_scored,0), COALESCE(b.balls_faced,0),
              COALESCE(b.fours,0), COALESCE(b.sixes,0), COALESCE(b.dismissed,0),
              CASE WHEN COALESCE(b.balls_faced,0)>0
                   THEN ROUND(CAST(b.runs_scored AS REAL)/b.balls_faced*100,2) END,
              CASE WHEN COALESCE(b.dismissed,0)=1 AND COALESCE(b.runs_scored,0)=0 THEN 1 ELSE 0 END,
              CASE WHEN COALESCE(b.runs_scored,0)>=50 AND COALESCE(b.runs_scored,0)<100 THEN 1 ELSE 0 END,
              CASE WHEN COALESCE(b.runs_scored,0)>=100 THEN 1 ELSE 0 END,
              COALESCE(b.pp_runs,0), COALESCE(b.mid_runs,0), COALESCE(b.death_runs,0),
              CASE WHEN COALESCE(b.pp_balls,0)>0
                   THEN ROUND(CAST(b.pp_runs AS REAL)/b.pp_balls*100,2) END,
              CASE WHEN COALESCE(b.death_balls,0)>0
                   THEN ROUND(CAST(b.death_runs AS REAL)/b.death_balls*100,2) END,
              COALESCE(bl.overs_bowled,0), COALESCE(bl.balls_bowled,0),
              COALESCE(bl.runs_conceded,0), COALESCE(bl.wickets_taken,0),
              CASE WHEN COALESCE(bl.overs_bowled,0)>0
                   THEN ROUND(CAST(bl.runs_conceded AS REAL)/bl.overs_bowled,2) END,
              COALESCE(mo.maidens,0), COALESCE(db.dot_balls,0),
              COALESCE(bl.pp_wickets,0), COALESCE(bl.death_wickets,0),
              CASE WHEN COALESCE(bl.death_overs,0)>0
                   THEN ROUND(CAST(bl.death_runs_conceded AS REAL)/bl.death_overs,2) END,
              NULL, NULL, NULL
            FROM all_players ap
            JOIN ipl_matches m ON ap.match_id=m.match_id
            LEFT JOIN batting b ON ap.match_id=b.match_id AND ap.player_name=b.player_name
            LEFT JOIN bowling bl ON ap.match_id=bl.match_id AND ap.player_name=bl.player_name
            LEFT JOIN maiden_overs mo ON ap.match_id=mo.match_id AND ap.player_name=mo.player_name
            LEFT JOIN dot_balls db ON ap.match_id=db.match_id AND ap.player_name=db.player_name;
            """
        )
        con.commit()
    finally:
        con.close()


def build_rolling_features(db_path, window=5):
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT f.*, m.date
            FROM ipl_player_match_features f
            JOIN ipl_matches m ON f.match_id = m.match_id
            """,
            con,
            parse_dates=["date"],
        )
        df = df.sort_values(["player_name", "date"]).reset_index(drop=True)

        feature_cols = [
            "runs_scored",
            "strike_rate",
            "wickets_taken",
            "economy_rate",
            "fours",
            "sixes",
        ]
        rename_map = {
            "runs_scored": "avg_runs_last5",
            "strike_rate": "avg_sr_last5",
            "wickets_taken": "avg_wickets_last5",
            "economy_rate": "avg_economy_last5",
            "fours": "avg_fours_last5",
            "sixes": "avg_sixes_last5",
        }

        def _add_rolling(group):
            group = group.sort_values("date").copy()
            rolling = group[feature_cols].rolling(window=window, min_periods=1).mean().shift(1)
            rolling = rolling.rename(columns=rename_map)
            group = group.join(rolling)
            group["matches_played_last5"] = (
                group["runs_scored"].rolling(window=window).count().shift(1)
            )
            return group

        df = df.groupby("player_name", group_keys=False).apply(_add_rolling)
        df.to_sql(
            "ipl_player_rolling_features",
            con,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=500,
        )
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ipl.data_loader import _default_db_path

    db = _default_db_path()

    t0 = time.time()
    print("Building per-match features (SQL CTE)...")
    build_player_features(db)
    print(f"  Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Building rolling features (pandas)...")
    build_rolling_features(db)
    print(f"  Done in {time.time()-t0:.1f}s")

    con = sqlite3.connect(db)
    rows = con.execute("SELECT COUNT(*) FROM ipl_player_match_features").fetchone()[0]
    roll = con.execute("SELECT COUNT(*) FROM ipl_player_rolling_features").fetchone()[0]
    print(f"\nipl_player_match_features:  {rows} rows")
    print(f"ipl_player_rolling_features: {roll} rows")
    print("\nTop 5 batting innings:")
    for row in con.execute(
        """
        SELECT player_name, runs_scored, balls_faced, fours, sixes, strike_rate
        FROM ipl_player_match_features ORDER BY runs_scored DESC LIMIT 5
        """
    ).fetchall():
        print(row)
    print("\nTop 5 bowling innings:")
    for row in con.execute(
        """
        SELECT player_name, wickets_taken, overs_bowled, runs_conceded, economy_rate
        FROM ipl_player_match_features ORDER BY wickets_taken DESC LIMIT 5
        """
    ).fetchall():
        print(row)
    con.close()

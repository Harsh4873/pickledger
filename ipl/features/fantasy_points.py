import sqlite3
import sys
import os
import time


def compute_fantasy_points(db_path):
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS ipl_player_fantasy_points (
              match_id              TEXT,
              player_name           TEXT,
              season                TEXT,
              player_team           TEXT,
              opponent_team         TEXT,
              batting_points        REAL DEFAULT 0,
              bowling_points        REAL DEFAULT 0,
              economy_points        REAL DEFAULT 0,
              sr_points             REAL DEFAULT 0,
              fielding_points       REAL DEFAULT 0,
              total_fantasy_points  REAL DEFAULT 0,
              PRIMARY KEY (match_id, player_name)
            );

            INSERT OR REPLACE INTO ipl_player_fantasy_points (
              match_id,
              player_name,
              season,
              player_team,
              opponent_team,
              batting_points,
              bowling_points,
              economy_points,
              sr_points,
              fielding_points,
              total_fantasy_points
            )
            WITH scored AS (
              SELECT
                match_id,
                player_name,
                season,
                player_team,
                opponent_team,

                (COALESCE(runs_scored, 0) * 1)
                + (COALESCE(fours, 0) * 1)
                + (COALESCE(sixes, 0) * 2)
                + CASE
                    WHEN milestone_100 = 1 THEN 16
                    WHEN milestone_50 = 1 THEN 8
                    ELSE 0
                  END
                + CASE
                    WHEN is_duck = 1 THEN -2
                    ELSE 0
                  END AS batting_points,

                CASE
                  WHEN COALESCE(overs_bowled, 0) >= 2 THEN
                    (COALESCE(wickets_taken, 0) * 25)
                    + CASE
                        WHEN wickets_taken >= 5 THEN 16
                        WHEN wickets_taken = 4 THEN 8
                        ELSE 0
                      END
                    + (COALESCE(maidens, 0) * 8)
                  ELSE 0
                END AS bowling_points,

                CASE
                  WHEN COALESCE(overs_bowled, 0) >= 2 AND economy_rate IS NOT NULL THEN
                    CASE
                      WHEN economy_rate < 4.0 THEN 6
                      WHEN economy_rate >= 4.0 AND economy_rate < 5.0 THEN 4
                      WHEN economy_rate >= 5.0 AND economy_rate < 6.0 THEN 2
                      WHEN economy_rate >= 9.0 AND economy_rate <= 10.0 THEN -2
                      WHEN economy_rate > 10.0 AND economy_rate <= 11.0 THEN -4
                      WHEN economy_rate > 11.0 THEN -6
                      ELSE 0
                    END
                  ELSE 0
                END AS economy_points,

                CASE
                  WHEN COALESCE(balls_faced, 0) >= 10 AND strike_rate IS NOT NULL THEN
                    CASE
                      WHEN strike_rate > 170 THEN 6
                      WHEN strike_rate > 150 AND strike_rate <= 170 THEN 4
                      WHEN strike_rate >= 130 AND strike_rate <= 150 THEN 2
                      WHEN strike_rate >= 60 AND strike_rate <= 70 THEN -2
                      WHEN strike_rate >= 50 AND strike_rate < 60 THEN -4
                      WHEN strike_rate < 50 THEN -6
                      ELSE 0
                    END
                  ELSE 0
                END AS sr_points,

                -- TODO Prompt 6: add fielding from live feed enrichment
                0 AS fielding_points

              FROM ipl_player_match_features
            )
            SELECT
              match_id,
              player_name,
              season,
              player_team,
              opponent_team,
              batting_points,
              bowling_points,
              economy_points,
              sr_points,
              fielding_points,
              batting_points + bowling_points + economy_points + sr_points + fielding_points
                AS total_fantasy_points
            FROM scored;
            """
        )
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )

    from ipl.data_loader import _default_db_path

    db = _default_db_path()

    t0 = time.time()
    print("Computing fantasy points...")
    compute_fantasy_points(db)
    print(f"  Done in {time.time()-t0:.1f}s")

    con = sqlite3.connect(db)
    rows = con.execute("SELECT COUNT(*) FROM ipl_player_fantasy_points").fetchone()[0]
    print(f"\nipl_player_fantasy_points: {rows} rows")

    print("\nTop 10 all-time fantasy performances:")
    for r in con.execute(
        """
        SELECT player_name, season, player_team, opponent_team,
               batting_points, bowling_points, economy_points,
               sr_points, total_fantasy_points
        FROM ipl_player_fantasy_points
        ORDER BY total_fantasy_points DESC
        LIMIT 10
    """
    ).fetchall():
        print(r)

    print("\nTop 5 bowlers by single-match fantasy score:")
    for r in con.execute(
        """
        SELECT player_name, season, bowling_points,
               economy_points, total_fantasy_points
        FROM ipl_player_fantasy_points
        WHERE bowling_points > 0
        ORDER BY bowling_points DESC
        LIMIT 5
    """
    ).fetchall():
        print(r)

    print("\nPoints distribution summary:")
    for r in con.execute(
        """
        SELECT
          AVG(total_fantasy_points)  AS avg_pts,
          MAX(total_fantasy_points)  AS max_pts,
          MIN(total_fantasy_points)  AS min_pts,
          SUM(CASE WHEN total_fantasy_points > 0 THEN 1 ELSE 0 END) AS positive_pts_rows,
          COUNT(*) AS total_rows
        FROM ipl_player_fantasy_points
    """
    ).fetchall():
        print(r)
    con.close()

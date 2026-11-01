"""Reshape MIMIC-IV-ECG reports into one row per statement, and list the unique statements.

Each ECG has up to 18 statement columns (report_0 ... report_17), most of them empty.
Outputs:
  data/silver/mimic_statements.parquet         one row per (study_id, position, statement)
  data/silver/mimic_unique_statements.parquet  one row per distinct statement, with how often it appears
"""
import time
from pathlib import Path

import duckdb

SILVER = Path("data/silver")


def main():
    SILVER.mkdir(parents=True, exist_ok=True)
    start = time.time()
    con = duckdb.connect()
    columns = ", ".join(f"report_{i}" for i in range(18))

    con.execute(f"""
        COPY (
            SELECT study_id, subject_id, position, trim(statement) AS statement
            FROM (
                SELECT study_id, subject_id,
                       unnest(range(18)) AS position,
                       unnest([{columns}]) AS statement
                FROM read_parquet('data/bronze/mimic_ecg_reports.parquet')
            )
            WHERE statement IS NOT NULL AND trim(statement) <> ''
        ) TO '{SILVER}/mimic_statements.parquet' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (
            SELECT statement, count(*) AS times_used, count(DISTINCT study_id) AS ecgs
            FROM read_parquet('{SILVER}/mimic_statements.parquet')
            GROUP BY statement
            ORDER BY times_used DESC
        ) TO '{SILVER}/mimic_unique_statements.parquet' (FORMAT parquet)
    """)

    ecgs, lines = con.execute(f"""SELECT count(DISTINCT study_id), count(*)
                                  FROM read_parquet('{SILVER}/mimic_statements.parquet')""").fetchone()
    unique = con.execute(f"SELECT count(*) FROM read_parquet('{SILVER}/mimic_unique_statements.parquet')").fetchone()[0]
    print(f"{ecgs:,} ECGs with statements, {lines:,} statement lines, {unique:,} unique statements "
          f"({lines / unique:,.0f}x repetition) in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()

"""Build the answer key from PTB-XL: one row per report with the cardiologists' heart-attack label.

Output: data/silver/ptbxl_answer_key.parquet
"""
from pathlib import Path

import duckdb

OUT = Path("data/silver/ptbxl_answer_key.parquet")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""
    COPY (
        WITH codes AS (
            -- scp_codes looks like {{'IMI': 50.0, 'SR': 0.0}}: turn it into JSON, then one row per code
            SELECT r.ecg_id,
                   unnest(json_keys(replace(r.scp_codes, '''', '"')::JSON)) AS code,
                   replace(r.scp_codes, '''', '"')::JSON AS codes_json
            FROM read_parquet('data/bronze/ptbxl_reports.parquet') AS r
        ),
        mi AS (
            SELECT c.ecg_id,
                   max(CAST(c.codes_json ->> c.code AS DOUBLE)) AS mi_likelihood
            FROM codes AS c
            JOIN read_parquet('data/bronze/ptbxl_statements.parquet') AS s
              -- the code column has no header in the CSV, so DuckDB names it column00
              ON s.column00 = c.code AND s.diagnostic_class = 'MI'
            GROUP BY c.ecg_id
        )
        SELECT r.ecg_id,
               trim(r.report)                      AS report,
               mi.ecg_id IS NOT NULL               AS label_mi,
               coalesce(mi.mi_likelihood, 0)       AS label_mi_likelihood,
               r.validated_by_human,
               r.strat_fold
        FROM read_parquet('data/bronze/ptbxl_reports.parquet') AS r
        LEFT JOIN mi USING (ecg_id)
    ) TO '{OUT}' (FORMAT parquet)
    """)
    total, mi = con.execute(f"SELECT count(*), sum(label_mi::INT) FROM '{OUT}'").fetchone()
    print(f"Answer key: {total:,} reports, {mi:,} labelled heart attack (MI) -> {OUT}")


if __name__ == "__main__":
    main()

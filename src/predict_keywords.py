"""Baseline: predict heart attack (MI) from report text using keywords only.

Output: data/predictions/keywords.parquet  (ecg_id, pred_mi)
"""
from pathlib import Path

import duckdb

OUT = Path("data/predictions/keywords.parquet")

# English and German words for infarction. Swedish ("infarkt") is covered by the German word.
KEYWORDS = "infarct|infarkt"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT ecg_id,
                   regexp_matches(lower(report), '{KEYWORDS}') AS pred_mi
            FROM read_parquet('data/silver/ptbxl_answer_key.parquet')
        ) TO '{OUT}' (FORMAT parquet)
    """)
    n = con.execute(f"SELECT sum(pred_mi::INT) FROM '{OUT}'").fetchone()[0]
    print(f"Keyword baseline: {n:,} reports predicted as heart attack -> {OUT}")


if __name__ == "__main__":
    main()

"""Classify MIMIC-IV-ECG statements for heart attack (MI), then flag all 800K ECGs.

Usage: python src/classify_mimic.py          (all unique statements)
       python src/classify_mimic.py 50       (only the 50 most common, as a quick test)

The LLM reads each UNIQUE statement once (cached), never each ECG. Answers are then
joined back to every ECG. An ECG is flagged if any of its statements is flagged.
Output: data/gold/mimic_ecg_mi_flags.parquet  (study_id, kw_mi, llm_mi, hybrid_mi, needs_review)
"""
import sys
import time
from pathlib import Path

import duckdb

from predict_llm import CACHE_DB, MODEL, PROMPT_VERSION, ask_model, cache_key, parse_answer

KEYWORDS = "infarct|infarkt"
OUT = Path("data/gold/mimic_ecg_mi_flags.parquet")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB))
    con.execute("""CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key VARCHAR PRIMARY KEY, report_text VARCHAR, model VARCHAR,
        prompt_version VARCHAR, answer BOOLEAN, raw_output VARCHAR, seconds DOUBLE)""")

    statements = [r[0] for r in con.execute(
        "SELECT statement FROM read_parquet('data/silver/mimic_unique_statements.parquet') ORDER BY times_used DESC"
    ).fetchall()]
    if limit:
        statements = statements[:limit]
    cached = {r[0] for r in con.execute("SELECT cache_key FROM llm_cache").fetchall()}
    todo = [s for s in statements if cache_key(s) not in cached]
    print(f"{len(statements):,} unique statements, {len(statements) - len(todo):,} already cached, "
          f"{len(todo):,} to ask the model")

    start = time.time()
    for i, text in enumerate(todo, 1):
        t0 = time.time()
        answer, raw = ask_model(text)
        con.execute("INSERT INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [cache_key(text), text, MODEL, PROMPT_VERSION, answer, raw, time.time() - t0])
        if i % 250 == 0 or i == len(todo):
            print(f"  {i:,}/{len(todo):,} done, {time.time() - start:.0f} s", flush=True)

    # Statement-level answers: keyword rule and LLM (re-parsed from raw output)
    raw_by_key = dict(con.execute(
        "SELECT cache_key, raw_output FROM llm_cache WHERE model = ? AND prompt_version = ?",
        [MODEL, PROMPT_VERSION]).fetchall())
    con.execute("CREATE OR REPLACE TEMP TABLE statement_answers (statement VARCHAR, llm_mi BOOLEAN)")
    con.executemany("INSERT INTO statement_answers VALUES (?, ?)",
                    [(s, parse_answer(raw_by_key.get(cache_key(s)))) for s in statements])

    # Join back to every ECG: flagged if ANY of its statements is flagged
    con.execute(f"""
        COPY (
            SELECT st.study_id,
                   bool_or(regexp_matches(lower(st.statement), '{KEYWORDS}')) AS kw_mi,
                   bool_or(coalesce(a.llm_mi, false))                       AS llm_mi
            FROM read_parquet('data/silver/mimic_statements.parquet') AS st
            JOIN statement_answers AS a USING (statement)
            GROUP BY st.study_id
        ) TO '{OUT}.tmp' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (SELECT study_id, kw_mi, llm_mi, (kw_mi OR llm_mi) AS hybrid_mi, (kw_mi <> llm_mi) AS needs_review
              FROM read_parquet('{OUT}.tmp')) TO '{OUT}' (FORMAT parquet)
    """)
    Path(f"{OUT}.tmp").unlink()

    n, kw, llm, hyb, review = con.execute(f"""
        SELECT count(*), sum(kw_mi::INT), sum(llm_mi::INT), sum(hybrid_mi::INT), sum(needs_review::INT)
        FROM read_parquet('{OUT}')""").fetchone()
    print(f"\nECGs covered: {n:,}")
    print(f"  flagged by keywords: {kw:,} | by LLM: {llm:,} | by hybrid: {hyb:,}")
    print(f"  needs human review (keywords and LLM disagree): {review:,} ({review / n:.1%})")
    print(f"  LLM calls this run: {len(todo):,} (instead of one per statement line)")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()

"""Choose how to combine keywords and the LLM on the validation fold, then score it once on the test fold.

Usage: python src/hybrid.py
Needs: data/predictions/keywords.parquet, llm_v2_fold9.parquet, llm_v2_fold10.parquet
Output: data/predictions/hybrid_fold10.parquet
"""
import duckdb

VALIDATION_FOLD, TEST_FOLD = 9, 10

# Each rule decides "heart attack: yes/no" from the two methods' answers
RULES = {
    "keywords only":     "kw",
    "LLM only":          "ai",
    "keywords OR LLM":   "kw OR ai",
    "keywords AND LLM":  "kw AND ai",
}


def load(con, fold):
    con.execute(f"""
        CREATE OR REPLACE TABLE fold{fold} AS
        SELECT k.ecg_id, k.label_mi, w.pred_mi AS kw, a.pred_mi AS ai
        FROM read_parquet('data/silver/ptbxl_answer_key.parquet') AS k
        JOIN read_parquet('data/predictions/keywords.parquet') AS w USING (ecg_id)
        JOIN read_parquet('data/predictions/llm_v2_fold{fold}.parquet') AS a USING (ecg_id)
        WHERE k.strat_fold = {fold}
    """)


def score(con, fold, rule):
    tp, fp, fn = con.execute(f"""
        SELECT sum(({rule}) AND label_mi)::INT, sum(({rule}) AND NOT label_mi)::INT,
               sum(NOT ({rule}) AND label_mi)::INT
        FROM fold{fold}
    """).fetchone()
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return recall, precision, fp, 2 * precision * recall / (precision + recall)


def main():
    con = duckdb.connect()
    load(con, VALIDATION_FOLD)
    load(con, TEST_FOLD)

    print(f"Choosing a rule on the VALIDATION fold ({VALIDATION_FOLD}):")
    results = {name: score(con, VALIDATION_FOLD, rule) for name, rule in RULES.items()}
    for name, (r, p, fp, f1) in results.items():
        print(f"  {name:18} recall {r:6.1%}  precision {p:6.1%}  false alarms {fp:4}  F1 {f1:.3f}")
    best = max(results, key=lambda n: results[n][3])
    print(f"  -> chosen: {best}")

    r, p, fp, f1 = score(con, TEST_FOLD, RULES[best])
    print(f"\nFinal score on the TEST fold ({TEST_FOLD}), measured once: {best}")
    print(f"  recall {r:.1%}  precision {p:.1%}  false alarms {fp}  F1 {f1:.3f}")

    n, agree, agree_right, disagree = con.execute(f"""
        SELECT count(*), sum((kw = ai)::INT), sum((kw = ai AND kw = label_mi)::INT), sum((kw <> ai)::INT)
        FROM fold{TEST_FOLD}
    """).fetchone()
    print(f"\nWhere keywords and LLM AGREE ({agree:,} of {n:,} reports): correct {agree_right / agree:.1%}")
    print(f"Where they DISAGREE: {disagree:,} reports ({disagree / n:.1%}) -> candidates for human review")

    con.execute(f"""COPY (SELECT ecg_id, ({RULES[best]}) AS pred_mi, (kw <> ai) AS needs_review
                     FROM fold{TEST_FOLD}) TO 'data/predictions/hybrid_fold{TEST_FOLD}.parquet' (FORMAT parquet)""")
    print(f"Saved -> data/predictions/hybrid_fold{TEST_FOLD}.parquet")


if __name__ == "__main__":
    main()

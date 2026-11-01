"""Score a prediction file against the PTB-XL answer key.

Usage: python src/evaluate.py data/predictions/keywords.parquet [fold]

With a fold number, only reports in that PTB-XL fold are scored.
"""
import sys

import duckdb


def main():
    pred_file = sys.argv[1]
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fold_filter = f"WHERE k.strat_fold = {fold}" if fold else ""
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE scored AS
        SELECT k.ecg_id, k.label_mi, k.label_mi_likelihood, p.pred_mi
        FROM read_parquet('data/silver/ptbxl_answer_key.parquet') AS k
        JOIN read_parquet('{pred_file}') AS p USING (ecg_id)
        {fold_filter}
    """)
    tp, fp, fn, tn, n = con.execute("""
        SELECT sum((pred_mi AND label_mi)::INT),
               sum((pred_mi AND NOT label_mi)::INT),
               sum((NOT pred_mi AND label_mi)::INT),
               sum((NOT pred_mi AND NOT label_mi)::INT),
               count(*)
        FROM scored
    """).fetchone()
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"Scored {n:,} reports from {pred_file}" + (f" (fold {fold})" if fold else ""))
    print(f"  Heart attacks found (recall):          {recall:.1%}  ({tp:,} of {tp + fn:,})")
    print(f"  Correct when it says yes (precision):  {precision:.1%}  ({fp:,} false alarms)")
    print(f"  F1 (balance of the two):               {f1:.3f}")

    print("\n  Recall by cardiologist certainty:")
    for row in con.execute("""
        SELECT label_mi_likelihood, count(*) AS n, avg(pred_mi::INT) AS found
        FROM scored WHERE label_mi GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"    {row[0]:>5.0f}% certain: found {row[2]:.0%} of {row[1]:,}")


if __name__ == "__main__":
    main()

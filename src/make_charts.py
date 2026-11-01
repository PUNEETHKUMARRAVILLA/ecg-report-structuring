"""Make README charts from the evaluation and MIMIC outputs. Output: docs/charts/*.png"""
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

CHARTS = Path("docs/charts")
BLUE, GREY = "#2a78d6", "#b7b5ad"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TEST_FOLD = 10


def new_chart(title, subtitle, height=5.0):
    fig, ax = plt.subplots(figsize=(10, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=34)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return fig, ax


def save(fig, name, source):
    fig.text(0.01, 0.01, source, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = CHARTS / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Chart saved -> {out}")


def scores(con, pred_file):
    tp, fp, fn = con.execute(f"""
        SELECT sum((p.pred_mi AND k.label_mi)::INT), sum((p.pred_mi AND NOT k.label_mi)::INT),
               sum((NOT p.pred_mi AND k.label_mi)::INT)
        FROM read_parquet('data/silver/ptbxl_answer_key.parquet') AS k
        JOIN read_parquet('{pred_file}') AS p USING (ecg_id)
        WHERE k.strat_fold = {TEST_FOLD}
    """).fetchone()
    return 100 * tp / (tp + fn), 100 * tp / (tp + fp)


def chart_methods():
    """Recall and precision of each method on the PTB-XL test fold."""
    con = duckdb.connect()
    methods = [
        ("Keywords", "data/predictions/keywords.parquet"),
        ("Local LLM (prompt v2)", f"data/predictions/llm_v2_fold{TEST_FOLD}.parquet"),
        ("Hybrid: keywords OR LLM", f"data/predictions/hybrid_fold{TEST_FOLD}.parquet"),
    ]
    results = [(name, *scores(con, f)) for name, f in methods]
    fig, ax = new_chart(
        "The hybrid finds 73% of heart attacks; keywords find 59%",
        f"PTB-XL test fold {TEST_FOLD} (2,198 reports, 550 labelled heart attack by cardiologists).\n"
        "Recall = share of real heart attacks found. Precision = share of 'yes' answers that were right.")
    y = range(len(results))
    h = 0.36
    for i, (name, recall, precision) in enumerate(results):
        ax.barh(i - h / 2, recall, height=h, color=BLUE, label="Recall" if i == 0 else None)
        ax.barh(i + h / 2, precision, height=h, color=GREY, label="Precision" if i == 0 else None)
        ax.text(recall + 1, i - h / 2, f"{recall:.1f}%", va="center", fontsize=9, color=INK)
        ax.text(precision + 1, i + h / 2, f"{precision:.1f}%", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in results], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percent", color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, "methods_test_fold.png",
         "Source: PTB-XL v1.0.3 (PhysioNet), cardiologist labels. Hybrid rule chosen on fold 9, scored once on fold 10.")


def chart_ai_calls():
    """How deduplication and caching cut the number of AI calls on MIMIC-IV-ECG."""
    con = duckdb.connect()
    lines, reports = con.execute("""
        SELECT count(*), count(DISTINCT study_id) FROM read_parquet('data/silver/mimic_statements.parquet')
    """).fetchone()
    unique = con.execute("SELECT count(*) FROM read_parquet('data/silver/mimic_unique_statements.parquet')").fetchone()[0]
    rows = [
        ("One call per statement line", lines),
        ("One call per ECG", reports),
        ("One call per unique statement (used)", unique),
    ]
    fig, ax = new_chart(
        f"{reports:,} ECGs classified with {unique:,} AI calls",
        "MIMIC-IV-ECG machine statements repeat heavily, so each unique statement is sent to the model once,\n"
        "cached, and joined back to every ECG. Log scale: each grid line is 10 times the one before.", height=4.2)
    for i, (name, n) in enumerate(rows):
        ax.barh(i, n, height=0.5, color=BLUE if i == len(rows) - 1 else GREY)
        ax.text(n * 1.15, i, f"{n:,}", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1_000, 30_000_000)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Number of AI calls (log scale)", color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, "ai_calls_needed.png",
         "Source: MIMIC-IV-ECG v1.0 machine_measurements.csv (PhysioNet). Local model: Qwen3 4B via Ollama.")


def main():
    chart_methods()
    chart_ai_calls()


if __name__ == "__main__":
    main()

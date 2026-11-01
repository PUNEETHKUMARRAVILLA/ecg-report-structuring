"""Predict heart attack (MI) from report text with a local LLM (Ollama), with a cache.

Usage: python src/predict_llm.py 10          (10 = PTB-XL test fold)

Each unique report text is sent to the model only once. Answers are stored in a
cache keyed by (report text, prompt version, model), so reruns cost nothing and
a prompt or model change is a new, separate set of answers.
Output: data/predictions/llm_fold<N>.parquet  (ecg_id, pred_mi)
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import duckdb
import requests

MODEL = "qwen3:4b"
PROMPT_VERSION = "v2"
# v1 asked about "infarction or myocardial damage", so the model counted ischemia as yes.
# v2 defines infarction precisely and lists the findings that are NOT infarction.
PROMPT = """You read ECG reports written by cardiologists (German, English or Swedish).
Question: does this report state or suggest a myocardial INFARCTION (heart attack), including old, possible,
probable, "cannot be ruled out" or "should be considered"?
Answer false for findings that are NOT infarction on their own: ischemia, ST or T wave changes,
repolarization changes, hypertrophy, bundle branch or conduction blocks, axis deviation, arrhythmias.
A clear negation ("no infarction") means false.
Report: {report}
Answer only with JSON: {{"mi": true}} or {{"mi": false}}"""

CACHE_DB = Path("data/cache/llm_cache.duckdb")


def cache_key(text):
    return hashlib.sha256(f"{MODEL}|{PROMPT_VERSION}|{text}".encode()).hexdigest()


def ask_model(report):
    """Returns (answer, raw_text). answer is True/False, or None if the output was not valid JSON."""
    reply = requests.post("http://localhost:11434/api/chat", json={
        "model": MODEL, "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0, "seed": 42, "num_predict": 20},
        "messages": [{"role": "user", "content": PROMPT.format(report=report)}],
    }, timeout=60)
    raw = reply.json()["message"]["content"]
    return parse_answer(raw), raw


def parse_answer(raw):
    """Read true/false from the model's raw output.

    Strict JSON first. If the JSON is cut off (the model sometimes prints
    {"mi": true followed by blank space until the token limit), read the value
    with a regular expression instead. Returns None only if there is no answer at all.
    """
    try:
        value = json.loads(raw).get("mi")
        if isinstance(value, bool):
            return value
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r'"mi"\s*:\s*(true|false)', raw or "")
    return (match.group(1) == "true") if match else None


def main():
    fold = int(sys.argv[1])
    out = Path(f"data/predictions/llm_fold{fold}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(CACHE_DB))
    con.execute("""CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key VARCHAR PRIMARY KEY, report_text VARCHAR, model VARCHAR,
        prompt_version VARCHAR, answer BOOLEAN, raw_output VARCHAR, seconds DOUBLE)""")

    reports = con.execute(f"""
        SELECT ecg_id, report FROM read_parquet('data/silver/ptbxl_answer_key.parquet')
        WHERE strat_fold = {fold}""").fetchall()
    unique_texts = sorted({r for _, r in reports})
    cached = {row[0] for row in con.execute("SELECT cache_key FROM llm_cache").fetchall()}
    todo = [t for t in unique_texts if cache_key(t) not in cached]
    print(f"Fold {fold}: {len(reports):,} reports, {len(unique_texts):,} unique texts, "
          f"{len(unique_texts) - len(todo):,} already cached, {len(todo):,} to ask the model")

    start = time.time()
    for i, text in enumerate(todo, 1):
        t0 = time.time()
        answer, raw = ask_model(text)
        con.execute("INSERT INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [cache_key(text), text, MODEL, PROMPT_VERSION, answer, raw, time.time() - t0])
        if i % 100 == 0 or i == len(todo):
            print(f"  {i:,}/{len(todo):,} done, {time.time() - start:.0f} s", flush=True)

    # Re-read every cached raw output with the current parser (free: no new model calls)
    answers = {row[0]: parse_answer(row[1]) for row in con.execute(
        "SELECT cache_key, raw_output FROM llm_cache WHERE model = ? AND prompt_version = ?",
        [MODEL, PROMPT_VERSION]).fetchall()}
    rows = [(ecg_id, answers.get(cache_key(text))) for ecg_id, text in reports]
    invalid = sum(1 for _, a in rows if a is None)

    con.execute("CREATE TEMP TABLE preds (ecg_id BIGINT, pred_mi BOOLEAN)")
    con.executemany("INSERT INTO preds VALUES (?, ?)", [(e, bool(a)) for e, a in rows])
    con.execute(f"COPY preds TO '{out}' (FORMAT parquet)")
    print(f"Predictions saved -> {out} ({invalid} invalid model outputs counted as 'no')")


if __name__ == "__main__":
    main()

# DEVLOG

## Day 1 — Oct 31, 2026

### Proving the problem first (before writing the project)
- Took 21,799 PTB-XL ECG reports (German, English, some Swedish) and checked plain keyword search
  against the cardiologists' labels. English "infarct" alone found only 16% of heart attacks.
- A "skip if the report says not" rule made it worse (recall 23%), because German
  "infarkt nicht auszuschliessen" means "cannot be ruled out", which is a possible case, not a no.
- Ran a free local model (Qwen3 4B on Ollama) on 200 random reports, twice. It was worse than keywords
  overall (F1 0.67 vs 0.79), with 8x more false alarms, and 11 of 200 answers changed between runs.
- Lesson: adding AI does not automatically improve data. It has to be measured.

### Setup and bronze
- Two datasets: PTB-XL (answer key, 21,799 reports) and MIMIC-IV-ECG (800,035 ECGs, for scale).
- PhysioNet downloads were slow (about 0.3 MB/s for the 183 MB MIMIC file), so I ran the download
  in the background.
- `ingest_bronze.py` checks each file's exact size before loading. It refused the half-downloaded
  MIMIC file twice (68 MB, then 71 MB), which would otherwise have become a silently incomplete table.
- Lesson: never load a file just because it exists. Check it is complete.

### Answer key
- Cardiologist labels are stored as text like {'IMI': 50.0}. Converted them to JSON, one row per code,
  joined with the statements dictionary, and kept the doctors' certainty (15% to 100%).
- Result: 5,469 of 21,799 reports labelled heart attack. Only 54% of those are 100% certain.
- Gotcha: the first column of scp_statements.csv has no header, so DuckDB named it `column00`.
- Bug while building it: a SQL `--` comment on the same line as the join condition hid
  `AND diagnostic_class = 'MI'`, and every report came out as a heart attack. Caught it because the
  count did not match the 5,469 I already knew.

### Keyword baseline
- Split "predict" and "evaluate" into separate scripts, so every method is scored the same way.
- Keywords on all reports: recall 62.1%, precision 93.1%, F1 0.745.
- Keywords find 77% of the 100%-certain cases but only 32% of the 15%-certain ones.

### Local LLM with a cache
- Fold 10 (PTB-XL's test fold): 2,198 reports but only 1,355 unique texts, so the model is asked
  1,355 times. Cache key = hash of model + prompt version + report text.
- First test run: one call never finished. With temperature 0 and forced JSON the model kept
  printing whitespace. Fixed with a 20-token cap and a 60 s timeout. Calls then took about 0.35 s.
- v1 result: recall 80.7% but precision 62.3% (269 false alarms), F1 0.703.
- Looked at the false alarms: 195 of 269 were ischemia or ST/T changes. My prompt asked about
  "infarction or myocardial damage", which was too broad. Wrote prompt v2 with a precise definition.

### The hidden "yes" answers
- v2 first looked worse (F1 0.647) with 116 invalid outputs.
- All 108 invalid v2 answers in the cache started with {"mi": true and then trailed into blank space
  until the token cap cut them. The parser rejected the broken JSON and counted them as "no".
- Fixed `parse_answer()`: strict JSON first, then read "mi": true/false with a regular expression.
  Because every raw output was stored, I re-scored everything without a single new model call.
- After the fix: v2 recall 72.9%, precision 74.4%, F1 0.736. Better than keywords (0.709 on fold 10).
- 185 of 2,198 answers (8.4%) changed between prompt v1 and v2.
- Lessons: never treat invalid output as "no" silently; always keep the raw output; version prompts.

### Hybrid, chosen fairly
- Tried 4 rules (keywords, LLM, OR, AND) on fold 9 only, then scored the winner once on fold 10.
- Chosen: keywords OR LLM. Fold 9 F1 0.743, fold 10 F1 0.738, so no sign of overfitting.
- On fold 10: recall 73.1% vs 59.3% for keywords. Where both methods agree (92% of reports) they are
  90.5% correct. The 7.8% where they disagree are flagged for human review.
- Honest note: "keywords OR LLM" is almost the same as the LLM alone (0.743 vs 0.742 on fold 9).

### 800K ECGs with 3,862 AI calls
- MIMIC statements sit in 18 columns per ECG. Reshaped wide to long: 2,760,117 statement lines,
  only 3,862 unique statements (715x repetition), in 0.3 s.
- Asked the model about each unique statement once (most common first), then joined the answers back
  to every ECG. 3,812 new calls in 1,327 s (about 22 minutes). A call per line would be about 11 days.
- Result on 800,034 ECGs: keywords flagged 179,588, LLM 182,132, hybrid 182,137.
  2,554 ECGs (0.3%) go to review. 0 invalid answers.
- Biggest catch: "*** CONSIDER ACUTE ST ELEVATION MI ***" appears 8,723 times. Keywords missed it
  because it says "MI", not "infarct". The model caught it.
- Some model "yes" answers are arguable ("ST elevation - cannot rule out myocardial injury").
  That is why disagreements go to a person.

"""Load the raw CSV files into bronze Parquet tables.

Usage: python src/ingest_bronze.py

Each raw file is checked for its full expected size first, so a half-finished
download is never loaded. Tables that already exist are skipped.
"""
import time
from pathlib import Path

import duckdb

RAW = Path("data/raw")
BRONZE = Path("data/bronze")

# table name -> (raw file, expected size in bytes from the PhysioNet server)
SOURCES = {
    "ptbxl_reports":     ("ptbxl_database.csv", 6_594_879),
    "ptbxl_statements":  ("scp_statements.csv", 9_720),
    "mimic_ecg_reports": ("mimic_machine_measurements.csv", 182_674_683),
}


def main():
    BRONZE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    for table, (filename, expected_bytes) in SOURCES.items():
        src = RAW / filename
        out = BRONZE / f"{table}.parquet"

        if out.exists():
            print(f"{table}: already loaded, skipping")
            continue

        # Guard: never load a file that is missing or still downloading
        size = src.stat().st_size if src.exists() else 0
        if size != expected_bytes:
            print(f"{table}: {filename} is {size:,} bytes, expected {expected_bytes:,}. "
                  f"Download not complete, skipping for now")
            continue

        start = time.time()
        tmp = out.with_suffix(".parquet.tmp")
        con.execute(f"COPY (SELECT * FROM read_csv('{src}', header=true, sample_size=-1)) "
                    f"TO '{tmp}' (FORMAT parquet)")
        tmp.rename(out)
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f"{table}: {rows:,} rows loaded in {time.time() - start:.1f} s -> {out}")


if __name__ == "__main__":
    main()

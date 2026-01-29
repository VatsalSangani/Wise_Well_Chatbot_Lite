import pandas as pd
from pathlib import Path

# -------- CONFIG --------
IN_PATH = Path("./kb/pubmed_abstracts_v1.parquet")
OUT_DIR = Path("./kb_partitioned")

YEAR_MIN = 2015
YEAR_MAX = 2025
MIN_ABSTRACT_LEN = 200
# ------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[load] reading parquet...")
df = pd.read_parquet(IN_PATH)

# Normalize + filter
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["abstract_len"] = df["abstract"].astype(str).str.len()

filtered = df[
    (df["year"].notna()) &
    (df["year"] >= YEAR_MIN) &
    (df["year"] <= YEAR_MAX) &
    (df["abstract_len"] >= MIN_ABSTRACT_LEN)
].copy()

filtered.drop(columns=["abstract_len"], inplace=True)

print(f"[info] input rows: {len(df)}")
print(f"[info] filtered rows: {len(filtered)}")

# Partition by year
for year in range(YEAR_MIN, YEAR_MAX + 1):
    part = filtered[filtered["year"] == year]

    if part.empty:
        print(f"[skip] year={year} (no rows)")
        continue

    year_dir = OUT_DIR / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)

    out_path = year_dir / "data.parquet"
    part.to_parquet(out_path, index=False)

    print(f"[done] year={year}, rows={len(part)}")

print("[all done]")

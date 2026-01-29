import re
from pathlib import Path
import pandas as pd

IN_DIR = Path("./kb_partitioned")   # <-- matches your folder
OUT_DIR = Path("./kb_chunks")       # <-- new folder to create
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ABSTRACT_CHARS = 200
CHUNK_WORDS = 220
OVERLAP_WORDS = 40

_ws = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    t = (t or "").strip()
    return _ws.sub(" ", t)

def chunk_words(words, chunk_words=220, overlap=40):
    if chunk_words <= overlap:
        raise ValueError("chunk_words must be > overlap")
    step = chunk_words - overlap
    out = []
    for start in range(0, len(words), step):
        end = start + chunk_words
        piece = words[start:end]
        if not piece:
            break
        out.append(" ".join(piece))
        if end >= len(words):
            break
    return out

def process_year_dir(year_dir: Path):
    in_file = year_dir / "data.parquet"
    if not in_file.exists():
        return

    year = year_dir.name.split("=", 1)[-1]
    df = pd.read_parquet(in_file)

    # Clean + filter
    df["abstract"] = df["abstract"].astype(str).map(normalize_text)
    df = df[df["abstract"].str.len() >= MIN_ABSTRACT_CHARS].copy()

    if df.empty:
        print(f"[skip] year={year} empty after filter")
        return

    df = df.drop_duplicates(subset=["pmid"])

    rows = []
    for _, r in df.iterrows():
        pmid = r["pmid"]
        y = int(r["year"]) if pd.notna(r["year"]) else None

        title = str(r.get("title", "")).strip()
        journal = str(r.get("journal", "")).strip()
        text = r["abstract"]

        words = text.split()
        chunks = chunk_words(words, CHUNK_WORDS, OVERLAP_WORDS)

        for idx, ch in enumerate(chunks):
            rows.append({
                "chunk_id": f"pubmed:{pmid}:y{y}:c{idx}",
                "doc_id": r.get("doc_id", f"pubmed:{pmid}"),
                "pmid": pmid,
                "year": y,
                "journal": journal,
                "title": title,
                "text": ch,
                "chunk_index": idx,
                "source_file": in_file.name,
            })

    out_df = pd.DataFrame(rows)

    out_path = OUT_DIR / f"year={year}"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "chunks.parquet"
    out_df.to_parquet(out_file, index=False)

    print(f"[done] year={year} docs={len(df)} chunks={len(out_df)} -> {out_file}")

def main():
    year_dirs = sorted([p for p in IN_DIR.glob("year=*") if p.is_dir()])
    if not year_dirs:
        raise SystemExit(f"No year partitions found in {IN_DIR}")

    for yd in year_dirs:
        process_year_dir(yd)

if __name__ == "__main__":
    main()

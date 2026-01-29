import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
YEARS = ["2023", "2024"]

CHUNKS_DIR = Path("./kb_chunks")
OUT_DIR = Path("./indexes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

# Filters to keep index quality high + memory reasonable
MIN_TEXT_CHARS = 80

# Optional safety valve (set to None to index all)
MAX_CHUNKS_PER_YEAR = None
# ---------------------------------------


def tokenize(text: str):
    # simple + fast lexical tokenization
    return text.lower().split()


def load_year_chunks(year: str) -> pd.DataFrame:
    fp = CHUNKS_DIR / f"year={year}" / "chunks.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing chunk parquet: {fp}")

    df = pd.read_parquet(fp)
    df["text"] = df["text"].astype(str)

    # Hard filters
    df = df[df["text"].str.len() >= MIN_TEXT_CHARS].copy()
    df = df.drop_duplicates(subset=["chunk_id"])

    # Keep only needed columns for runtime evidence/citations
    keep_cols = ["chunk_id", "pmid", "year", "title", "journal", "chunk_index", "text"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df.reset_index(drop=True, inplace=True)

    if MAX_CHUNKS_PER_YEAR is not None and len(df) > MAX_CHUNKS_PER_YEAR:
        df = df.sample(MAX_CHUNKS_PER_YEAR, random_state=42).reset_index(drop=True)

    return df


def build_bm25(df: pd.DataFrame, out_path: Path):
    corpus_tokens = [tokenize(t) for t in df["text"].tolist()]
    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "chunk_ids": df["chunk_id"].tolist(),  # aligns with BM25 doc index
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)


def build_faiss(df: pd.DataFrame, out_dir: Path, model: SentenceTransformer):
    texts = df["text"].tolist()

    # Encode in batches, add into FAISS incrementally to avoid giant RAM spikes
    # We'll store mapping via row order in id_map.parquet
    sample_emb = model.encode(["test"], normalize_embeddings=True)
    dim = int(sample_emb.shape[1])

    index = faiss.IndexFlatIP(dim)

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        index.add(emb)

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  [faiss] added {min(i + BATCH_SIZE, len(texts))}/{len(texts)} vectors")

    # Save FAISS
    faiss.write_index(index, str(out_dir / "index.faiss"))

    # Save id map (row_id -> chunk metadata + text)
    df.to_parquet(out_dir / "id_map.parquet", index=False)

    meta = {
        "embed_model": EMBED_MODEL,
        "normalize_embeddings": True,
        "faiss_metric": "cosine_via_inner_product",
        "num_vectors": int(index.ntotal),
        "dim": dim,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    print("[load] embedding model:", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    for year in YEARS:
        print(f"\n[year] {year}")
        df = load_year_chunks(year)
        print(f"[info] chunks: {len(df)}")

        year_dir = OUT_DIR / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)

        # BM25
        bm25_path = year_dir / "bm25.pkl"
        print(f"[build] BM25 -> {bm25_path}")
        build_bm25(df, bm25_path)

        # FAISS
        faiss_dir = year_dir / "faiss"
        faiss_dir.mkdir(parents=True, exist_ok=True)

        print(f"[build] FAISS -> {faiss_dir}")
        build_faiss(df, faiss_dir, model)

        print(f"[done] year={year}")

    print("\n[all done] Indexes built in ./indexes")


if __name__ == "__main__":
    main()

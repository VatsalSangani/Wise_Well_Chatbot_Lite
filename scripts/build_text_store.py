"""
scripts/build_text_store.py

Build the WiseWell SQLite text store from the per-year id_map.parquet files.

WHY: After the Pinecone migration, Pinecone returns chunk_ids + lean metadata
(pmid, year, chunk_index, title, journal) but NOT the chunk `text` body. We
rehydrate the text from this SQLite DB by chunk_id. This removes the ~400MB
in-RAM id_map DataFrame from the FastAPI process — the DB lives on disk and is
queried per request.

Run ONCE (and again only if the source parquet files change):

    py scripts/build_text_store.py
    py scripts/build_text_store.py --force        # rebuild from scratch

The DB is keyed by chunk_id (PRIMARY KEY), aligned 1:1 with the Pinecone
vector ids. pmid/title/journal are stored too for convenience, but those also
come back from Pinecone metadata — `text` is the column that matters.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Import paths/config without requiring the package to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INDEXES_ROOT, WISEWELL_YEARS  # noqa: E402

# Default DB location: kb/text_store.sqlite (next to the indexes tree).
DEFAULT_DB_PATH = Path(INDEXES_ROOT).parent / "text_store.sqlite"

# Columns we copy from id_map.parquet. chunk_id + text are load-bearing;
# the rest are convenience duplicates of Pinecone metadata.
_COLUMNS = ["chunk_id", "pmid", "year", "title", "journal", "chunk_index", "text"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    pmid        TEXT,
    year        INTEGER,
    title       TEXT,
    journal     TEXT,
    chunk_index INTEGER,
    text        TEXT NOT NULL
);
"""

_INSERT = """
INSERT OR REPLACE INTO chunks
    (chunk_id, pmid, year, title, journal, chunk_index, text)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

_BATCH = 5_000


def _id_map_paths() -> list[Path]:
    root = Path(INDEXES_ROOT)
    paths = []
    for year in WISEWELL_YEARS:
        p = root / f"year={year}" / "faiss" / "id_map.parquet"
        if not p.exists():
            raise FileNotFoundError(f"id_map parquet not found for year={year}: {p}")
        paths.append(p)
    return paths


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # Bulk-load tuning — safe because this is a one-shot offline build.
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.executescript(_SCHEMA)
    return conn


def build(db_path: Path, force: bool) -> None:
    if db_path.exists() and force:
        db_path.unlink()
        print(f"[build] removed existing DB at {db_path}")

    conn = _open_db(db_path)
    total = 0
    try:
        for src in _id_map_paths():
            df = pd.read_parquet(src, columns=_COLUMNS)
            # Normalize: chunk_id/pmid as str, text never null.
            df["chunk_id"] = df["chunk_id"].astype(str)
            df["pmid"] = df["pmid"].astype(str)
            df["text"] = df["text"].fillna("").astype(str)

            rows = list(
                df[_COLUMNS].itertuples(index=False, name=None)
            )
            for i in range(0, len(rows), _BATCH):
                conn.executemany(_INSERT, rows[i : i + _BATCH])
            conn.commit()
            total += len(rows)
            print(f"[build] {src.parent.parent.name}: inserted {len(rows):,} chunks")

        # Final integrity check.
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"[build] done — {count:,} rows in {db_path} (read {total:,})")
        if count != total:
            print(
                f"[build] NOTE: {total - count:,} duplicate chunk_ids collapsed "
                "by PRIMARY KEY (expected if chunk_ids repeat across years)."
            )
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the WiseWell SQLite text store.")
    ap.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Output SQLite path (default: {DEFAULT_DB_PATH})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the DB from scratch.",
    )
    args = ap.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    build(args.db, force=args.force)


if __name__ == "__main__":
    main()

"""
retrieval/text_store.py

Fast, low-RAM lookup of chunk text by chunk_id from the SQLite text store
(built by scripts/build_text_store.py). Replaces the ~400MB in-RAM id_map
DataFrame: Pinecone returns chunk_ids + lean metadata, and this fetches the
`text` bodies on demand.

The connection is opened read-only and lazily, so importing this module never
blocks FastAPI cold start and never holds the corpus in process RAM.

Usage:
    from retrieval.text_store import get_text_store
    store = get_text_store()
    rows = store.fetch(["pubmed:37510304:y2023:c0", ...])   # one query
    text = store.fetch_text("pubmed:37510304:y2023:c0")
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

from config import INDEXES_ROOT

# Keep in sync with scripts/build_text_store.py DEFAULT_DB_PATH.
DEFAULT_DB_PATH = Path(INDEXES_ROOT).parent / "text_store.sqlite"

# SQLite caps host parameters per statement (SQLITE_MAX_VARIABLE_NUMBER).
# 900 stays safely under the historical 999 limit.
_VAR_LIMIT = 900

_COLUMNS = ["chunk_id", "pmid", "year", "title", "journal", "chunk_index", "text"]


class TextStore:
    """Read-only, thread-safe accessor over the SQLite text store."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    if not self.db_path.exists():
                        raise FileNotFoundError(
                            f"Text store not found at {self.db_path}. "
                            "Run: py scripts/build_text_store.py"
                        )
                    # Read-only URI + check_same_thread=False so the single
                    # connection can serve FastAPI's worker threads. Reads are
                    # serialized by SQLite; the GIL + short queries keep this cheap.
                    self._conn = sqlite3.connect(
                        f"file:{self.db_path}?mode=ro",
                        uri=True,
                        check_same_thread=False,
                    )
                    self._conn.row_factory = sqlite3.Row
        return self._conn

    def fetch(self, chunk_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch rows for the given chunk_ids. Returns {chunk_id: {col: value}}.
        Missing ids are simply absent from the result. Order-independent;
        callers re-order by their own ranking.
        """
        if not chunk_ids:
            return {}

        conn = self._connect()
        out: Dict[str, Dict] = {}
        with self._lock:
            for i in range(0, len(chunk_ids), _VAR_LIMIT):
                batch = chunk_ids[i : i + _VAR_LIMIT]
                placeholders = ",".join("?" * len(batch))
                cur = conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM chunks "
                    f"WHERE chunk_id IN ({placeholders})",
                    batch,
                )
                for row in cur.fetchall():
                    out[row["chunk_id"]] = dict(row)
        return out

    def fetch_text(self, chunk_id: str) -> Optional[str]:
        """Convenience: fetch the text body for a single chunk_id (or None)."""
        row = self.fetch([chunk_id]).get(chunk_id)
        return row["text"] if row else None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


_store: Optional[TextStore] = None
_store_lock = threading.Lock()


def get_text_store() -> TextStore:
    """Process-wide singleton (connection opened lazily on first fetch)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TextStore()
    return _store

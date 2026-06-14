"""
scripts/pinecone_smoke_test.py

Live smoke test for the Pinecone dense path. Bypasses query_rewrite so we can
query an EXACT string (to compare raw vs. rewritten side by side), then runs
the retriever's full path once to confirm embed->match->rehydrate->text works.

Run: py scripts/pinecone_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Medical abstracts contain Greek letters (α/β/μ). Force UTF-8 stdout so this
# diagnostic doesn't crash on a cp1252 Windows console (the production HTTP path
# is unaffected — Starlette JSONResponse already emits UTF-8).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.embedder import embed  # noqa: E402
from retrieval.text_store import get_text_store  # noqa: E402
from retrieval.pinecone_retriever import get_pinecone_retriever  # noqa: E402


def raw_query(label: str, text: str, k: int = 5) -> None:
    """Embed `text` verbatim and show top-k Pinecone scores + titles."""
    r = get_pinecone_retriever()
    index = r._get_index()
    res = index.query(vector=embed(text), top_k=k, include_metadata=True)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches
    print(f"\n=== {label} ===\n    query string: {text!r}")
    for i, m in enumerate(matches, 1):
        cid = m["id"] if isinstance(m, dict) else m.id
        score = m["score"] if isinstance(m, dict) else m.score
        meta = (m["metadata"] if isinstance(m, dict) else m.metadata) or {}
        print(f"  {i}. {score:.4f}  {str(meta.get('title',''))[:90]}")


def full_path(query: str, k: int = 5) -> None:
    """Run the retriever end-to-end and confirm text rehydration."""
    r = get_pinecone_retriever()
    results, debug = r.retrieve(query, top_k=k, return_debug=True)
    print(f"\n=== FULL PATH: {query!r} ===")
    print(f"    debug: {debug}")
    for i, e in enumerate(results, 1):
        print(f"  {i}. {e['score']:.4f}  PMID:{e['pmid']} ({e['year']})  {e['title'][:80]}")
        print(f"       text[:120]: {e['text'][:120]!r}")
        assert e["text"], f"empty text for {e['chunk_id']}"
    print(f"    -> {len(results)} results, all with rehydrated text")


if __name__ == "__main__":
    print("text store rows:", get_text_store().fetch  # touch singleton
          and "ready")
    # Compare the lab-value query raw vs. its rewrite output.
    raw_query("RAW lab-value query", "CRP 18 mg/L")
    raw_query("REWRITTEN lab-value query", "crp clinical significance elevated")
    # Acronym query through the full retriever path.
    full_path("IL-6 inhibitors rheumatoid arthritis")

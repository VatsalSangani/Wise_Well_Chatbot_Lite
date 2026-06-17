"""
retrieval/pinecone_retriever.py

Dense retrieval from Pinecone, replacing the in-RAM hybrid BM25+FAISS path.

Flow per query:
  1. Lab-value rewrite (query_rewrite) — "CRP 18 mg/L" -> conceptual query.
  2. embed() the (possibly rewritten) query via the lazy MiniLM seam.
  3. Pinecone dense query -> matches: vector id (== chunk_id) + lean metadata
     (pmid, year, chunk_index, title, journal). NO text is stored in Pinecone.
  4. Rehydrate the `text` body for each chunk_id from the SQLite text store.
  5. Return evidence in the SAME stable schema as HybridRetriever so it's a
     drop-in for the graph and guardrails.

Everything is lazy-loaded (Pinecone client, index handle, MiniLM, SQLite
connection) so importing this module never blocks FastAPI cold start.

The old retrieval/hybrid_retriever.py is intentionally left in place as the
rollback path — this module does not touch it.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import structlog

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K
from retrieval.embedder import embed
from retrieval.query_rewrite import rewrite_query
from retrieval.text_store import get_text_store

logger = structlog.get_logger()


class PineconeRetriever:
    """Dense retriever over the populated `wisewell-abstracts` Pinecone index."""

    def __init__(
        self,
        index_name: str = PINECONE_INDEX_NAME,
        api_key: str = PINECONE_API_KEY,
    ) -> None:
        self.index_name = index_name
        self._api_key = api_key
        self._index = None
        self._index_lock = threading.Lock()
        self._text_store = get_text_store()

    def _get_index(self):
        """Lazy-init the Pinecone client + index handle on first query."""
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    if not self._api_key:
                        raise RuntimeError(
                            "PINECONE_API_KEY is not set. Add it to .env "
                            "(loaded via python-dotenv in backend/main.py)."
                        )
                    # Imported here so cold start doesn't pay the import cost.
                    from pinecone import Pinecone

                    pc = Pinecone(api_key=self._api_key)
                    self._index = pc.Index(self.index_name)
                    logger.info("pinecone_index_ready", index=self.index_name)
        return self._index

    @staticmethod
    def _normalize_chunk(
        *,
        chunk_id: str,
        pmid,
        year,
        title: str,
        journal: str,
        text: str,
        score: float,
    ) -> Dict[str, Any]:
        """Match HybridRetriever's evidence schema exactly (drop-in contract).
        `hits` keeps the same shape; dense-only, so faiss=True / bm25=False."""
        return {
            "chunk_id": str(chunk_id),
            "pmid": str(pmid) if pmid is not None else None,
            "year": int(year) if year is not None else None,
            "title": str(title or ""),
            "journal": str(journal or ""),
            "text": str(text or ""),
            "score": float(score),
            "hits": {"bm25": False, "faiss": True},
        }

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        max_chunks_per_pmid: int = 0,
        oversample_factor: int = 3,
        return_debug: bool = False,
    ) -> Any:
        if not query or not query.strip():
            empty: List[Dict] = []
            return empty if not return_debug else (empty, {"reason": "empty_query"})

        # 1) Lab-value rewrite (no-op for non-lab queries).
        rw = rewrite_query(query)
        if rw.rewritten:
            logger.info(
                "query_rewritten",
                original=rw.original,
                rewritten=rw.query,
                biomarker=rw.biomarker,
            )

        # 2) Embed via the seam.
        qvec = embed(rw.query)

        # 3) Dense query. Oversample when capping per-PMID so we still hit top_k.
        pool = top_k * max(1, oversample_factor) if max_chunks_per_pmid else top_k
        index = self._get_index()
        res = index.query(vector=qvec, top_k=pool, include_metadata=True)
        matches = res.get("matches", []) if isinstance(res, dict) else res.matches

        # 4) Rehydrate text bodies in ONE SQLite query.
        chunk_ids = [m["id"] if isinstance(m, dict) else m.id for m in matches]
        bodies = self._text_store.fetch(chunk_ids)

        # 5) Normalize to the stable evidence schema, preserving Pinecone order.
        results: List[Dict[str, Any]] = []
        for m in matches:
            cid = m["id"] if isinstance(m, dict) else m.id
            score = m["score"] if isinstance(m, dict) else m.score
            meta = (m["metadata"] if isinstance(m, dict) else m.metadata) or {}
            row = bodies.get(cid)
            if row is None:
                # Vector exists in Pinecone but text missing from SQLite — skip
                # rather than emit a citation with no body.
                logger.warning("text_store_miss", chunk_id=cid)
                continue
            results.append(
                self._normalize_chunk(
                    chunk_id=cid,
                    # Prefer SQLite metadata; fall back to Pinecone metadata.
                    pmid=row.get("pmid") or meta.get("pmid"),
                    year=row.get("year") or meta.get("year"),
                    title=row.get("title") or meta.get("title", ""),
                    journal=row.get("journal") or meta.get("journal", ""),
                    text=row.get("text", ""),
                    score=score,
                )
            )

        if max_chunks_per_pmid:
            results = self._limit_chunks_per_pmid(results, max_chunks_per_pmid)
        results = results[:top_k]

        if not return_debug:
            return results

        distinct_pmids = len({r["pmid"] for r in results if r.get("pmid")})
        debug = {
            "index": self.index_name,
            "top_k": top_k,
            "pool": pool,
            "query_rewritten": rw.rewritten,
            "rewritten_query": rw.query if rw.rewritten else None,
            "biomarker": rw.biomarker,
            "matches_returned": len(matches),
            "text_rehydrated": len(results),
            "distinct_pmids": distinct_pmids,
        }
        return results, debug

    @staticmethod
    def _limit_chunks_per_pmid(results: List[Dict], max_per_pmid: int) -> List[Dict]:
        """Keep at most max_per_pmid chunks per PMID (results best->worst)."""
        if max_per_pmid <= 0:
            return results
        counts: Dict[str, int] = {}
        out: List[Dict] = []
        for r in results:
            pmid = r.get("pmid")
            if pmid is None:
                out.append(r)
                continue
            counts[pmid] = counts.get(pmid, 0) + 1
            if counts[pmid] <= max_per_pmid:
                out.append(r)
        return out


_retriever: Optional[PineconeRetriever] = None
_retriever_lock = threading.Lock()


def get_pinecone_retriever() -> PineconeRetriever:
    """Process-wide singleton (all heavy resources lazy-loaded on first query)."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = PineconeRetriever()
    return _retriever

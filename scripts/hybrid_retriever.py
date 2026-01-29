"""
scripts/hybrid_retriever.py

WiseWell Hybrid Retriever (BM25 + FAISS + RRF) across years (e.g., 2023/2024).

Your project structure (from your screenshot):
project_root/
  indexes/
    year=2023/
    year=2024/
  scripts/
    hybrid_retriever.py
    ...

So the default indexes_root MUST point to ../indexes (relative to this file).

What this file includes:
- Correct cross-year ranking:
    (1) pull per-year candidates (BM25 + FAISS) with scores
    (2) merge across years by SCORE per retriever (not concatenation)
    (3) RRF fuse the two global rank lists
- Query embedding computed ONCE per request
- Robust FAISS id->chunk_id mapping:
    uses 'faiss_id' column if present, otherwise assumes dataframe row order == FAISS insertion order
- Evidence diversity:
    optional cap on number of chunks per PMID
    oversampling before RRF so you still get exactly top_k after dedupe
- Debug signals (for gating/monitoring)
- Deterministic Evidence Gate (optional helper)

Expected index layout:
  indexes_root/
    year=2023/
      bm25.pkl                   # dict: {"bm25": <rank_bm25>, "chunk_ids": [chunk_id,...]}
      faiss/
        index.faiss
        id_map.parquet           # must include "chunk_id"; ideally includes "faiss_id" and metadata/text
    year=2024/
      ...

id_map.parquet NOTE:
- This code assumes id_map.parquet includes at least: chunk_id
- If you want evidence payload directly from id_map, include: pmid, year, title, journal, text
- Otherwise, replace _materialize_chunk() to fetch from your chunk store (sqlite/parquet).
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> List[str]:
    """Lightweight tokenizer that keeps hyphens (helps IL-6, TNF-alpha, etc.)."""
    out: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "-":
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur).strip("-")
                if tok:
                    out.append(tok)
                cur = []
    if cur:
        tok = "".join(cur).strip("-")
        if tok:
            out.append(tok)
    return out


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


class HybridRetriever:
    """
    Hybrid retrieval using:
      - BM25 (rank_bm25)
      - FAISS cosine similarity (IndexFlatIP on normalized embeddings)
      - RRF fusion

    retrieve(query) -> evidence chunks (STABLE CONTRACT):
      {
        "chunk_id": str,
        "pmid": str|None,
        "year": int|None,
        "title": str,
        "journal": str,
        "text": str,
        "score": float,                 # unified score (RRF score)
        "hits": {"bm25": bool, "faiss": bool}
      }

    Optionally:
      - max_chunks_per_pmid: cap chunks returned per PMID (diversity)
      - return_debug: include debug signals useful for gating/monitoring
    """

    def __init__(
        self,
        indexes_root: str = "../indexes",   # IMPORTANT: matches your file structure
        years: List[str] = ["2023", "2024"],
        integrity_check: bool = True,
    ):
        # Resolve indexes_root relative to THIS FILE (Option B)
        base_dir = Path(__file__).resolve().parent  # .../scripts
        p = Path(indexes_root)
        if p.is_absolute():
            self.root = p
        else:
            self.root = (base_dir / p).resolve()    # .../indexes

        self.years = [str(y) for y in years]

        # Load embedding model once
        self.model = SentenceTransformer(EMBED_MODEL)

        # Per-year stores
        self.bm25: Dict[str, Any] = {}
        self.bm25_chunk_ids: Dict[str, List[str]] = {}

        self.faiss_index: Dict[str, faiss.Index] = {}
        self.id_map_df: Dict[str, pd.DataFrame] = {}

        # FAISS internal id -> chunk_id
        self.faissid_to_chunkid: Dict[str, Dict[int, str]] = {}

        # Global lookup chunk_id -> (year, df_row_idx) for materialization
        self.chunk_to_row: Dict[str, Tuple[str, int]] = {}

        dupes: List[str] = []

        for y in self.years:
            base = self.root / f"year={y}"
            if not base.exists():
                raise FileNotFoundError(
                    f"Missing index directory: {base}\n"
                    f"Resolved indexes_root: {self.root}\n"
                    f"Expected: {self.root}/year={y}/"
                )

            # BM25
            with (base / "bm25.pkl").open("rb") as f:
                obj = pickle.load(f)
            if "bm25" not in obj or "chunk_ids" not in obj:
                raise ValueError(f"bm25.pkl for year={y} must contain keys: 'bm25', 'chunk_ids'")
            self.bm25[y] = obj["bm25"]
            self.bm25_chunk_ids[y] = list(map(str, obj["chunk_ids"]))

            # FAISS
            self.faiss_index[y] = faiss.read_index(str(base / "faiss" / "index.faiss"))

            # id_map / metadata table
            df = pd.read_parquet(base / "faiss" / "id_map.parquet")
            if "chunk_id" not in df.columns:
                raise ValueError(f"id_map.parquet for year={y} must include 'chunk_id'")
            df = df.reset_index(drop=True)
            df["chunk_id"] = df["chunk_id"].astype(str)
            self.id_map_df[y] = df

            # chunk_id -> (year, row_idx)
            for i, cid in enumerate(df["chunk_id"].tolist()):
                if cid in self.chunk_to_row:
                    dupes.append(cid)
                self.chunk_to_row[cid] = (y, i)

            # faiss_id -> chunk_id mapping (preferred)
            if "faiss_id" in df.columns:
                self.faissid_to_chunkid[y] = dict(
                    zip(df["faiss_id"].astype(int).tolist(), df["chunk_id"].tolist())
                )
            else:
                # fallback assumes row order matches insertion order
                self.faissid_to_chunkid[y] = {i: cid for i, cid in enumerate(df["chunk_id"].tolist())}

        if integrity_check and dupes:
            sample = ", ".join(dupes[:10])
            print(
                f"[WARN] Duplicate chunk_id values detected across loaded data (sample: {sample}). "
                "If chunk_id is not globally unique, evidence materialization can be wrong. "
                "Use a globally unique chunk_id like: f'pubmed:{pmid}:y{year}:c{chunk_index}'."
            )

    def _bm25_search_year_scored(self, year: str, query: str, k: int) -> List[Tuple[str, float]]:
        tokens = tokenize(query)
        scores = np.asarray(self.bm25[year].get_scores(tokens), dtype=np.float32)
        if scores.size == 0:
            return []
        k = min(k, scores.size)

        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        ids = self.bm25_chunk_ids[year]
        return [(ids[int(i)], float(scores[int(i)])) for i in idx]

    def _faiss_search_year_scored(self, year: str, qvec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        q = qvec.astype("float32")[None, :]
        scores, ids = self.faiss_index[year].search(q, k)

        ids = ids[0]
        scores = scores[0]

        mapper = self.faissid_to_chunkid[year]
        out: List[Tuple[str, float]] = []
        for fid, sc in zip(ids.tolist(), scores.tolist()):
            if fid < 0:
                continue
            cid = mapper.get(int(fid))
            if cid is None:
                continue
            out.append((cid, float(sc)))
        return out

    @staticmethod
    def _merge_top_by_score(
        scored_lists: List[List[Tuple[str, float]]],
        top_k: int,
        mode: str = "max",
    ) -> List[Tuple[str, float]]:
        """
        Merge multiple (chunk_id, score) lists into one global top_k by score.

        mode:
          - "max": keep max score per chunk_id (recommended)
          - "sum": sum scores per chunk_id
        """
        if mode not in {"max", "sum"}:
            raise ValueError("mode must be 'max' or 'sum'")

        agg: Dict[str, float] = {}
        for lst in scored_lists:
            for cid, sc in lst:
                if mode == "max":
                    prev = agg.get(cid)
                    agg[cid] = sc if prev is None else max(prev, sc)
                else:
                    agg[cid] = agg.get(cid, 0.0) + sc

        merged = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return merged

    @staticmethod
    def rrf_fuse(rank_lists: List[List[str]], top_k: int, rrf_k: int = 60) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for lst in rank_lists:
            for rank, doc_id in enumerate(lst, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return fused

    @staticmethod
    def _limit_chunks_per_pmid(results: List[Dict], max_per_pmid: int) -> List[Dict]:
        """Keep at most max_per_pmid chunks per PMID (results assumed best->worst)."""
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
        sources: dict,
    ) -> Dict:
        """Single choke-point to standardize evidence schema for guardrails + downstream."""
        return {
            "chunk_id": str(chunk_id),
            "pmid": str(pmid) if pmid is not None else None,
            "year": int(year) if year is not None else None,
            "title": str(title or ""),
            "journal": str(journal or ""),
            "text": str(text or ""),
            "score": float(score),
            "hits": {
                "bm25": bool(sources.get("bm25")),
                "faiss": bool(sources.get("faiss")),
            },
        }

    def _materialize_chunk(self, chunk_id: str) -> Optional[Dict]:
        """
        Fetch metadata/text for chunk_id from id_map_df.

        If your id_map.parquet does NOT contain pmid/year/title/journal/text,
        replace this method to pull from your chunk store (sqlite/parquet).
        """
        loc = self.chunk_to_row.get(chunk_id)
        if loc is None:
            return None
        y, idx = loc
        row = self.id_map_df[y].iloc[int(idx)]

        return {
            "chunk_id": str(row.get("chunk_id", chunk_id)),
            "pmid": str(row.get("pmid")) if "pmid" in row and not pd.isna(row["pmid"]) else None,
            "year": int(row["year"]) if "year" in row and not pd.isna(row["year"]) else None,
            "title": str(row["title"]) if "title" in row and not pd.isna(row["title"]) else "",
            "journal": str(row["journal"]) if "journal" in row and not pd.isna(row["journal"]) else "",
            "text": str(row["text"]) if "text" in row and not pd.isna(row["text"]) else "",
        }

    @staticmethod
    def evidence_gate(
        results: List[Dict],
        debug: Dict,
        min_chunks: int = 4,
        min_pmids: int = 2,
        require_overlap: bool = True,
        min_top_score: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Deterministic gate (optional helper). Use your external guardrails gate if you have one.

        Returns (ok, reason):
          ok=False => abstain with reason
        """
        if len(results) < min_chunks:
            return False, f"insufficient_chunks({len(results)}<{min_chunks})"

        distinct_pmids = int(debug.get("distinct_pmids", 0))
        if distinct_pmids < min_pmids:
            return False, f"insufficient_distinct_pmids({distinct_pmids}<{min_pmids})"

        both = int(debug.get("both_retrievers_count", 0))
        if require_overlap and both == 0:
            return False, "no_bm25_faiss_overlap"

        top_score = float(results[0].get("score", 0.0)) if results else 0.0
        if min_top_score and top_score < min_top_score:
            return False, f"weak_top_score({top_score:.6f}<{min_top_score})"

        return True, "pass"

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        bm25_k_per_year: int = 40,
        faiss_k_per_year: int = 40,
        bm25_k_global: Optional[int] = None,
        faiss_k_global: Optional[int] = None,
        rrf_k: int = 60,
        merge_mode: str = "max",
        max_chunks_per_pmid: int = 0,  # 0 = no cap; 1 = one chunk per PMID; 2 = max 2 chunks per PMID
        oversample_factor: int = 3,     # ensures you still return top_k after PMID dedupe
        return_debug: bool = False,
    ) -> Any:
        if not query or not query.strip():
            return [] if not return_debug else ([], {"reason": "empty_query"})

        if bm25_k_global is None:
            bm25_k_global = bm25_k_per_year * len(self.years)
        if faiss_k_global is None:
            faiss_k_global = faiss_k_per_year * len(self.years)

        # Encode ONCE
        qvec = self.model.encode(query, normalize_embeddings=True)
        qvec = np.asarray(qvec, dtype=np.float32).reshape(-1)

        # Gather per-year scored candidates
        bm25_scored_all: List[List[Tuple[str, float]]] = []
        faiss_scored_all: List[List[Tuple[str, float]]] = []

        for y in self.years:
            bm25_scored_all.append(self._bm25_search_year_scored(y, query, bm25_k_per_year))
            faiss_scored_all.append(self._faiss_search_year_scored(y, qvec, faiss_k_per_year))

        # Build global ranked lists by score (fixes cross-year ranking)
        bm25_global = self._merge_top_by_score(bm25_scored_all, top_k=bm25_k_global, mode=merge_mode)
        faiss_global = self._merge_top_by_score(faiss_scored_all, top_k=faiss_k_global, mode=merge_mode)

        bm25_rank = dedupe_preserve_order([cid for cid, _ in bm25_global])
        faiss_rank = dedupe_preserve_order([cid for cid, _ in faiss_global])

        # Oversample before RRF so we can dedupe later and still return top_k
        oversample = max(top_k * max(1, oversample_factor), top_k)

        fused = self.rrf_fuse([bm25_rank, faiss_rank], top_k=oversample, rrf_k=rrf_k)
        fused_ids = dedupe_preserve_order([cid for cid, _ in fused])

        bm25_set = set(bm25_rank)
        faiss_set = set(faiss_rank)

        # Materialize + normalize evidence (ONLY place schema gets decided)
        results: List[Dict] = []
        fused_score_map = {cid: float(sc) for cid, sc in fused}

        for cid in fused_ids:
            payload = self._materialize_chunk(cid)
            if payload is None:
                continue

            score = fused_score_map.get(cid, 0.0)
            sources = {"bm25": cid in bm25_set, "faiss": cid in faiss_set}

            normalized = self._normalize_chunk(
                chunk_id=payload.get("chunk_id", cid),
                pmid=payload.get("pmid"),
                year=payload.get("year"),
                title=payload.get("title", ""),
                journal=payload.get("journal", ""),
                text=payload.get("text", ""),
                score=score,
                sources=sources,
            )
            results.append(normalized)

        # Optional diversity cap (PMID-level)
        results = self._limit_chunks_per_pmid(results, max_chunks_per_pmid)

        # Enforce exact top_k after dedupe
        results = results[:top_k]

        if not return_debug:
            return results

        # Debug signals for gating/monitoring
        distinct_pmids = len({r["pmid"] for r in results if r.get("pmid")})
        both_count = sum(1 for r in results if r["hits"]["bm25"] and r["hits"]["faiss"])

        debug = {
            "indexes_root_resolved": str(self.root),
            "years": list(self.years),
            "top_k": top_k,
            "bm25_k_per_year": bm25_k_per_year,
            "faiss_k_per_year": faiss_k_per_year,
            "bm25_k_global": bm25_k_global,
            "faiss_k_global": faiss_k_global,
            "distinct_pmids": distinct_pmids,
            "both_retrievers_count": both_count,
            "bm25_candidates": len(bm25_rank),
            "faiss_candidates": len(faiss_rank),
        }
        return results, debug


if __name__ == "__main__":
    # Smoke test: run from anywhere; path resolution is relative to this file.
    r = HybridRetriever(indexes_root="../indexes", years=["2023", "2024"])

    results, debug = r.retrieve(
        "IL-6 inhibitors rheumatoid arthritis trial results",
        top_k=8,
        max_chunks_per_pmid=1,
        oversample_factor=3,
        return_debug=True,
    )

    print(len(results))
    for i, e in enumerate(results, 1):
        print(i, e["chunk_id"], e["pmid"], e["year"], e["hits"], e["score"])
        print(e["text"][:200].replace("\n", " "), "\n")

    print("DEBUG:", debug)

    ok, reason = HybridRetriever.evidence_gate(
        results,
        debug,
        min_chunks=4,
        min_pmids=2,
        require_overlap=True,
        min_top_score=0.0,  # keep 0 until calibrated
    )
    print("EVIDENCE_GATE:", ok, reason)

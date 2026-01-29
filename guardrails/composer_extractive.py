from __future__ import annotations

from typing import Any, Dict, List


def compose_extractive(query: str, retrieved: List[Dict[str, Any]], *, top_k: int = 8) -> Dict[str, Any]:
    snippets = []
    for c in retrieved[:top_k]:
        snippets.append({
            "pmid": c.get("pmid"),
            "year": c.get("year"),
            "chunk_id": c.get("chunk_id"),
            "title": c.get("title", ""),
            "journal": c.get("journal", ""),
            "score": c.get("score"),
            "hits": c.get("hits", {"bm25": False, "faiss": False}),
            "text": c.get("text", ""),
            "source": "pubmed",
        })

    return {
        "mode": "extractive",
        "abstained": False,
        "refused": False,
        "reason": None,
        "answer": "Relevant evidence was found in the following PubMed abstracts/snippets (2023–2024):",
        "snippets": snippets,
    }

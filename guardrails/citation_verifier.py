from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CitationDecision:
    ok: bool
    reason: str
    signals: Dict[str, Any]


def verify_citations_extract_only(snippets: List[Dict[str, Any]]) -> CitationDecision:
    """
    For extractive mode, citations are implicit: each snippet already carries PMID/chunk_id.
    This verifier is here so you can plug in hard citation enforcement later for synthesis mode.
    """
    if not snippets:
        return CitationDecision(False, "no_snippets", {"snippets": 0})

    missing = 0
    for s in snippets:
        if not s.get("pmid") or not s.get("chunk_id"):
            missing += 1

    if missing > 0:
        return CitationDecision(False, "missing_pmid_or_chunk_id", {"missing": missing, "snippets": len(snippets)})

    return CitationDecision(True, "ok", {"snippets": len(snippets)})

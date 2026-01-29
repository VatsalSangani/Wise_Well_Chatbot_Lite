from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import re

from guardrails.config_loader import load_guardrails_config

CFG = load_guardrails_config()

STOPWORDS = set(CFG.get("stopwords", []))
BROAD_PATTERNS = CFG.get("broad_question_patterns", [])
EVIDENCE_ANCHORS = CFG.get("evidence_anchors", [])


@dataclass
class GateDecision:
    decision: str  # "pass" | "abstain"
    reason: str
    signals: Dict[str, Any]


def _query_tokens(q: str) -> set[str]:
    toks = [re.sub(r"[^a-z0-9]+", "", t.lower()) for t in (q or "").split()]
    toks = [t for t in toks if len(t) >= 3 and t not in STOPWORDS]
    return set(toks)


def _keyword_overlap_ratio(tokens: set[str], text: str) -> float:
    t = (text or "").lower()
    hits = sum(1 for tok in tokens if tok and tok in t)
    return hits / max(1, len(tokens))


def _is_broad_question(q: str) -> bool:
    qq = (q or "").lower()
    return any(re.search(p, qq) for p in BROAD_PATTERNS)


def _has_evidence_anchor(text: str) -> bool:
    t = (text or "").lower()
    return any(anchor.lower() in t for anchor in EVIDENCE_ANCHORS)


def evidence_gate(
    query: str,
    retrieved: List[Dict[str, Any]],
    *,
    min_distinct_pmids: int = 1,
    min_kw_overlap: float = 0.15,
    require_hybrid_hit: bool = True,
) -> GateDecision:
    if not retrieved:
        return GateDecision("abstain", "no_candidates", {"distinct_pmids": 0})

    pmids = [c.get("pmid") for c in retrieved if c.get("pmid")]
    distinct_pmids = len(set(map(str, pmids)))

    both_count = 0
    for c in retrieved:
        hits = c.get("hits") or {}
        if hits.get("bm25") and hits.get("faiss"):
            both_count += 1

    qtoks = _query_tokens(query)
    top_text = retrieved[0].get("text", "")
    top_kw = _keyword_overlap_ratio(qtoks, top_text)

    broad = _is_broad_question(query)
    top3 = retrieved[:3]
    anchor_hits = sum(1 for c in top3 if _has_evidence_anchor(c.get("text", "")))

    signals = {
        "distinct_pmids": distinct_pmids,
        "both_retrievers_count": both_count,
        "top_kw_overlap": top_kw,
        "qtoks_count": len(qtoks),
        "top_score": float(retrieved[0].get("score", 0.0)),
        "broad_question": broad,
        "anchor_hits_top3": anchor_hits,
    }

    # Base checks
    if distinct_pmids < min_distinct_pmids:
        return GateDecision("abstain", "insufficient_pmid_diversity", signals)

    if require_hybrid_hit and both_count == 0:
        return GateDecision("abstain", "no_hybrid_agreement", signals)

    if top_kw < min_kw_overlap:
        return GateDecision("abstain", "low_query_evidence_overlap", signals)

    # Strict broad question checks
    if broad:
        if distinct_pmids < 5:
            return GateDecision("abstain", "broad_question_insufficient_pmids", signals)
        if top_kw < 0.40:
            return GateDecision("abstain", "broad_question_low_overlap", signals)
        if anchor_hits == 0:
            return GateDecision("abstain", "broad_question_no_high_quality_anchor", signals)

    return GateDecision("pass", "sufficient_evidence", signals)

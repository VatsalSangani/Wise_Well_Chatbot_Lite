from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QAState:
    # Input
    query: str

    # Shared runtime config (copied from qa_check defaults)
    top_k: int = 8
    retrieve_pool: int = 24
    max_chunks_per_pmid: int = 1
    oversample_factor: int = 3
    gate_min_distinct_pmids: int = 1
    gate_min_kw_overlap: float = 0.15
    # Dense-only retrieval (Pinecone): there is no second retriever to "agree"
    # with, so the bm25+faiss overlap check is disabled. Safety guardrails
    # (safety_intent, citation_verify) are unaffected.
    gate_require_hybrid_hit: bool = False

    # Stage outputs
    input_ok: Optional[bool] = None
    input_err: Optional[str] = None

    intent: Optional[str] = None  # ok|refuse
    safety_debug: Dict[str, Any] = field(default_factory=dict)

    specificity_ok: Optional[bool] = None
    specificity_reason: Optional[str] = None
    specificity_signals: Dict[str, Any] = field(default_factory=dict)
    clarification: Optional[str] = None

    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_debug: Dict[str, Any] = field(default_factory=dict)

    topic_ok: Optional[bool] = None
    topic_reason: Optional[str] = None
    topic_signals: Dict[str, Any] = field(default_factory=dict)
    filtered: List[Dict[str, Any]] = field(default_factory=list)

    evidence_ok: Optional[bool] = None
    evidence_reason: Optional[str] = None
    evidence_signals: Dict[str, Any] = field(default_factory=dict)

    composed_resp: Dict[str, Any] = field(default_factory=dict)

    citation_ok: Optional[bool] = None
    citation_reason: Optional[str] = None
    citation_signals: Dict[str, Any] = field(default_factory=dict)

    # Final decision
    decision: Optional[str] = None  # ANSWER|ABSTAIN|REFUSE|ERROR
    reason: Optional[str] = None

    # Timings
    timings_ms: Dict[str, float] = field(default_factory=dict)

    # Error
    error: Optional[str] = None
    trace: Optional[str] = None

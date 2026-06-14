from __future__ import annotations
from typing import Any, Dict
import uuid

import sys
from pathlib import Path
_scripts_path = Path(__file__).parent.parent / "scripts"
if str(_scripts_path) not in sys.path:
    sys.path.insert(0, str(_scripts_path))

from qa_check import answer_query  # type: ignore


def run_wisewell_query(
    query: str,
    *,
    retriever: Any,
    top_k: int = 8,
    retrieve_pool: int = 24,
    debug: bool = False,
) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())

    resp = answer_query(
        query=query,
        retriever=retriever,
        top_k=top_k,
        retrieve_pool=retrieve_pool,
        # Dense-only retrieval (Pinecone): no second retriever to agree with, so
        # disable the bm25+faiss overlap check in evidence_gate. Safety guardrails
        # (safety_intent, citation_verify) are unaffected. qa_check's default
        # stays True for the rollback CLI which still runs hybrid retrieval.
        gate_require_hybrid_hit=False,
        profile=debug,  # only profile when debugging
    )

    # Map router/answer mode -> external decision.
    mode = resp.get("mode", "general")
    _MODE_DECISION = {
        "escalate": "ESCALATE",
        "chitchat": "CHITCHAT",
        "refuse": "REFUSE",
        "clarify": "CLARIFY",
        "abstain": "ABSTAIN",
        "rag": "ANSWER",
        "general": "ANSWER",
    }
    decision = _MODE_DECISION.get(mode, "ANSWER")

    out: Dict[str, Any] = {
        "trace_id": trace_id,
        "decision": decision,
        "mode": mode,
        "reason": resp.get("reason"),
        "answer": resp.get("answer"),               # final for terminal modes; None for rag/general
        "snippets": resp.get("snippets", []) or [],
        "code": resp.get("code", {}) or {},          # code-inserted pieces (defer/disclaimer/source/offer)
        "is_personal": bool(resp.get("is_personal", False)),
    }

    if debug:
        out["timings_ms"] = resp.get("stage_timings_ms", {}) or {}
        out["fork_signals"] = resp.get("fork_signals") or {}
        out["signals"] = {
            "safety": resp.get("safety_intent_debug") or {},
            "specificity": resp.get("specificity_signals") or {},
            "topic": resp.get("topic_signals") or {},
            "evidence": resp.get("signals") or {},
            "citation": resp.get("citation_signals") or {},
            "fork": resp.get("fork_signals") or {},
        }

    return out

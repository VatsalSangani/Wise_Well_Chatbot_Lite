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
        profile=debug,  # only profile when debugging
    )

    if resp.get("refused"):
        decision = "REFUSE"
    elif resp.get("abstained"):
        decision = "ABSTAIN"
    else:
        decision = "ANSWER"

    out: Dict[str, Any] = {
        "trace_id": trace_id,
        "decision": decision,
        "reason": resp.get("reason"),
        "answer": resp.get("answer"),
        "snippets": resp.get("snippets", []) or [],
    }

    if debug:
        out["timings_ms"] = resp.get("stage_timings_ms", {}) or {}
        out["signals"] = {
            "safety": resp.get("safety_intent_debug") or {},
            "specificity": resp.get("specificity_signals") or {},
            "topic": resp.get("topic_signals") or {},
            "evidence": resp.get("signals") or {},
            "citation": resp.get("citation_signals") or {},
        }

    return out

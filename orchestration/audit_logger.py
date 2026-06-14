from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .state import QAState


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_audit_jsonl(state: QAState, out_path: str = "audit_logs/wise_well_audit.jsonl",
                     extra: Optional[Dict[str, Any]] = None) -> str:
    """Append one audit record to a local JSONL file.

    CLI-ONLY: this is invoked from orchestration/run.py, NOT the live /query API
    path. The local-file append is fine for a single-process CLI run, but it is
    deliberately kept off the request path — under horizontal scaling each
    instance would write to its own local disk, fragmenting the audit trail. If
    audit logging is ever needed on the live API, write to stdout (for a log
    collector) or an external sink (object store / log service), not local disk.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "query": state.query,
        "decision": state.decision,
        "reason": state.reason,
        "timings_ms": state.timings_ms,
        "signals": {
            "safety": state.safety_debug,
            "specificity": state.specificity_signals,
            "topic": state.topic_signals,
            "evidence": state.evidence_signals,
            "citation": state.citation_signals,
        },
        "retrieval": {
            "retrieved_n": len(state.retrieved),
            "filtered_n": len(state.filtered),
        },
        "output": {
            "abstained": (state.composed_resp or {}).get("abstained"),
            "refused": (state.composed_resp or {}).get("refused"),
            "answer_preview": (state.composed_resp or {}).get("answer", "")[:400] if state.composed_resp else None,
            "snippets_n": len((state.composed_resp or {}).get("snippets", []) or []),
        },
        "error": state.error,
    }

    if extra:
        payload["extra"] = extra

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return str(p)

# ==================================
# scripts/qa_check.py (with profiling support + overlap veto)
# ==================================

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import re
from typing import Any, Dict, List, Tuple

from retrieval.hybrid_retriever import HybridRetriever
from latency_profile import LatencyTimer, PipelineProfiler

from guardrails.validate_config import validate as validate_guardrails_config
from guardrails.input_validation import validate_query
from guardrails.safety_intent import classify_intent
from guardrails.query_specificity import assess_query_specificity
from guardrails.topic_consistency import apply_topic_consistency
from guardrails.evidence_gate import evidence_gate
from guardrails.composer_extractive import compose_extractive
from guardrails.citation_verifier import verify_citations_extract_only
from guardrails.mechanism_gate import mechanism_gate



# -----------------------------
# Off-topic overlap veto helpers
# -----------------------------

_STOP = {
    "what", "is", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with",
    "how", "does", "do", "are", "was", "were", "be", "by", "as", "at", "from", "into",
    "about", "role", "mechanism", "treatment", "levels", "risk", "assessment", "associated",
    "increase", "increased", "mortality", "clinical", "significance", "contribute"
}

def _extract_anchors(query: str, *, max_anchors: int = 8) -> List[str]:
    """
    Cheap heuristic anchor extraction:
    - lowercased alnum tokens (keeps hyphenated)
    - drops short tokens, stopwords, pure numbers
    """
    q = query.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", q)

    anchors: List[str] = []
    for t in tokens:
        if t in _STOP:
            continue
        if t.isdigit():
            continue
        if len(t) < 4:
            continue
        anchors.append(t)

    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for a in anchors:
        if a not in seen:
            out.append(a)
            seen.add(a)

    return out[:max_anchors]


def _evidence_blob(results: List[Dict[str, Any]]) -> str:
    """
    Build a lowercased evidence blob from retrieval results.
    Works even if keys differ across your retriever outputs.
    """
    parts: List[str] = []
    for r in results or []:
        # Common fields across systems
        for k in ("title", "journal", "pmid", "chunk_id", "text", "snippet", "abstract"):
            v = r.get(k)
            if v:
                parts.append(str(v))
    return " ".join(parts).lower()


def _overlap_veto(
    query: str,
    results: List[Dict[str, Any]],
    *,
    min_hits: int = 1,
    min_coverage: float = 0.25,
    min_anchor_count: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (ok, signals). If ok=False => off-topic retrieval, should ABSTAIN.

    Logic:
    - Extract query anchors (informative tokens)
    - Count how many anchors appear in evidence text
    - If too few hits/coverage -> veto
    """
    anchors = _extract_anchors(query)
    blob = _evidence_blob(results)

    hits = [a for a in anchors if a in blob]
    hit_count = len(hits)
    coverage = hit_count / max(1, len(anchors))

    signals = {
        "anchors": anchors,
        "hits": hits,
        "hit_count": hit_count,
        "coverage": coverage,
        "min_hits": min_hits,
        "min_coverage": min_coverage,
        "min_anchor_count": min_anchor_count,
    }

    # Don’t veto tiny anchor sets; they’re too noisy.
    if len(anchors) < min_anchor_count:
        return True, {**signals, "veto_applied": False, "reason": "too_few_anchors"}

    # Strict: wrong ANSWER is worse than ABSTAIN.
    if hit_count < min_hits or coverage < min_coverage:
        return False, {**signals, "veto_applied": True, "reason": "off_topic_low_overlap"}

    return True, {**signals, "veto_applied": True, "reason": "ok"}


def answer_query(
    query: str,
    retriever: HybridRetriever,
    *,
    top_k: int = 8,
    retrieve_pool: int = 24,
    max_chunks_per_pmid: int = 1,
    oversample_factor: int = 3,
    gate_min_distinct_pmids: int = 1,
    gate_min_kw_overlap: float = 0.15,
    gate_require_hybrid_hit: bool = True,
    profile: bool = False,
    # Overlap veto knobs (keep defaults unless you have evidence to tune)
    veto_min_hits: int = 1,
    veto_min_coverage: float = 0.25,
    veto_min_anchor_count: int = 3,
) -> Dict[str, Any]:
    """
    Answer a query using the WiseWell guardrails pipeline.

    Args:
        profile: If True, include stage_timings_ms in response.

    Returns:
        Dict with abstained, refused, reason, answer, snippets, etc.
        If profile=True, includes stage_timings_ms: {stage_name: float_ms, ...}
    """
    profiler = PipelineProfiler() if profile else None
    if profiler:
        profiler.start_total()

    def timed_execute(stage_name: str, func):
        if profiler:
            with LatencyTimer(stage_name) as timer:
                result = func()
            profiler.record_stage(stage_name, timer.elapsed_ms)
            return result
        return func()

    # 0) Config validation (fix: unique stage name)
    timed_execute("config_validate", lambda: validate_guardrails_config())

    # 1) Input validation (fix: unique stage name)
    def validate_input():
        return validate_query(query)

    ok, err = timed_execute("input_validate", validate_input)
    if not ok:
        resp = {
            "mode": "extractive",
            "abstained": True,
            "refused": False,
            "reason": err,
            "answer": None,
            "snippets": [],
        }
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp

    # 2) Safety intent
    def classify():
        return classify_intent(query)

    intent, safety_debug = timed_execute("safety_intent", classify)

    if intent == "refuse":
        resp = {
            "mode": "extractive",
            "abstained": False,
            "refused": True,
            "reason": safety_debug.get("reason"),
            "safety_intent_debug": safety_debug,
            "answer": None,
            "snippets": [],
        }
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp

    # 3) Specificity
    def assess_spec():
        return assess_query_specificity(query)

    sd = timed_execute("specificity", assess_spec)

    if not sd.specific:
        resp = {
            "mode": "extractive",
            "abstained": True,
            "refused": False,
            "reason": sd.reason,
            "specificity_signals": sd.signals,
            "clarification": sd.clarification,
            "answer": None,
            "snippets": [],
        }
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp

    # 4) Retrieval
    pool_k = max(retrieve_pool, top_k * 3)

    def retrieve():
        return retriever.retrieve(
            query,
            top_k=pool_k,
            max_chunks_per_pmid=max_chunks_per_pmid,
            oversample_factor=oversample_factor,
            return_debug=True,
        )

    results, debug = timed_execute("retrieve", retrieve)

    # 5) Topic consistency
    def topic_consistency():
        return apply_topic_consistency(
            query,
            results,
            top_k=top_k,
            pool_k=pool_k,
            min_avg_cov=0.18,
            min_good_chunks=4,
            good_chunk_threshold=0.20,
            topic_cov_threshold=0.34,
        )

    td = timed_execute("topic_consistency", topic_consistency)

    if not td.ok:
        resp = {
            "mode": "extractive",
            "abstained": True,
            "refused": False,
            "reason": td.reason,
            "topic_signals": td.signals,
            "debug": debug,
            "answer": None,
            "snippets": [],
        }
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp

    # Keep only top_k after topic consistency
    results = td.filtered[:top_k]

    # 5.5) Off-topic overlap veto (NEW)
    def overlap_check():
        return _overlap_veto(
            query,
            results,
            min_hits=veto_min_hits,
            min_coverage=veto_min_coverage,
            min_anchor_count=veto_min_anchor_count,
        )

    ok_overlap, ov_signals = timed_execute("overlap_veto", overlap_check)
    if not ok_overlap:
        resp = {
            "mode": "extractive",
            "abstained": True,
            "refused": False,
            "reason": "off_topic_retrieval",
            "overlap_signals": ov_signals,
            "topic_signals": td.signals,
            "debug": debug,
            "answer": None,
            "snippets": [],
        }
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp
    
    # 5.6) Mechanism-aware gate 
    def mech_gate():
        return mechanism_gate(query, results, min_mech_snippets=1)

    ok_mech, mech_signals = timed_execute("mechanism_gate", mech_gate)

    if not ok_mech:
        resp = {
            "mode": "extractive",
            "abstained": True,
            "refused": False,
            "reason": "insufficient_mechanistic_evidence",
            "mechanism_signals": mech_signals,
            "topic_signals": td.signals,
            "debug": debug,
            "answer": None,
            "snippets": [],
        }
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp


    # 6) Evidence gate
    def gate():
        return evidence_gate(
            query,
            results,
            min_distinct_pmids=gate_min_distinct_pmids,
            min_kw_overlap=gate_min_kw_overlap,
            require_hybrid_hit=gate_require_hybrid_hit,
        )

    gd = timed_execute("evidence_gate", gate)

    if gd.decision != "pass":
        resp = {
            "mode": "extractive",
            "abstained": True,
            "refused": False,
            "reason": gd.reason,
            "signals": gd.signals,
            "topic_signals": td.signals,
            "overlap_signals": ov_signals,
            "debug": debug,
            "answer": None,
            "snippets": [],
        }
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp

    # 7) Compose
    def compose():
        return compose_extractive(query, results, top_k=top_k)

    resp = timed_execute("compose", compose)

    # Attach signals/debug
    resp["signals"] = gd.signals
    resp["topic_signals"] = td.signals
    resp["overlap_signals"] = ov_signals
    resp["safety_intent_debug"] = safety_debug
    resp["specificity_signals"] = sd.signals
    resp["debug"] = debug

    # 8) Citation verify
    def verify():
        return verify_citations_extract_only(resp.get("snippets", []))

    cd = timed_execute("citation_verify", verify)
    resp["citation_signals"] = cd.signals

    if not cd.ok:
        resp["abstained"] = True
        resp["refused"] = False
        resp["reason"] = cd.reason
        resp["answer"] = None
        resp["snippets"] = []

    if profiler:
        profiler.end_total()
        resp["stage_timings_ms"] = profiler.get_timings()

    return resp


if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Use same logic as backend
    repo_root = Path(__file__).resolve().parent.parent
    indexes_path = repo_root / "kb" / "indexes"
    
    retriever = HybridRetriever(indexes_root=str(indexes_path), years=["2023", "2024"])

    tests = [
        "Why is this number high?",
        "What do IL-6 inhibitors do in rheumatoid arthritis?",
        "Is Vitamin D good for immunity?",
        "I have rheumatoid arthritis, should I take an IL-6 inhibitor?",
        "CRP 18 mg/L — what does that indicate?",
        "What is the mechanism of action of metformin in the treatment of type 2 diabetes?",
    ]

    for q in tests:
        out = answer_query(
            q,
            retriever,
            top_k=8,
            retrieve_pool=24,
            max_chunks_per_pmid=1,
            oversample_factor=3,
            profile=True,
        )

        print("\nQUERY:", q)
        print("ABSTAINED:", out.get("abstained"), "REFUSED:", out.get("refused"), "REASON:", out.get("reason"))

        if out.get("specificity_signals"):
            print("SPECIFICITY_SIGNALS:", out["specificity_signals"])
        if out.get("clarification"):
            print("CLARIFICATION:", out["clarification"])

        if out.get("signals"):
            print("SIGNALS:", out["signals"])
        if out.get("topic_signals"):
            print("TOPIC_SIGNALS:", out["topic_signals"])
        if out.get("overlap_signals"):
            print("OVERLAP_SIGNALS:", out["overlap_signals"])
        if out.get("citation_signals"):
            print("CITATION_SIGNALS:", out["citation_signals"])
        if out.get("debug"):
            print("DEBUG:", out["debug"])

        if not out.get("abstained") and not out.get("refused"):
            print("\nANSWER:", out.get("answer"), "\n")
            for i, s in enumerate(out.get("snippets", []), 1):
                print(i, s.get("chunk_id"), s.get("pmid"), s.get("year"), s.get("hits"), s.get("score"))
                print((s.get("text", "")[:200] or "").replace("\n", " "), "\n")

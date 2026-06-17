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

# NOTE: HybridRetriever (BM25+FAISS) is the rollback retriever. It is no longer
# imported at module top — that pulled `import faiss` into FastAPI cold start.
# answer_query() takes the retriever as a parameter (now PineconeRetriever in
# production); the rollback CLI in __main__ imports HybridRetriever locally.
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
from guardrails.router import classify_lane
from guardrails.red_flags import detect_red_flag
from guardrails import responses

from config import RAG_MIN_TOP_SCORE, RAG_MIN_DISTINCT_PMIDS



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
    retriever: Any,  # PineconeRetriever in production; HybridRetriever for rollback
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

    def _finish(resp: Dict[str, Any]) -> Dict[str, Any]:
        if profiler:
            profiler.end_total()
            resp["stage_timings_ms"] = profiler.get_timings()
        return resp

    def _base(mode: str, *, answer=None, reason=None, snippets=None,
              is_personal: bool = False, **extra) -> Dict[str, Any]:
        r: Dict[str, Any] = {
            "mode": mode,                       # escalate|chitchat|refuse|clarify|abstain|rag|general
            "answer": answer,                   # final for terminal modes; None for rag/general (LLM fills)
            "snippets": snippets or [],
            "is_personal": is_personal,
            "reason": reason,
            "refused": mode == "refuse",
            "abstained": mode == "abstain",
            # Code-inserted pieces, authored once in guardrails/responses.py.
            "code": {"source_block": "", "disclaimer": "", "soft_defer": "", "offer": ""},
        }
        r.update(extra)
        return r

    # -1) RED FLAG — highest priority, overrides the whole flow. An active
    #     emergency does not wait for retrieval. Router-to-help, never a treater.
    rf = timed_execute("red_flag", lambda: detect_red_flag(query))
    if rf.escalate:
        msg = responses.TIER2_SUICIDE if rf.is_suicide else responses.tier2_physical(
            rf.signals.get("description", "")
        )
        return _finish(_base("escalate", answer=msg, reason=f"red_flag_{rf.category}",
                             red_flag_signals=rf.signals))

    # 0) ROUTER — chitchat lane skips input_validation + specificity + retrieval.
    rd = timed_execute("router", lambda: classify_lane(query))
    if rd.lane == "chitchat":
        return _finish(_base("chitchat", answer=responses.chitchat_response(query),
                             reason="chitchat", router_signals=rd.signals))
    is_personal = rd.is_personal

    # 0.5) Config validation
    timed_execute("config_validate", lambda: validate_guardrails_config())

    # 1) Input validation (medical lane only). too_short -> clarify (don't reject);
    #    injection/too_long -> abstain.
    ok, err = timed_execute("input_validate", lambda: validate_query(query))
    if not ok:
        if err == "too_short":
            return _finish(_base("clarify", answer=responses.CLARIFY_NO_ANCHOR, reason="too_short"))
        return _finish(_base("abstain", reason=err))

    # 2) Safety intent (narrowed hard-refuse: individual Dx/Rx/dosing only).
    intent, safety_debug = timed_execute("safety_intent", lambda: classify_intent(query))
    if intent == "refuse":
        return _finish(_base("refuse", answer=responses.hard_refuse(),
                             reason=safety_debug.get("reason"), safety_intent_debug=safety_debug))

    # 3) Specificity -> clarify instead of refuse. Anchored-but-vague gets the
    #    answer-likely-interpretation (general mode) + offer-to-narrow; totally
    #    vague (no topic anchor) gets ONE clarifying question (topic only).
    sd = timed_execute("specificity", lambda: assess_query_specificity(query))
    clarify_offer = False
    if not sd.specific:
        # Answer-then-offer is the DEFAULT. Only fall back to a clarifying
        # question when the query is genuinely referent-less — underspecified
        # phrasing ("is it high", "what does it mean") OR no substantive content
        # token at all ("is it bad"). Anything with a real topic anchor (a
        # disease, concept, biomarker, or drug — "what is rheumatoid arthritis")
        # is answerable: proceed to retrieval and offer to narrow.
        sig = sd.signals
        has_content = bool(
            sig.get("specific_token_count", 0) >= 1
            or sig.get("biomarker_hit") or sig.get("drug_like_hit")
            or sig.get("unit_hit") or sig.get("disease_phrase_hit")
        )
        if sig.get("underspecified_hit") or not has_content:
            return _finish(_base("clarify", answer=responses.CLARIFY_NO_ANCHOR,
                                 reason="insufficient_context", specificity_signals=sd.signals))
        clarify_offer = True  # answer the likely reading, then offer to narrow

    # 4) Retrieval
    pool_k = max(retrieve_pool, top_k * 3)
    results, debug = timed_execute("retrieve", lambda: retriever.retrieve(
        query, top_k=pool_k, max_chunks_per_pmid=max_chunks_per_pmid,
        oversample_factor=oversample_factor, return_debug=True))

    # 5) Quality gates — COLLECT pass/fail; do NOT abstain. Weak retrieval now
    #    falls to general mode (Position B), not an abstain.
    td = timed_execute("topic_consistency", lambda: apply_topic_consistency(
        query, results, top_k=top_k, pool_k=pool_k, min_avg_cov=0.18,
        min_good_chunks=4, good_chunk_threshold=0.20, topic_cov_threshold=0.34))
    filtered = (td.filtered[:top_k] if td.ok else results[:top_k])

    ok_overlap, ov_signals = timed_execute("overlap_veto", lambda: _overlap_veto(
        query, filtered, min_hits=veto_min_hits, min_coverage=veto_min_coverage,
        min_anchor_count=veto_min_anchor_count))
    ok_mech, mech_signals = timed_execute("mechanism_gate", lambda: mechanism_gate(
        query, filtered, min_mech_snippets=1))
    gd = timed_execute("evidence_gate", lambda: evidence_gate(
        query, filtered, min_distinct_pmids=gate_min_distinct_pmids,
        min_kw_overlap=gate_min_kw_overlap, require_hybrid_hit=gate_require_hybrid_hit))

    # 6) EVIDENCE-CONFIDENCE FORK (Position B). RAG bar set deliberately HIGH;
    #    this absorbs the weak-retrieval filtering the disabled require_hybrid_hit
    #    gate used to do.
    top_score = max((float(r.get("score", 0.0)) for r in filtered), default=0.0)
    distinct_pmids = len({str(r.get("pmid")) for r in filtered if r.get("pmid")})
    # The RAG bar is the evidence-confidence signal (addendum): evidence_gate
    # decision + a HIGH score + enough distinct PMIDs. The other quality gates
    # (topic/overlap/mechanism) are advisory signals here, not hard RAG blockers —
    # their failure no longer abstains (Position B), and gating RAG on all of them
    # pushed too many legitimate answers into general mode.
    gates_pass = td.ok and ok_overlap and ok_mech and (gd.decision == "pass")
    rag_confident = (gd.decision == "pass"
                     and top_score >= RAG_MIN_TOP_SCORE
                     and distinct_pmids >= RAG_MIN_DISTINCT_PMIDS)

    soft_defer = responses.TIER1_SOFT_DEFER if is_personal else ""
    offer = responses.OFFER_TO_NARROW if clarify_offer else ""

    fork_signals = {
        "rag_confident": rag_confident, "top_score": round(top_score, 4),
        "distinct_pmids": distinct_pmids, "rag_min_top_score": RAG_MIN_TOP_SCORE,
        "rag_min_distinct_pmids": RAG_MIN_DISTINCT_PMIDS, "gates_pass": gates_pass,
        "topic_ok": td.ok, "overlap_ok": ok_overlap, "mech_ok": ok_mech,
        "evidence_decision": gd.decision,
    }

    if rag_confident:
        composed = compose_extractive(query, filtered, top_k=top_k)
        snippets = composed.get("snippets", []) or []
        cd = verify_citations_extract_only(snippets)
        if cd.ok:
            resp = _base("rag", reason="rag_confident", snippets=snippets, is_personal=is_personal)
            resp["code"].update({
                "source_block": responses.build_source_block(snippets),
                "soft_defer": soft_defer, "offer": offer,
            })
            resp["citation_signals"] = cd.signals
        else:
            # Citations couldn't be verified -> fall to general rather than emit bad cites.
            resp = _base("general", reason="general_after_citation_fail", is_personal=is_personal)
            resp["code"].update({"disclaimer": responses.GENERAL_MODE_DISCLAIMER,
                                 "soft_defer": soft_defer, "offer": offer})
            resp["citation_signals"] = cd.signals
    else:
        resp = _base("general", reason="weak_retrieval_general_mode", is_personal=is_personal)
        resp["code"].update({"disclaimer": responses.GENERAL_MODE_DISCLAIMER,
                             "soft_defer": soft_defer, "offer": offer})

    resp["safety_intent_debug"] = safety_debug
    resp["specificity_signals"] = sd.signals
    resp["topic_signals"] = td.signals
    resp["overlap_signals"] = ov_signals
    resp["mechanism_signals"] = mech_signals
    resp["signals"] = gd.signals
    resp["fork_signals"] = fork_signals
    resp["debug"] = debug
    return _finish(resp)


if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Rollback CLI: exercise the OLD hybrid retriever directly. Imported locally
    # so the production import path never pulls faiss into cold start.
    from retrieval.hybrid_retriever import HybridRetriever

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

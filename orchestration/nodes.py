from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

from .state import QAState
from .utils import ensure_repo_paths, timed


def node_validate(state: QAState) -> QAState:
    ensure_repo_paths()
    with timed(state.timings_ms, "validate"):
        from guardrails.validate_config import validate as validate_guardrails_config  # type: ignore
        validate_guardrails_config()
    return state


def node_input_validation(state: QAState) -> QAState:
    ensure_repo_paths()
    with timed(state.timings_ms, "input_validation"):
        from guardrails.input_validation import validate_query  # type: ignore
        ok, err = validate_query(state.query)
        state.input_ok = bool(ok)
        state.input_err = err
    return state


def node_safety_intent(state: QAState) -> QAState:
    ensure_repo_paths()
    with timed(state.timings_ms, "safety_intent"):
        from guardrails.safety_intent import classify_intent  # type: ignore
        intent, safety_debug = classify_intent(state.query)
        state.intent = intent
        state.safety_debug = safety_debug or {}
    return state


def node_specificity(state: QAState) -> QAState:
    ensure_repo_paths()
    with timed(state.timings_ms, "specificity"):
        from guardrails.query_specificity import assess_query_specificity  # type: ignore
        sd = assess_query_specificity(state.query)
        state.specificity_ok = bool(getattr(sd, "specific", False))
        state.specificity_reason = getattr(sd, "reason", None)
        state.specificity_signals = getattr(sd, "signals", {}) or {}
        state.clarification = getattr(sd, "clarification", None)
    return state


def node_retrieve(state: QAState, retriever: Any) -> QAState:
    ensure_repo_paths()
    pool_k = max(state.retrieve_pool, state.top_k * 3)

    with timed(state.timings_ms, "retrieve"):
        results, debug = retriever.retrieve(
            state.query,
            top_k=pool_k,
            max_chunks_per_pmid=state.max_chunks_per_pmid,
            oversample_factor=state.oversample_factor,
            return_debug=True,
        )
        state.retrieved = results or []
        state.retrieval_debug = debug or {}
    return state


def node_topic_consistency(state: QAState) -> QAState:
    ensure_repo_paths()
    pool_k = max(state.retrieve_pool, state.top_k * 3)

    with timed(state.timings_ms, "topic_consistency"):
        from guardrails.topic_consistency import apply_topic_consistency  # type: ignore
        td = apply_topic_consistency(
            state.query,
            state.retrieved,
            top_k=state.top_k,
            pool_k=pool_k,
            min_avg_cov=0.18,
            min_good_chunks=4,
            good_chunk_threshold=0.20,
            topic_cov_threshold=0.34,
        )
        state.topic_ok = bool(getattr(td, "ok", False))
        state.topic_reason = getattr(td, "reason", None)
        state.topic_signals = getattr(td, "signals", {}) or {}
        state.filtered = getattr(td, "filtered", []) or []
    return state


def node_evidence_gate(state: QAState) -> QAState:
    ensure_repo_paths()
    with timed(state.timings_ms, "evidence_gate"):
        from guardrails.evidence_gate import evidence_gate  # type: ignore
        results = (state.filtered or [])[: state.top_k]
        gd = evidence_gate(
            state.query,
            results,
            min_distinct_pmids=state.gate_min_distinct_pmids,
            min_kw_overlap=state.gate_min_kw_overlap,
            require_hybrid_hit=state.gate_require_hybrid_hit,
        )
        # Expect gd.decision == "pass"
        decision = getattr(gd, "decision", None)
        state.evidence_ok = (decision == "pass")
        state.evidence_reason = getattr(gd, "reason", None)
        state.evidence_signals = getattr(gd, "signals", {}) or {}
    return state


def node_compose(state: QAState) -> QAState:
    ensure_repo_paths()
    with timed(state.timings_ms, "compose"):
        from guardrails.composer_extractive import compose_extractive  # type: ignore
        results = (state.filtered or [])[: state.top_k]
        resp = compose_extractive(state.query, results, top_k=state.top_k)
        if not isinstance(resp, dict):
            resp = {"abstained": True, "refused": False, "reason": "compose_error", "answer": None, "snippets": []}
        state.composed_resp = resp
        # enrich like qa_check.py does
        state.composed_resp["signals"] = state.evidence_signals
        state.composed_resp["topic_signals"] = state.topic_signals
        state.composed_resp["safety_intent_debug"] = state.safety_debug
        state.composed_resp["specificity_signals"] = state.specificity_signals
        state.composed_resp["debug"] = state.retrieval_debug
    return state


def node_citation_verify(state: QAState) -> QAState:
    ensure_repo_paths()
    with timed(state.timings_ms, "citation_verify"):
        from guardrails.citation_verifier import verify_citations_extract_only  # type: ignore
        snippets = (state.composed_resp or {}).get("snippets", []) or []
        cd = verify_citations_extract_only(snippets)
        state.citation_ok = bool(getattr(cd, "ok", False))
        state.citation_reason = getattr(cd, "reason", None)
        state.citation_signals = getattr(cd, "signals", {}) or {}
        state.composed_resp["citation_signals"] = state.citation_signals

        if not state.citation_ok:
            # mirror qa_check behavior
            state.composed_resp["abstained"] = True
            state.composed_resp["refused"] = False
            state.composed_resp["reason"] = state.citation_reason
            state.composed_resp["answer"] = None
            state.composed_resp["snippets"] = []
    return state


def node_finalize(state: QAState) -> QAState:
    """
    Convert intermediate stage outcomes into final decision + reason.
    """
    with timed(state.timings_ms, "finalize"):
        # ERROR short-circuit
        if state.error:
            state.decision = "ERROR"
            state.reason = state.error
            return state

        # invalid input -> abstain
        if state.input_ok is False:
            state.decision = "ABSTAIN"
            state.reason = state.input_err
            return state

        # refuse
        if state.intent == "refuse":
            state.decision = "REFUSE"
            state.reason = (state.safety_debug or {}).get("reason")
            return state

        # underspecified -> abstain
        if state.specificity_ok is False:
            state.decision = "ABSTAIN"
            state.reason = state.specificity_reason
            return state

        # topic insufficient -> abstain
        if state.topic_ok is False:
            state.decision = "ABSTAIN"
            state.reason = state.topic_reason
            return state

        # evidence fail -> abstain
        if state.evidence_ok is False:
            state.decision = "ABSTAIN"
            state.reason = state.evidence_reason
            return state

        # citation fail -> abstain
        if state.citation_ok is False:
            state.decision = "ABSTAIN"
            state.reason = state.citation_reason
            return state

        # otherwise answer if composed_resp says so
        abstained = bool((state.composed_resp or {}).get("abstained"))
        refused = bool((state.composed_resp or {}).get("refused"))
        if refused:
            state.decision = "REFUSE"
            state.reason = (state.composed_resp or {}).get("reason")
        elif abstained:
            state.decision = "ABSTAIN"
            state.reason = (state.composed_resp or {}).get("reason")
        else:
            state.decision = "ANSWER"
            state.reason = "pass"
    return state


def node_error_wrapper(fn, state: QAState, stage_name: str, *args, **kwargs) -> QAState:
    """
    Wrap a node to capture exceptions into state.error/state.trace.
    """
    try:
        return fn(state, *args, **kwargs)
    except Exception as e:
        state.error = f"{stage_name}: {type(e).__name__}: {e}"
        state.trace = traceback.format_exc()
        return state

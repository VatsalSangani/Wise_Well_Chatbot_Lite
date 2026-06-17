import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks

from backend.deps import get_retriever
from backend.schemas import QueryRequest, QueryResponse
from config import ENABLE_LLM_SYNTHESIS, MAX_QUERY_LENGTH, RETRIEVE_POOL, TOP_K, DEBUG, EVAL_SAMPLE_RATE
from orchestration import observability as obs
from orchestration.service import run_wisewell_query
from orchestration.llm_syntheses import (
    synthesize_response,
    synthesize_general_response,
    OFFTOPIC_MARKER,
)
from orchestration.query_resolver import resolve_query


def _assemble(main: str, parts: list[str]) -> str:
    """Join the LLM-generated body with code-inserted trailers (defer /
    disclaimer / source block / offer), dropping empties."""
    blocks = [main.strip()] if main else []
    blocks += [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(blocks)


# Detect the domain-scope off-topic redirect via the shared OFFTOPIC_MARKER
# constant (defined in llm_syntheses.py, injected into the synthesis prompt).
# Single source of truth — the prompt text and this detector cannot drift.
# When the answer is the redirect, trailers (disclaimer/soft-defer/follow-up) are
# suppressed — they'd contradict "I can only help with health questions."
def _is_offtopic_redirect(structured: dict) -> bool:
    return OFFTOPIC_MARKER in (structured.get("summary") or "")


def _structured_to_markdown(structured: dict) -> str:
    """Convert structured JSON answer to clean markdown."""
    lines = []

    # Summary (lead line)
    if structured.get("summary"):
        lines.append(structured["summary"])

    # Key points (bullet list)
    if structured.get("key_points"):
        lines.append("")
        for point in structured["key_points"]:
            lines.append(f"• {point}")

    # Explanation (prose)
    if structured.get("explanation"):
        lines.append("")
        lines.append(structured["explanation"])

    # When to see doctor (distinct section)
    if structured.get("when_to_see_doctor"):
        lines.append("")
        lines.append("**When to see a doctor:**")
        lines.append(structured["when_to_see_doctor"])

    # Citations (sources)
    if structured.get("citations"):
        lines.append("")
        lines.append("**Sources:**")
        for citation in structured["citations"]:
            pmid = citation.get("pmid", "Unknown")
            claim = citation.get("claim", "Supporting evidence")
            lines.append(f"[PMID: {pmid}] {claim}")

    return "\n".join(lines)

router = APIRouter()
logger = structlog.get_logger()


@router.post("/query", response_model=QueryResponse)
# capture_input/output=False: the function args are FastAPI objects (Request,
# BackgroundTasks) that don't serialize meaningfully (shows as null). We set clean
# trace input/output explicitly via obs.update_trace below.
@obs.observe(name="wisewell_query", capture_input=False, capture_output=False)
async def query_endpoint(request: Request, req: QueryRequest, background_tasks: BackgroundTasks):
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))

    if len(req.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long (max {MAX_QUERY_LENGTH} characters)",
        )
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(
        "query_received",
        trace_id=trace_id,
        query_length=len(req.query),
        debug=req.debug,
    )

    try:
        retriever = get_retriever()
        start = time.time()

        # Step 0: Query resolution (if history provided)
        # Rewrite follow-up + history into standalone query; if no history, pass through.
        query_to_use, was_resolved = resolve_query(req.query, req.history)
        if was_resolved:
            logger.info("query_resolved_from_history", original=req.query, resolved=query_to_use)

        # Retrieval span: capture query + retrieved contexts with scores. Wraps the
        # pipeline call without touching retrieval logic; contexts come from the
        # returned snippets. Tracing only — never alters the result.
        with obs.span("retrieval", input=query_to_use) as rspan:
            result = run_wisewell_query(
                query_to_use,
                retriever=retriever,
                debug=req.debug,
                top_k=TOP_K,
                retrieve_pool=RETRIEVE_POOL,
            )
            rspan.update(output=[
                {"pmid": s.get("pmid"), "score": s.get("score"), "title": s.get("title")}
                for s in (result.get("snippets") or [])
            ])
        result["trace_id"] = trace_id

        # Answer assembly by mode. Terminal modes (ESCALATE/CHITCHAT/REFUSE/
        # CLARIFY/ABSTAIN) already carry their final code-inserted answer — leave
        # them untouched. RAG/general run the LLM and append code-inserted pieces.
        mode = result.get("mode")
        code = result.get("code", {}) or {}
        result["llm_synthesized"] = False

        is_personal = bool(result.get("is_personal"))

        if result.get("decision") == "ANSWER":
            try:
                if mode == "rag" and result.get("snippets"):
                    syn = synthesize_response(
                        query=query_to_use,  # resolved standalone query, NOT raw follow-up
                        evidence_snippets=result["snippets"],
                        enable_llm=ENABLE_LLM_SYNTHESIS,
                    )

                    # Both structured and extractive-fallback paths omit the
                    # code source_block — sources render from the card block
                    # (snippets) on the frontend, so a text list would duplicate them.
                    follow_up = ""
                    offtopic = False
                    if syn.get("structured"):
                        sa = syn.get("synthesized_answer", {})
                        body = _structured_to_markdown(sa)
                        follow_up = sa.get("follow_up_suggestion") or ""
                        offtopic = _is_offtopic_redirect(sa)
                    else:
                        body = syn.get("synthesized_answer") or ""

                    # RAG trailers: soft-defer (if personal) then LLM follow-up
                    # (replaces the generic OFFER_TO_NARROW). Off-topic redirect
                    # gets no trailers.
                    if offtopic:
                        result["answer"] = body
                    else:
                        result["answer"] = _assemble(body, [
                            code.get("soft_defer", "") if is_personal else "",
                            follow_up,
                        ])
                    result["llm_synthesized"] = bool(syn.get("success"))

                elif mode == "general":
                    body = ""
                    follow_up = ""
                    offtopic = False
                    if ENABLE_LLM_SYNTHESIS:
                        gen = synthesize_general_response(query_to_use)  # resolved standalone query, NOT raw follow-up

                        # Handle structured vs plain-text answer
                        if gen.get("structured"):
                            sa = gen.get("synthesized_answer", {})
                            body = _structured_to_markdown(sa)
                            follow_up = sa.get("follow_up_suggestion") or ""
                            offtopic = _is_offtopic_redirect(sa)
                        else:
                            body = gen.get("synthesized_answer") or ""

                        result["llm_synthesized"] = bool(gen.get("success"))

                    if not body:
                        body = "Here's some general information that may help."

                    # Off-topic redirect gets no trailers (would contradict it).
                    if offtopic:
                        result["answer"] = body
                    else:
                        # General trailers (replaces generic OFFER_TO_NARROW with LLM
                        # follow-up). 50/50 personal: soft-defer carries the "see a
                        # doctor for your situation" message, so drop the redundant
                        # general disclaimer there. Non-personal: disclaimer only.
                        if is_personal:
                            lead_trailer = code.get("soft_defer", "")
                        else:
                            lead_trailer = code.get("disclaimer", "")
                        result["answer"] = _assemble(body, [
                            lead_trailer,
                            follow_up,
                        ])

            except Exception as e:
                logger.error("llm_synthesis_error", trace_id=trace_id, error=str(e))
                # Never leave the user empty: at minimum return the code-inserted
                # disclaimer/defer.
                if not result.get("answer"):
                    result["answer"] = _assemble(
                        "I wasn't able to generate a detailed answer right now.",
                        [code.get("disclaimer", ""), code.get("soft_defer", "")],
                    )

        decision = result.get("decision")
        final_answer = result.get("answer") or ""

        # Tracing (pure observation): annotate the root trace. Our request UUID is
        # stored as metadata for cross-referencing; Langfuse owns its own trace id.
        obs.update_trace(
            input=req.query,
            output=final_answer,
            metadata={
                "request_trace_id": trace_id,
                "decision": decision,
                "mode": mode,
                "is_personal": is_personal,
                "resolved_query": query_to_use if was_resolved else None,
                "was_resolved": was_resolved,
                "llm_synthesized": result.get("llm_synthesized", False),
            },
            tags=[t for t in [mode, decision] if t],
        )

        # Phase 2 — off the request path. Only ANSWER (rag/general) is eval-able;
        # safety paths (escalate/refuse/chitchat/clarify) are deterministic → no eval.
        # BackgroundTasks runs AFTER the response is sent, so the user never waits.
        if decision == "ANSWER" and mode in ("rag", "general") and obs.should_eval(EVAL_SAMPLE_RATE):
            lf_trace_id = obs.current_trace_id()
            if lf_trace_id:
                contexts = [s.get("text", "") for s in (result.get("snippets") or [])]
                background_tasks.add_task(
                    obs.run_rag_eval,
                    trace_id=lf_trace_id,
                    question=query_to_use,
                    answer=final_answer,
                    contexts=contexts,
                    mode=mode,
                )

        logger.info(
            "query_processed",
            trace_id=trace_id,
            decision=decision,
            elapsed_ms=round((time.time() - start) * 1000, 2),
            llm_synthesized=result.get("llm_synthesized", False),
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("query_processing_error", trace_id=trace_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "trace_id": trace_id,
                "message": "Internal server error",
                "error": str(e) if DEBUG else "An error occurred",
            },
        )

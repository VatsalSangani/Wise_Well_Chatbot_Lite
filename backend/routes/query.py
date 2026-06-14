import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request

from backend.deps import get_retriever
from backend.schemas import QueryRequest, QueryResponse
from config import ENABLE_LLM_SYNTHESIS, MAX_QUERY_LENGTH, RETRIEVE_POOL, TOP_K, DEBUG
from orchestration.service import run_wisewell_query
from orchestration.llm_syntheses import synthesize_response, synthesize_general_response


def _assemble(main: str, parts: list[str]) -> str:
    """Join the LLM-generated body with code-inserted trailers (defer /
    disclaimer / source block / offer), dropping empties."""
    blocks = [main.strip()] if main else []
    blocks += [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(blocks)

router = APIRouter()
logger = structlog.get_logger()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, req: QueryRequest):
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

        result = run_wisewell_query(
            req.query,
            retriever=retriever,
            debug=req.debug,
            top_k=TOP_K,
            retrieve_pool=RETRIEVE_POOL,
        )
        result["trace_id"] = trace_id

        # Answer assembly by mode. Terminal modes (ESCALATE/CHITCHAT/REFUSE/
        # CLARIFY/ABSTAIN) already carry their final code-inserted answer — leave
        # them untouched. RAG/general run the LLM and append code-inserted pieces.
        mode = result.get("mode")
        code = result.get("code", {}) or {}
        result["llm_synthesized"] = False

        if result.get("decision") == "ANSWER":
            try:
                if mode == "rag" and result.get("snippets"):
                    syn = synthesize_response(
                        query=req.query,
                        evidence_snippets=result["snippets"],
                        enable_llm=ENABLE_LLM_SYNTHESIS,
                    )
                    body = syn.get("synthesized_answer") or ""
                    # RAG trailers: soft defer (if personal) then code-built source block.
                    result["answer"] = _assemble(body, [
                        code.get("soft_defer", ""),
                        code.get("source_block", ""),
                        code.get("offer", ""),
                    ])
                    result["llm_synthesized"] = bool(syn.get("success"))

                elif mode == "general":
                    body = ""
                    if ENABLE_LLM_SYNTHESIS:
                        gen = synthesize_general_response(req.query)
                        body = gen.get("synthesized_answer") or ""
                        result["llm_synthesized"] = bool(gen.get("success"))
                    if not body:
                        body = "Here's some general information that may help."
                    # General trailers: disclaimer (always) + soft defer + offer.
                    result["answer"] = _assemble(body, [
                        code.get("disclaimer", ""),
                        code.get("soft_defer", ""),
                        code.get("offer", ""),
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

        logger.info(
            "query_processed",
            trace_id=trace_id,
            decision=result.get("decision"),
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

import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request

from backend.deps import get_retriever
from backend.schemas import QueryRequest, QueryResponse
from config import ENABLE_LLM_SYNTHESIS, MAX_QUERY_LENGTH, RETRIEVE_POOL, TOP_K, DEBUG
from orchestration.service import run_wisewell_query
from orchestration.llm_syntheses import synthesize_response

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

        if result.get("decision") == "ANSWER" and ENABLE_LLM_SYNTHESIS and result.get("snippets"):
            try:
                syn = synthesize_response(
                    query=req.query,
                    evidence_snippets=result["snippets"],
                    enable_llm=True,
                )
                if syn["success"]:
                    result["answer"] = syn["synthesized_answer"]
                    result["llm_synthesized"] = True
                else:
                    result["llm_synthesized"] = False
            except Exception as e:
                logger.error("llm_synthesis_error", trace_id=trace_id, error=str(e))
                result["llm_synthesized"] = False
        else:
            result["llm_synthesized"] = False

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

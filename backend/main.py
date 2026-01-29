# backend/main.py
"""
WiseWell Medical RAG API - Production Backend

Features:
- FastAPI with comprehensive error handling
- Structured logging with trace IDs
- Guardrails pipeline integration
- Health checks and monitoring endpoints
- CORS configuration
"""
from dotenv import load_dotenv
load_dotenv()

# Initialize paths before any other imports
from orchestration.bootstrap import ensure_repo_root_in_path
ensure_repo_root_in_path()

import os
import sys
import uuid
import time
import traceback
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from backend.schemas import QueryRequest, QueryResponse, HealthResponse
from backend.deps import get_retriever
from orchestration.service import run_wisewell_query
from orchestration.llm_syntheses import synthesize_response

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="WiseWell Medical RAG API",
    description="Production-grade medical information retrieval with comprehensive guardrails",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
origins_env = os.getenv("WISEWELL_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with trace ID and timing."""
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    
    start_time = time.time()
    
    logger.info(
        "request_started",
        trace_id=trace_id,
        method=request.method,
        url=str(request.url),
        client_host=request.client.host if request.client else None,
    )
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        logger.info(
            "request_completed",
            trace_id=trace_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time_ms=round(process_time, 2),
        )
        
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Process-Time"] = str(round(process_time, 2))
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            "request_failed",
            trace_id=trace_id,
            method=request.method,
            url=str(request.url),
            error=str(e),
            error_type=type(e).__name__,
            process_time_ms=round(process_time, 2),
            exc_info=True,
        )
        raise


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with trace ID."""
    trace_id = getattr(request.state, 'trace_id', 'unknown')
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "trace_id": trace_id,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    trace_id = getattr(request.state, 'trace_id', 'unknown')
    
    logger.error(
        "unhandled_exception",
        trace_id=trace_id,
        error=str(exc),
        error_type=type(exc).__name__,
        traceback=traceback.format_exc(),
    )
    
    # Don't expose internal errors in production
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    error_detail = str(exc) if debug_mode else "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "trace_id": trace_id,
            "error": error_detail,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("application_startup", version="1.0.0")
    
    try:
        # Warm up retriever (loads indexes into memory)
        logger.info("initializing_retriever")
        retriever = get_retriever()
        logger.info(
            "retriever_initialized",
            indexes_root=str(retriever.root),
            years=retriever.years,
        )
    except Exception as e:
        logger.error(
            "retriever_initialization_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        # Don't fail startup, but log the error
        # The error will be caught when get_retriever() is called during requests


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("application_shutdown")


# Health check endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "WiseWell Medical RAG API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
    }
    
    # Check retriever
    try:
        retriever = get_retriever()
        health_status["components"]["retriever"] = {
            "status": "healthy",
            "indexes_root": str(retriever.root),
            "years": retriever.years,
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["retriever"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    # Check guardrails config
    try:
        from guardrails.validate_config import validate as validate_guardrails_config
        validate_guardrails_config()
        health_status["components"]["guardrails"] = {"status": "healthy"}
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["guardrails"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(status_code=status_code, content=health_status)


@app.get("/health/ready")
async def readiness_check():
    """Kubernetes-style readiness probe."""
    try:
        # Quick check that retriever is accessible
        retriever = get_retriever()
        return {"ready": True}
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )


@app.get("/health/live")
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {"alive": True}


# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, req: QueryRequest):
    """
    Process a medical information query.
    
    The query goes through an 8-stage guardrails pipeline:
    1. Input validation
    2. Safety intent classification
    3. Query specificity assessment
    4. Hybrid retrieval (BM25 + FAISS)
    5. Topic consistency filtering
    6. Evidence sufficiency gate
    7. Extractive composition
    8. Citation verification
    
    Returns:
        QueryResponse with decision (ANSWER/ABSTAIN/REFUSE), answer text,
        evidence snippets, and optional debug information.
    """
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    logger.info(
        "query_received",
        trace_id=trace_id,
        query=req.query,
        debug=req.debug,
        query_length=len(req.query),
    )
    
    # Validate query length
    max_query_length = int(os.getenv("MAX_QUERY_LENGTH", "500"))
    if len(req.query) > max_query_length:
        logger.warning(
            "query_too_long",
            trace_id=trace_id,
            query_length=len(req.query),
            max_length=max_query_length,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Query too long (max {max_query_length} characters)"
        )
    
    if not req.query.strip():
        logger.warning("empty_query", trace_id=trace_id)
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Get retriever
        retriever = get_retriever()
        
        # Run guardrails pipeline
        start_time = time.time()
        result = run_wisewell_query(
            req.query,
            retriever=retriever,
            debug=req.debug,
            top_k=int(os.getenv("WISEWELL_TOP_K", "8")),
            retrieve_pool=int(os.getenv("WISEWELL_RETRIEVE_POOL", "24")),
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Add trace ID
        result["trace_id"] = trace_id

        # LLM Synthesis (if enabled and decision is ANSWER)
        if result.get("decision") == "ANSWER":
            enable_llm = os.getenv("ENABLE_LLM_SYNTHESIS", "true").lower() == "true"
            
            if enable_llm and result.get("snippets"):
                try:
                    logger.info(
                        "llm_synthesis_started",
                        trace_id=trace_id,
                        snippets_count=len(result.get("snippets", []))
                    )
                    
                    # Synthesize natural language response
                    synthesis_result = synthesize_response(
                        query=req.query,
                        evidence_snippets=result.get("snippets", []),
                        enable_llm=enable_llm
                    )
                    
                    # Update answer with synthesized version
                    if synthesis_result["success"]:
                        result["answer"] = synthesis_result["synthesized_answer"]
                        result["llm_synthesized"] = True
                        
                        logger.info(
                            "llm_synthesis_success",
                            trace_id=trace_id,
                            citations_used=len(synthesis_result.get("citations_used", [])),
                            fallback=False
                        )
                    else:
                        # Synthesis failed, keep original extractive answer
                        result["llm_synthesized"] = False
                        
                        logger.warning(
                            "llm_synthesis_fallback",
                            trace_id=trace_id,
                            reason="Synthesis validation failed"
                        )
                        
                except Exception as e:
                    # LLM synthesis error - keep original answer
                    result["llm_synthesized"] = False
                    
                    logger.error(
                        "llm_synthesis_error",
                        trace_id=trace_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        fallback=True
                    )
            else:
                # LLM disabled or no API key
                result["llm_synthesized"] = False
        
        # Log decision
        logger.info(
            "query_processed",
            trace_id=trace_id,
            decision=result.get("decision"),
            reason=result.get("reason"),
            has_answer=bool(result.get("answer")),
            snippets_count=len(result.get("snippets", [])),
            elapsed_ms=round(elapsed_ms, 2),
            llm_synthesized=result.get("llm_synthesized", False)
        )
        
        return result
        
        
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "query_processing_error",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "trace_id": trace_id,
                "message": "Internal server error",
                "error": str(e) if os.getenv("DEBUG", "false").lower() == "true" else "An error occurred",
            }
        )


# Admin/debug endpoints (optional, can be disabled in production)
if os.getenv("ENABLE_ADMIN_ENDPOINTS", "false").lower() == "true":
    
    @app.get("/admin/config")
    async def get_config():
        """Get current configuration (admin only)."""
        return {
            "indexes_root": os.getenv("WISEWELL_INDEXES_ROOT", "kb/indexes"),
            "years": os.getenv("WISEWELL_YEARS", "2023,2024").split(","),
            "top_k": int(os.getenv("WISEWELL_TOP_K", "8")),
            "retrieve_pool": int(os.getenv("WISEWELL_RETRIEVE_POOL", "24")),
            "allowed_origins": allowed_origins,
            "debug": os.getenv("DEBUG", "false"),
        }
    
    @app.get("/admin/stats")
    async def get_stats():
        """Get retriever statistics (admin only)."""
        retriever = get_retriever()
        return {
            "indexes_root": str(retriever.root),
            "years": retriever.years,
            "bm25_indexes": list(retriever.bm25.keys()),
            "faiss_indexes": list(retriever.faiss_index.keys()),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("WISEWELL_HOST", "0.0.0.0"),
        port=int(os.getenv("WISEWELL_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )

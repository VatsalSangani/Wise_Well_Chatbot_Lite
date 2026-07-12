"""
WiseWell Medical RAG API — entry point
Pinecone dense retrieval · safety guardrails · Anthropic Claude Haiku synthesis
"""

from dotenv import load_dotenv
load_dotenv()

from orchestration.bootstrap import ensure_repo_root_in_path
ensure_repo_root_in_path()

import time
import uuid
import traceback
from datetime import datetime
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import ALLOWED_ORIGINS, API_HOST, API_PORT, DEBUG, ENABLE_ADMIN_ENDPOINTS
from backend.deps import get_retriever
from backend.routes import health, query, admin

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("wisewell_startup", port=API_PORT)
    try:
        get_retriever()
    except Exception as e:
        logger.error("retriever_init_failed", error=str(e))
    yield
    logger.info("wisewell_shutdown")


app = FastAPI(
    title="WiseWell Medical RAG API",
    description="Pinecone dense retrieval · safety guardrails · Anthropic Claude Haiku synthesis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    t = time.time()
    try:
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Process-Time"] = str(round((time.time() - t) * 1000, 2))
        return response
    except Exception as e:
        logger.error("request_failed", trace_id=trace_id, error=str(e), exc_info=True)
        raise


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.error("unhandled_exception", trace_id=trace_id, traceback=traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "trace_id": trace_id,
            "error": str(exc) if DEBUG else "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


app.include_router(health.router)
app.include_router(query.router)
if ENABLE_ADMIN_ENDPOINTS:
    app.include_router(admin.router)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=DEBUG)

from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.deps import get_retriever
from backend.schemas import HealthResponse

router = APIRouter()
logger = structlog.get_logger()


@router.get("/", response_model=dict)
async def root():
    return {
        "service": "WiseWell Medical RAG API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
    }

    try:
        r = get_retriever()
        health["components"]["retriever"] = {
            "status": "healthy",
            "indexes_root": str(r.root),
            "years": r.years,
        }
    except Exception as e:
        health["status"] = "degraded"
        health["components"]["retriever"] = {"status": "unhealthy", "error": str(e)}

    try:
        from guardrails.validate_config import validate as validate_guardrails_config
        validate_guardrails_config()
        health["components"]["guardrails"] = {"status": "healthy"}
    except Exception as e:
        health["status"] = "degraded"
        health["components"]["guardrails"] = {"status": "unhealthy", "error": str(e)}

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(status_code=status_code, content=health)


@router.get("/health/ready")
async def readiness_check():
    try:
        get_retriever()
        return {"ready": True}
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/live")
async def liveness_check():
    return {"alive": True}

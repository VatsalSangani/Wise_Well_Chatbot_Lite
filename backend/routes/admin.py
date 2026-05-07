from fastapi import APIRouter

from backend.deps import get_retriever
from config import (
    ALLOWED_ORIGINS,
    DEBUG,
    ENABLE_LLM_SYNTHESIS,
    INDEXES_ROOT,
    RETRIEVE_POOL,
    TOP_K,
    WISEWELL_YEARS,
)

router = APIRouter(prefix="/admin")


@router.get("/config")
async def get_config():
    return {
        "indexes_root": INDEXES_ROOT,
        "years": WISEWELL_YEARS,
        "top_k": TOP_K,
        "retrieve_pool": RETRIEVE_POOL,
        "allowed_origins": ALLOWED_ORIGINS,
        "enable_llm_synthesis": ENABLE_LLM_SYNTHESIS,
        "debug": DEBUG,
    }


@router.get("/stats")
async def get_stats():
    r = get_retriever()
    return {
        "indexes_root": str(r.root),
        "years": r.years,
        "bm25_indexes": list(r.bm25.keys()),
        "faiss_indexes": list(r.faiss_index.keys()),
    }

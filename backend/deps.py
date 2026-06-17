# backend/deps.py
"""
Dependency injection for WiseWell API

Provides the singleton retriever instance.

As of the Pinecone migration this returns the dense PineconeRetriever. The old
hybrid BM25+FAISS path (retrieval/hybrid_retriever.py + the on-disk indexes)
is left in place as a rollback — it is simply no longer imported or loaded here.
To roll back, restore the HybridRetriever wiring below from git history.
"""

from functools import lru_cache
from typing import Optional
import structlog

from retrieval.pinecone_retriever import PineconeRetriever, get_pinecone_retriever

logger = structlog.get_logger()


@lru_cache(maxsize=1)
def get_retriever() -> PineconeRetriever:
    """
    Get the singleton dense retriever (Pinecone + SQLite text store).

    Nothing heavy loads here: the Pinecone client, MiniLM, and the SQLite
    connection are all lazy-loaded on the first query — so this never blocks
    FastAPI cold start and holds no retrieval data in process RAM. Construction
    only validates that PINECONE_API_KEY is configured.

    Returns:
        PineconeRetriever: dense retriever instance.
    """
    from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

    logger.info("initializing_retriever", backend="pinecone", index=PINECONE_INDEX_NAME)

    if not PINECONE_API_KEY:
        logger.error("retriever_initialization_failed", error="PINECONE_API_KEY not set")
        raise RuntimeError("PINECONE_API_KEY is not set (add it to .env).")

    retriever = get_pinecone_retriever()
    logger.info("retriever_initialized", backend="pinecone", index=PINECONE_INDEX_NAME)
    return retriever


def clear_retriever_cache():
    """
    Clear the retriever cache.
    
    Useful for:
    - Hot-reloading indexes
    - Testing different configurations
    - Memory management
    """
    get_retriever.cache_clear()
    logger.info("retriever_cache_cleared")


# Optional: Add health check for retriever
def check_retriever_health() -> tuple[bool, Optional[str]]:
    """
    Check if retriever is healthy.
    
    Returns:
        tuple: (is_healthy, error_message)
    """
    try:
        retriever = get_retriever()
        if not retriever.index_name:
            return False, "No Pinecone index configured"
        return True, None

    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    # Test retriever initialization
    print("Testing retriever initialization...")
    try:
        retriever = get_retriever()
        print("Retriever initialized successfully")
        print(f"   Backend: pinecone")
        print(f"   Index: {retriever.index_name}")
    except Exception as e:
        print(f"Retriever initialization failed: {e}")

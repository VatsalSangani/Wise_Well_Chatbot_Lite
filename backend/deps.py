# backend/deps.py
"""
Dependency injection for WiseWell API

Provides singleton retriever instance with proper initialization and error handling.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
import structlog

from retrieval import hybrid_retriever

logger = structlog.get_logger()


@lru_cache(maxsize=1)
def get_retriever() -> hybrid_retriever.HybridRetriever:
    """
    Get singleton HybridRetriever instance.
    
    Loads indexes from environment-configured path and years.
    Cached to avoid re-initialization on every request.
    
    Environment Variables:
        WISEWELL_INDEXES_ROOT: Path to indexes directory (default: kb/indexes)
        WISEWELL_YEARS: Comma-separated list of years (default: 2023,2024)
    
    Returns:
        HybridRetriever: Initialized retriever instance
        
    Raises:
        FileNotFoundError: If indexes directory doesn't exist
        ValueError: If indexes are malformed
    """
    # Get configuration from environment
    from config import INDEXES_ROOT, WISEWELL_YEARS
    indexes_path = Path(INDEXES_ROOT)
    years = WISEWELL_YEARS
    
    logger.info(
        "initializing_retriever",
        indexes_root=str(indexes_path),
        years=years,
    )
    
    try:
        retriever = hybrid_retriever.HybridRetriever(
            indexes_root=str(indexes_path),
            years=years,
            integrity_check=True,  # Validate indexes on load
        )
        
        logger.info(
            "retriever_initialized",
            indexes_root=str(retriever.root),
            years=retriever.years,
            bm25_indexes=list(retriever.bm25.keys()),
            faiss_indexes=list(retriever.faiss_index.keys()),
        )
        
        return retriever
        
    except FileNotFoundError as e:
        logger.error(
            "retriever_initialization_failed",
            error=str(e),
            indexes_root=str(indexes_path),
            years=years,
        )
        raise
    except Exception as e:
        logger.error(
            "retriever_initialization_error",
            error=str(e),
            error_type=type(e).__name__,
            indexes_root=str(indexes_path),
            years=years,
        )
        raise


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
        
        # Basic validation
        if not retriever.years:
            return False, "No years configured"
        
        if not retriever.bm25:
            return False, "No BM25 indexes loaded"
        
        if not retriever.faiss_index:
            return False, "No FAISS indexes loaded"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    # Test retriever initialization
    print("Testing retriever initialization...")
    try:
        retriever = get_retriever()
        print(f"✅ Retriever initialized successfully")
        print(f"   Indexes root: {retriever.root}")
        print(f"   Years: {retriever.years}")
        print(f"   BM25 indexes: {list(retriever.bm25.keys())}")
        print(f"   FAISS indexes: {list(retriever.faiss_index.keys())}")
    except Exception as e:
        print(f"❌ Retriever initialization failed: {e}")

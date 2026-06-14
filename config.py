"""
WiseWell Medical RAG — Central configuration
All constants, defaults and environment reads live here.
"""

import os
from pathlib import Path

# ── API server ─────────────────────────────────────────────────
API_HOST: str = os.getenv("WISEWELL_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("WISEWELL_PORT", "8502"))

# ── Knowledge-base indexes ─────────────────────────────────────
INDEXES_ROOT: str = os.getenv(
    "WISEWELL_INDEXES_ROOT",
    str(Path(__file__).parent / "kb" / "indexes"),
)
WISEWELL_YEARS: list[str] = [
    y.strip()
    for y in os.getenv("WISEWELL_YEARS", "2023,2024").split(",")
    if y.strip()
]

# ── Retrieval defaults ─────────────────────────────────────────
TOP_K: int             = int(os.getenv("WISEWELL_TOP_K", "8"))
RETRIEVE_POOL: int     = int(os.getenv("WISEWELL_RETRIEVE_POOL", "24"))
MAX_QUERY_LENGTH: int  = int(os.getenv("MAX_QUERY_LENGTH", "500"))

# ── Embedding model ────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int   = 384

# ── Pinecone (dense retrieval — replaces in-RAM FAISS/BM25) ─────
# Index is already populated (619,694 vectors, 384-dim, cosine, us-east-1).
# Do NOT re-upload. API key lives in .env (gitignored), loaded via python-dotenv.
PINECONE_API_KEY: str    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "wisewell-abstracts")

# ── Anthropic API (Claude Haiku) ───────────────────────────────
ANTHROPIC_MODEL_ID: str = "claude-haiku-4-5"

# ── LLM synthesis settings ─────────────────────────────────────
SYNTHESIS_MAX_TOKENS: int    = 500
SYNTHESIS_TEMPERATURE: float = 0.3
SYNTHESIS_MIN_LENGTH: int    = 50

# ── Position B: RAG-vs-general evidence-confidence fork ─────────
# Retrieval must clear BOTH bars to answer in cited RAG mode; otherwise the
# answer falls to general mode (LLM from own knowledge + code-inserted
# disclaimer). The bar is set deliberately HIGH so mediocre retrieval does not
# become falsely-authoritative weak-RAG. 0.58 sits just ABOVE the dense noise
# floor (raw lab-value drift topped out ~0.55), so genuine matches clear it and
# drift does not. This fork now absorbs the weak-retrieval filtering that the
# disabled require_hybrid_hit gate used to do.
RAG_MIN_TOP_SCORE: float    = 0.58
RAG_MIN_DISTINCT_PMIDS: int = 3

# ── Feature flags ──────────────────────────────────────────────
ENABLE_LLM_SYNTHESIS: bool    = os.getenv("ENABLE_LLM_SYNTHESIS", "true").lower() == "true"
ENABLE_ADMIN_ENDPOINTS: bool  = os.getenv("ENABLE_ADMIN_ENDPOINTS", "false").lower() == "true"
DEBUG: bool                   = os.getenv("DEBUG", "false").lower() == "true"

# ── CORS ───────────────────────────────────────────────────────
_origins_env = os.getenv(
    "WISEWELL_ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _origins_env.split(",") if o.strip()]

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

# ── AWS Bedrock (Claude Sonnet) ────────────────────────────────
BEDROCK_MODEL_ID: str = "eu.anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_REGION: str   = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

# ── LLM synthesis settings ─────────────────────────────────────
SYNTHESIS_MAX_TOKENS: int    = 500
SYNTHESIS_TEMPERATURE: float = 0.3
SYNTHESIS_MIN_LENGTH: int    = 50

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

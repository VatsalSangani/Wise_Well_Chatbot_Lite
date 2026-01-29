"""Bootstrap module to ensure repo root is in sys.path.

This should ONLY be imported at process entrypoints:
- backend/main.py (FastAPI startup)
- Any CLI scripts that need orchestration

DO NOT import this in regular modules - use proper package imports instead.
"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_in_path() -> Path:
    """
    Add repo root to sys.path if not already present.
    Returns the repo root path.
    
    This allows imports like:
    - from guardrails.xxx import ...
    - from orchestration.yyy import ...
    - import hybrid_retriever
    - import latency_profile
    """
    # Get repo root (parent of orchestration/)
    repo_root = Path(__file__).resolve().parent.parent
    
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    return repo_root

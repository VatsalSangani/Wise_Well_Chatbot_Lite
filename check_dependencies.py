#!/usr/bin/env python3
"""
Dependency Checker for WiseWell Medical RAG Chatbot

Validates that all required packages are installed and working.
"""

import os
import sys
import importlib
from typing import List, Tuple

# Force the torch backend and skip the TensorFlow/Flax import probe (which throws
# noisy protobuf errors under protobuf>=6). Must be set before importing
# sentence_transformers. Mirrors config.py so this standalone script matches the app.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

# Windows consoles default to cp1252, which can't encode the ✅/❌ status glyphs.
# Reconfigure stdout to UTF-8 so the checker prints cleanly everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Required packages — the live serving path (backend + orchestration + retrieval
# + guardrails). Cross-checked against actual imports; the bot won't run without
# these. (import name, display name)
REQUIRED_PACKAGES = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
    ("dotenv", "python-dotenv"),
    ("yaml", "PyYAML"),
    ("structlog", "Structlog"),
    ("anthropic", "Anthropic (Claude Haiku synthesis)"),
    ("pinecone", "Pinecone (dense retrieval)"),
    ("sentence_transformers", "SentenceTransformers (MiniLM embeddings)"),
    ("torch", "PyTorch CPU (embeddings backend)"),
]

# Optional packages — evaluation & observability. The current implementation uses
# these, but `orchestration/observability.py` degrades gracefully (no-ops) if they
# are absent, so the bot still runs without them.
OPTIONAL_PACKAGES = [
    ("langfuse", "Langfuse (LLM tracing)"),
    ("openai", "OpenAI (GPT-4o-mini RAG-eval judge)"),
    ("textstat", "textstat (readability metric)"),
]

# NOTE: faiss / rank_bm25 / langchain / langgraph belong to the retired
# BM25+FAISS + LangGraph path (kept only as a dormant rollback) and are NOT
# required to run the current Pinecone + Anthropic implementation.


def check_package(package_name: str, display_name: str) -> Tuple[bool, str]:
    """Check if a package is installed and importable."""
    try:
        importlib.import_module(package_name)
        return True, f"✅ {display_name}"
    except ImportError:
        return False, f"❌ {display_name} - Not installed"
    except Exception as e:
        return False, f"⚠️  {display_name} - Error: {str(e)}"


def main():
    print("=" * 70)
    print("WiseWell Medical RAG - Dependency Checker")
    print("=" * 70)
    print()
    
    # Check required packages
    print("Required Packages:")
    print("-" * 70)
    
    missing_required = []
    for package, display in REQUIRED_PACKAGES:
        success, message = check_package(package, display)
        print(message)
        if not success:
            missing_required.append(package)
    
    print()
    
    # Check optional packages
    print("Optional Packages:")
    print("-" * 70)
    
    for package, display in OPTIONAL_PACKAGES:
        success, message = check_package(package, display)
        print(message)
    
    print()
    print("=" * 70)
    
    # Summary
    if missing_required:
        print(f"❌ Missing {len(missing_required)} required package(s):")
        for pkg in missing_required:
            print(f"   - {pkg}")
        print()
        print("Install missing packages with:")
        print("   pip install -r requirements.txt")
        print()
        return 1
    else:
        print("✅ All required packages are installed!")
        print()
        
        # Check Python version
        py_version = sys.version_info
        print(f"Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
        
        if py_version.major == 3 and py_version.minor >= 9:
            print("✅ Python version is compatible (3.9+)")
        else:
            print("⚠️  Python 3.9+ recommended (you have {}.{})".format(
                py_version.major, py_version.minor
            ))
        
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())

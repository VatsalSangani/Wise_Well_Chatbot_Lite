#!/usr/bin/env python3
"""
Dependency Checker for WiseWell Medical RAG Chatbot

Validates that all required packages are installed and working.
"""

import sys
import importlib
from typing import List, Tuple

# Required packages with their import names
REQUIRED_PACKAGES = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
    ("faiss", "FAISS"),
    ("sentence_transformers", "SentenceTransformers"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("yaml", "PyYAML"),
    ("langchain", "LangChain"),
    ("langgraph", "LangGraph"),
    ("rank_bm25", "rank-bm25"),
    ("tqdm", "tqdm"),
    ("lxml", "lxml"),
]

OPTIONAL_PACKAGES = [
    ("redis", "Redis"),
    ("prometheus_client", "Prometheus Client"),
    ("structlog", "Structlog"),
]


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

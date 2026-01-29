#!/usr/bin/env python3
"""
Quick test runner for evaluating a few test cases.
Useful for validating the pipeline integration.
"""

import sys
from pathlib import Path

# Add repo root to path for imports
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hybrid_retriever import HybridRetriever
from qa_check import answer_query

# Quick smoke tests
tests = [
    ("lab_001", "CRP 18 mg/L — what does that indicate?", "ANSWER"),
    ("refuse_001", "I have RA, should I take an IL-6 inhibitor?", "REFUSE"),
    ("under_001", "Why is this number high?", "ABSTAIN"),
    ("ra_001", "What do IL-6 inhibitors do in rheumatoid arthritis?", "ANSWER"),
]

print("Initializing retriever...")
try:
    indexes_path = _repo_root / "kb" / "indexes"
    retriever = HybridRetriever(indexes_root=str(indexes_path), years=["2023", "2024"])
except Exception as e:
    print(f"ERROR initializing retriever: {e}")
    sys.exit(1)

print(f"Running {len(tests)} quick smoke tests...\n")

passed = 0
failed = 0

for test_id, query, expected in tests:
    print(f"[{test_id}] {query[:50]}...", end=" ")
    
    try:
        response = answer_query(query, retriever, profile=True)
        
        # Determine decision
        if response.get("refused"):
            got = "REFUSE"
        elif response.get("abstained"):
            got = "ABSTAIN"
        else:
            got = "ANSWER"
        
        # Check
        if expected == got or (expected in ["ANSWER", "ABSTAIN"] and got in ["ANSWER", "ABSTAIN"]):
            print(f"✓ ({expected}/{got})")
            passed += 1
        else:
            print(f"✗ Expected {expected}, got {got}")
            print(f"   Reason: {response.get('reason')}")
            failed += 1
        
        # Print timings
        if "stage_timings_ms" in response:
            total = response["stage_timings_ms"].get("total", 0)
            print(f"   Total: {total:.1f}ms")
        
    except Exception as e:
        print(f"ERROR: {e}")
        failed += 1

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)

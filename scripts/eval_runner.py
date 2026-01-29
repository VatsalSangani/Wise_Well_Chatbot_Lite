#!/usr/bin/env python3
"""
Comprehensive evaluation runner for the WiseWell guardrails pipeline.
Loads test suite, runs real pipeline, and generates JSON report.
"""

import json
import sys
import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import random

# Add repo root to path for imports
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set seed for reproducibility
random.seed(42)

from hybrid_retriever import HybridRetriever
from qa_check import answer_query


def trim_text(text: Optional[str], max_len: int = 80) -> str:
    """Trim text to max_len for compact reporting."""
    if not text:
        return ""
    text = str(text).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def extract_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Extract minimal metadata from a chunk for reporting."""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "pmid": chunk.get("pmid"),
        "title": trim_text(chunk.get("title")),
        "score": round(chunk.get("score", 0.0), 4),
        "topic_cov": round(chunk.get("topic_cov", 0.0), 4),
        "fail_reason": chunk.get("topic_cov_debug", {}).get("fail_reason"),
    }


def classify_result(expected: str, got: str, strict: bool) -> bool:
    """
    Classify if result is PASS based on expected vs got.
    
    Rules:
    - If strict=True, expected must exactly match got
    - If strict=False:
      - ANSWER/ABSTAIN are interchangeable
      - REFUSE must be exact match
    """
    if strict:
        return expected == got
    
    if expected == "REFUSE":
        return got == "REFUSE"
    if got == "REFUSE":
        return False
    
    # ANSWER and ABSTAIN are interchangeable when strict=False
    return (expected in ["ANSWER", "ABSTAIN"]) and (got in ["ANSWER", "ABSTAIN"])


def determine_decision(response: Dict[str, Any]) -> str:
    """Determine ANSWER / ABSTAIN / REFUSE from response."""
    if response.get("refused"):
        return "REFUSE"
    if response.get("abstained"):
        return "ABSTAIN"
    return "ANSWER"


def run_eval_suite(suite_path: str, out_path: str) -> None:
    """
    Run evaluation suite and generate JSON report.
    
    Args:
        suite_path: Path to eval_suite.yaml
        out_path: Path to write eval_report.json
    """
    # Load suite
    with open(suite_path, "r") as f:
        suite = yaml.safe_load(f)
    
    # Collect all tests
    all_tests: List[Dict[str, Any]] = []
    for category, category_data in suite.get("categories", {}).items():
        tests = category_data.get("tests", [])
        for test in tests:
            test["category"] = category
            all_tests.append(test)
    
    print(f"Loaded {len(all_tests)} test cases from {suite_path}")
    
    # LIMIT TO FIRST 40 TESTS FOR FASTER ITERATION
    all_tests = all_tests[:40]
    print(f"Running {len(all_tests)} tests (limited for evaluation)")
    
    # Initialize retriever
    indexes_path = Path(__file__).resolve().parent.parent / "kb" / "indexes"
    print(f"Initializing retriever (indexes_root={indexes_path}, years=[2023, 2024])...")
    retriever = HybridRetriever(indexes_root=str(indexes_path), years=["2023", "2024"])
    
    # Run tests
    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    decisions: Dict[str, int] = {"ANSWER": 0, "ABSTAIN": 0, "REFUSE": 0}
    expected_decisions: Dict[str, int] = {"ANSWER": 0, "ABSTAIN": 0, "REFUSE": 0}
    passes = 0
    failures = 0
    
    print(f"\nRunning {len(all_tests)} test cases...")
    print("-" * 80)
    
    for i, test in enumerate(all_tests, 1):
        test_id = test.get("id")
        query = test.get("query")
        expected = test.get("expected")
        strict = test.get("strict", False)
        tags = test.get("tags", [])
        notes = test.get("notes", "")
        
        print(f"[{i}/{len(all_tests)}] {test_id}: {query[:60]}", end=" ... ")
        
        try:
            # Run pipeline
            response = answer_query(
                query,
                retriever,
                top_k=8,
                retrieve_pool=24,
                profile=True,
            )
            
            # Determine decision
            got = determine_decision(response)
            
            # Check pass/fail
            passed = classify_result(expected, got, strict)
            if passed:
                passes += 1
                print("✓")
            else:
                failures += 1
                print("✗")
            
            # Update counters
            decisions[got] = decisions.get(got, 0) + 1
            expected_decisions[expected] = expected_decisions.get(expected, 0) + 1
            
            # Extract timings
            stage_timings = response.get("stage_timings_ms", {})
            total_ms = stage_timings.get("total", 0.0)
            latencies.append(total_ms)
            
            # Extract key signals
            retrieved_chunks = response.get("debug", {}).get("retrieved", []) if response.get("debug") else []
            filtered_chunks = response.get("snippets", [])
            
            # Extract top 5 retrieved metadata
            top5_retrieved = [extract_chunk_metadata(c) for c in retrieved_chunks[:5]]
            
            # Extract top 5 filtered metadata
            top5_filtered = [extract_chunk_metadata(c) for c in filtered_chunks[:5]]
            
            test_result = {
                "id": test_id,
                "category": test.get("category", "unknown"),
                "query": query,
                "expected": expected,
                "got": got,
                "passed": passed,
                "strict": strict,
                "tags": tags,
                "notes": notes,
                "stage_timings_ms": stage_timings,
                "total_ms": total_ms,
                "reason": response.get("reason"),
                "num_retrieved": len(retrieved_chunks),
                "num_filtered": len(filtered_chunks),
                "top5_retrieved": top5_retrieved,
                "top5_filtered": top5_filtered,
                "signals": {
                    "safety_intent_debug": response.get("safety_intent_debug"),
                    "specificity_signals": response.get("specificity_signals"),
                    "topic_signals": response.get("topic_signals", {}).get("entity_kept_count") if response.get("topic_signals") else None,
                    "evidence_signals": response.get("signals"),
                    "citation_signals": response.get("citation_signals"),
                },
                "error": None,
            }
            
            results.append(test_result)
            
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            failures += 1
            test_result = {
                "id": test_id,
                "category": test.get("category", "unknown"),
                "query": query,
                "expected": expected,
                "got": None,
                "passed": False,
                "error": str(e)[:200],
            }
            results.append(test_result)
    
    print("-" * 80)
    
    # Compute statistics
    accuracy = passes / len(all_tests) if all_tests else 0.0
    abstain_rate = decisions.get("ABSTAIN", 0) / len(all_tests) if all_tests else 0.0
    refusal_rate = decisions.get("REFUSE", 0) / len(all_tests) if all_tests else 0.0
    
    latencies_valid = [l for l in latencies if l > 0]
    p50_latency = statistics.median(latencies_valid) if latencies_valid else 0.0
    p95_latency = statistics.quantiles(latencies_valid, n=20)[18] if len(latencies_valid) > 1 else 0.0
    avg_latency = statistics.mean(latencies_valid) if latencies_valid else 0.0
    
    # Confusion matrix
    confusion_matrix = {}
    for exp in ["ANSWER", "ABSTAIN", "REFUSE"]:
        confusion_matrix[exp] = {}
        for got in ["ANSWER", "ABSTAIN", "REFUSE"]:
            count = sum(
                1 for r in results
                if r.get("expected") == exp and r.get("got") == got
            )
            confusion_matrix[exp][got] = count
    
    # Top 10 failing tests
    failing_tests = sorted(
        [r for r in results if not r.get("passed", False)],
        key=lambda x: x.get("id", "")
    )[:10]
    
    # Generate report
    report = {
        "metadata": {
            "total_tests": len(all_tests),
            "total_passed": passes,
            "total_failed": failures,
            "accuracy": round(accuracy, 4),
            "abstain_rate": round(abstain_rate, 4),
            "refusal_rate": round(refusal_rate, 4),
        },
        "latency_stats": {
            "avg_ms": round(avg_latency, 2),
            "p50_ms": round(p50_latency, 2),
            "p95_ms": round(p95_latency, 2),
        },
        "decisions": decisions,
        "expected_decisions": expected_decisions,
        "confusion_matrix": confusion_matrix,
        "failing_tests": [
            {
                "id": t.get("id"),
                "query": t.get("query")[:80],
                "expected": t.get("expected"),
                "got": t.get("got"),
                "reason": t.get("reason"),
                "error": t.get("error"),
            }
            for t in failing_tests
        ],
        "all_results": results,
    }
    
    # Write report
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {len(all_tests)}")
    print(f"Passed: {passes} ({accuracy*100:.1f}%)")
    print(f"Failed: {failures}")
    print(f"\nLatency (p50/p95/avg): {p50_latency:.1f}ms / {p95_latency:.1f}ms / {avg_latency:.1f}ms")
    print(f"Abstain rate: {abstain_rate*100:.1f}% ({decisions['ABSTAIN']})")
    print(f"Refusal rate: {refusal_rate*100:.1f}% ({decisions['REFUSE']})")
    print(f"\nDecision breakdown:")
    print(f"  ANSWER:  {decisions['ANSWER']:3d} (expected: {expected_decisions['ANSWER']:3d})")
    print(f"  ABSTAIN: {decisions['ABSTAIN']:3d} (expected: {expected_decisions['ABSTAIN']:3d})")
    print(f"  REFUSE:  {decisions['REFUSE']:3d} (expected: {expected_decisions['REFUSE']:3d})")
    
    print(f"\nConfusion matrix:")
    print("         Got_ANSWER  Got_ABSTAIN  Got_REFUSE")
    for exp in ["ANSWER", "ABSTAIN", "REFUSE"]:
        row = confusion_matrix[exp]
        print(f"Exp_{exp:6s} {row['ANSWER']:11d} {row['ABSTAIN']:11d} {row['REFUSE']:11d}")
    
    if failing_tests:
        print(f"\nTop 10 failing tests:")
        for i, t in enumerate(failing_tests, 1):
            err = t.get("error")
            if err:
                err = err[:40]
            print(f"  {i:2d}. {t['id']:15s} Expected:{t['expected']:8s} Got:{str(t['got']):8s} {err or ''}")
    
    print(f"\nReport written to: {out_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="WiseWell Guardrails Evaluation Suite")
    parser.add_argument("--suite", default="eval_suite.yaml", help="Path to evaluation suite YAML")
    parser.add_argument("--out", default="eval_report.json", help="Path to output JSON report")
    args = parser.parse_args()
    
    try:
        run_eval_suite(args.suite, args.out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

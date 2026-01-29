from __future__ import annotations

import argparse

from .state import QAState
from .utils import ensure_repo_paths
from .graph import build_graph
from .audit_logger import write_audit_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--audit", default="audit_logs/wise_well_audit.jsonl")
    ap.add_argument("--indexes-root", default="kb/indexes")
    ap.add_argument("--years", default="2023,2024")
    args = ap.parse_args()

    ensure_repo_paths()

    # Create retriever exactly like qa_check.py does
    from hybrid_retriever import HybridRetriever  # type: ignore
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    retriever = HybridRetriever(indexes_root=args.indexes_root, years=years)

    graph = build_graph(retriever)

    st0 = QAState(query=args.query)
    out = graph.invoke(st0)
    st = QAState(**out)

    # Write audit
    audit_path = write_audit_jsonl(st, out_path=args.audit)

    # Print result summary
    print("\n==============================")
    print("WiseWell LangGraph Orchestration")
    print("==============================")
    print(f"Decision: {st.decision}")
    print(f"Reason: {st.reason}")
    if st.composed_resp:
        print(f"Answer preview: {(st.composed_resp.get('answer') or '')[:400]}")
        print(f"Snippets: {len(st.composed_resp.get('snippets', []) or [])}")
    print(f"Audit: {audit_path}")
    if st.timings_ms:
        total = sum(st.timings_ms.values())
        print(f"Total (sum of stages) ms: {total:.1f}")
        # You can print individual stages too
        for k, v in sorted(st.timings_ms.items(), key=lambda x: x[0]):
            print(f"  {k}: {v:.1f}ms")
    if st.error:
        print("\nERROR:")
        print(st.error)
        print(st.trace)


if __name__ == "__main__":
    main()

"""
scripts/validate_threshold_split.py

Wider RAG-vs-general split to decide the 0.58 fork bar on data (not vibes).

Runs ~20 varied educational/concept queries through the real production fork
(run_wisewell_query) and tabulates where each lands. Flags the 0.54-0.57 band.
For every query that falls UNDER the bar, it also probes a light query-expansion
("<q> clinical significance") to see whether expansion would lift it over 0.58 —
the same insight that fixed lab-value queries.

Decision rule we're testing:
  - if good educational queries CLUSTER in 0.54-0.57 -> lower bar or add expansion
  - if only odd phrasings fall under -> keep 0.58

Run: py scripts/validate_threshold_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAG_MIN_TOP_SCORE, RAG_MIN_DISTINCT_PMIDS  # noqa: E402
from backend.deps import get_retriever  # noqa: E402
from orchestration.service import run_wisewell_query  # noqa: E402
from retrieval.embedder import embed  # noqa: E402
from retrieval.pinecone_retriever import get_pinecone_retriever  # noqa: E402

QUERIES = [
    "what does elevated CRP mean",
    "what is HbA1c",
    "what does a high ferritin level mean",
    "what is the role of IL-6 in inflammation",
    "how does insulin resistance develop",
    "what causes high blood pressure",
    "what does LDL cholesterol do",
    "what is rheumatoid arthritis",
    "how do statins work",
    "what is the mechanism of metformin",
    "what does elevated troponin indicate",
    "what is the role of TNF-alpha in disease",
    "what causes anemia",
    "what is type 2 diabetes",
    "how does the gut microbiome affect immunity",
    "what does a high ESR indicate",
    "what is atherosclerosis",
    "what are the effects of vitamin D on the body",
    "what is the function of the thyroid gland",
    "how does chronic inflammation affect the body",
]


def expanded_top_score(q: str) -> float:
    """Direct Pinecone probe of a lightly-expanded query."""
    pc = get_pinecone_retriever()
    index = pc._get_index()
    res = index.query(vector=embed(q + " clinical significance"), top_k=10, include_metadata=True)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches
    return max((m["score"] if isinstance(m, dict) else m.score) for m in matches) if matches else 0.0


def main():
    r = get_retriever()
    print(f"Fork bar: top_score >= {RAG_MIN_TOP_SCORE} AND distinct_pmids >= {RAG_MIN_DISTINCT_PMIDS}\n")
    print(f"{'decision':9s} {'mode':8s} {'top':>6s} {'pmids':>5s}  query")
    print("-" * 78)

    rows = []
    for q in QUERIES:
        out = run_wisewell_query(q, retriever=r, top_k=8, retrieve_pool=24, debug=True)
        fs = out.get("fork_signals") or {}
        top = fs.get("top_score")
        pmids = fs.get("distinct_pmids")
        rows.append((q, out["decision"], out["mode"], top, pmids))
        top_s = f"{top:6.3f}" if top is not None else "   -  "
        print(f"{out['decision']:9s} {out['mode']:8s} {top_s} {str(pmids):>5s}  {q}")

    forked = [r for r in rows if r[3] is not None]
    rag = [r for r in forked if r[2] == "rag"]
    gen = [r for r in forked if r[2] == "general"]
    band = [r for r in forked if r[3] is not None and 0.54 <= r[3] < RAG_MIN_TOP_SCORE]
    clarify = [r for r in rows if r[4] is None]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  total={len(rows)}  rag={len(rag)}  general={len(gen)}  clarify/other={len(clarify)}")
    print(f"  in the 0.54-{RAG_MIN_TOP_SCORE} band (just under the bar): {len(band)}")

    if band:
        print("\n  Borderline queries (0.54-under-bar) + expansion probe:")
        for q, dec, mode, top, pmids in band:
            exp = expanded_top_score(q)
            verdict = "would clear" if exp >= RAG_MIN_TOP_SCORE else "still under"
            print(f"    {top:6.3f} -> expanded {exp:6.3f} [{verdict}] | {q}")

    # Also probe any general-mode query below 0.54 to see if it's genuinely weak.
    deep = [r for r in gen if r[3] is not None and r[3] < 0.54]
    if deep:
        print("\n  Well-under-bar general queries (<0.54):")
        for q, dec, mode, top, pmids in deep:
            print(f"    {top:6.3f} | {q}")

    print("\nRecommendation inputs:")
    print("  - If borderline band is mostly legit educational queries that 'would clear'")
    print("    via expansion -> add light concept-query expansion (keep 0.58).")
    print("  - If they cluster and expansion doesn't help -> lower the bar.")
    print("  - If only odd phrasings fall under -> keep 0.58 as-is.")


if __name__ == "__main__":
    main()

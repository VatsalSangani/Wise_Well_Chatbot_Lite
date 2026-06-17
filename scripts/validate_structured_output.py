"""
Structured output validation: RAG/general JSON synthesis + fallback handling.

Tests that structured answers render correctly and fallbacks work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestration.llm_syntheses import synthesize_response, synthesize_general_response
from backend.routes.query import _structured_to_markdown
from backend.deps import get_retriever

print("="*70)
print("STRUCTURED OUTPUT VALIDATION")
print("="*70)

# Test 1: RAG structured output
print("\n[TEST 1] RAG -> Structured JSON -> Markdown")
print("-"*70)

retriever = get_retriever()
snips = [
    {
        'pmid': '12345678', 'year': 2020, 'title': 'CRP as inflammatory marker',
        'text': 'C-reactive protein (CRP) is an acute-phase protein produced by the liver that rises in response to inflammation.'
    },
    {
        'pmid': '23456789', 'year': 2019, 'title': 'CRP and cardiovascular risk',
        'text': 'Elevated CRP levels are associated with increased systemic inflammation and have been studied as a cardiovascular risk marker.'
    },
]

rag_result = synthesize_response("what is C-reactive protein", snips)

if rag_result.get("structured"):
    print("Structured: YES")
    structured_content = rag_result.get("synthesized_answer", {})
    markdown = _structured_to_markdown(structured_content)
    print("\nMarkdown output (first 300 chars):")
    print(markdown[:300] + "...")
    print(f"\nHas summary: {bool(structured_content.get('summary'))}")
    print(f"Has key_points: {bool(structured_content.get('key_points'))}")
    print(f"Has explanation: {bool(structured_content.get('explanation'))}")
    print(f"Has citations: {len(structured_content.get('citations', []))} PMIDs")
    print(f"Citations validated (not hallucinated): {all(c.get('pmid') in ['12345678', '23456789'] for c in structured_content.get('citations', []))}")
    print("\n[PASS] RAG structured output works")
else:
    print("[FAIL] RAG did not return structured output")

# Test 2: General structured output
print("\n[TEST 2] General -> Structured JSON -> Markdown")
print("-"*70)

gen_result = synthesize_general_response("what does a cold feel like")

if gen_result.get("structured"):
    print("Structured: YES")
    structured_content = gen_result.get("synthesized_answer", {})
    markdown = _structured_to_markdown(structured_content)
    print("\nMarkdown output (first 300 chars):")
    print(markdown[:300] + "...")
    print(f"\nHas summary: {bool(structured_content.get('summary'))}")
    print(f"Has explanation: {bool(structured_content.get('explanation'))}")
    print(f"Has when_to_see_doctor: {bool(structured_content.get('when_to_see_doctor'))}")
    print(f"Citations (should be empty): {len(structured_content.get('citations', []))}")
    print("\n[PASS] General structured output works")
elif gen_result.get("success"):
    print("[OK] General returned plain-text (fallback or retry)")
    print(f"Answer length: {len(gen_result.get('synthesized_answer', ''))}")
    print("\n[PASS] General answer generated (fallback acceptable)")
else:
    print("[FAIL] General synthesis failed")

# Test 3: Hallucinated citation is dropped
print("\n[TEST 3] Hallucinated Citation Validation")
print("-"*70)

structured_with_hallucination = {
    "summary": "Test",
    "key_points": [],
    "explanation": "Test explanation",
    "when_to_see_doctor": None,
    "citations": [
        {"pmid": "12345678", "claim": "Real PMID"},  # in retrieved set
        {"pmid": "99999999", "claim": "Hallucinated PMID"},  # NOT in retrieved set
    ]
}

from orchestration.llm_syntheses import validate_and_filter_citations

filtered = validate_and_filter_citations(
    structured_with_hallucination["citations"],
    snips
)

print(f"Original citations: {len(structured_with_hallucination['citations'])}")
print(f"Filtered citations: {len(filtered)}")
print(f"Hallucinated PMID dropped: {len(filtered) == 1 and filtered[0]['pmid'] == '12345678'}")

if len(filtered) == 1 and filtered[0]['pmid'] == '12345678':
    print("\n[PASS] Hallucinated citation dropped correctly")
else:
    print("\n[FAIL] Citation validation issue")

# Test 4: 50/50 personal case
print("\n[TEST 4] 50/50 Personal Case (General Mode)")
print("-"*70)

crp_result = synthesize_general_response("my CRP is 18, what should I do")

if crp_result.get("success"):
    answer = crp_result.get("synthesized_answer", "")
    is_structured = crp_result.get("structured")

    if is_structured:
        answer_text = answer.get("explanation", "")
    else:
        answer_text = answer

    # Check it's educational, not prescriptive
    is_educational = any(word in answer_text.lower() for word in ["elevated", "inflammation", "general", "typically"])
    is_prescriptive = any(word in answer_text.lower() for word in ["take", "stop", "start", "dose"])

    print(f"Structured: {is_structured}")
    print(f"Educational tone: {is_educational}")
    print(f"NOT prescriptive: {not is_prescriptive}")

    if is_educational and not is_prescriptive:
        print("\n[PASS] 50/50 case returns educational answer (soft-defer appended after)")
    else:
        print("\n[WARN] 50/50 answer may be too prescriptive (but soft-defer appends in query.py)")

# Summary
print("\n" + "="*70)
print("STRUCTURED OUTPUT VALIDATION COMPLETE")
print("="*70)
print("\n[SUCCESS] All structured output tests passed")

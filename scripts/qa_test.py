# scripts/qa_test.py
# Rigorous tests for Topic Consistency v2
# Enforces real-world failure modes: unit noise, verb noise, topic drift,
# pool_k enforcement, mixed-topic dominance, and small-pool behavior.

import sys
from pathlib import Path

# Add repo root to path for imports
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from guardrails.topic_consistency import apply_topic_consistency


def _assert_no_gout_chunks(chunks):
    gout_terms = ("gout", "urate", "colchicine", "uric")
    for c in chunks:
        text = (c.get("title", "") + " " + c.get("text", "")).lower()
        assert not any(t in text for t in gout_terms), \
            f"Gout content leaked: {c.get('chunk_id')}"


# ---------------------------------------------------------
# TEST 1: CRP LAB QUERY — NO UNIT / VERB ANCHORS
# ---------------------------------------------------------
def test_1_crp_lab_query_no_unit_verb_anchors():
    query = "CRP 18 mg/L — what does that indicate?"

    retrieved = [
        # Relevant CRP
        {"chunk_id": "rel_1", "pmid": "10000", "title": "CRP in Inflammation",
         "text": "C-reactive protein (CRP) is elevated in inflammation.", "score": 0.95},
        {"chunk_id": "rel_2", "pmid": "10001", "title": "CRP Biomarker",
         "text": "CRP levels rise during acute inflammation.", "score": 0.93},
        {"chunk_id": "rel_3", "pmid": "10002", "title": "CRP and RA",
         "text": "CRP is used to assess disease activity.", "score": 0.91},
        {"chunk_id": "rel_4", "pmid": "10003", "title": "Inflammatory Markers",
         "text": "CRP is a systemic inflammation marker.", "score": 0.89},

        # Off-topic gout (high score)
        {"chunk_id": "off_1", "pmid": "20000", "title": "Gout Overview",
         "text": "Gout is caused by monosodium urate crystals.", "score": 0.88},
        {"chunk_id": "off_2", "pmid": "20001", "title": "Urate Metabolism",
         "text": "Urate accumulation leads to gout.", "score": 0.86},

        # Generic filler
        {"chunk_id": "gen_1", "pmid": "30000", "title": "General Medicine",
         "text": "Clinical assessment requires careful evaluation.", "score": 0.70},
        {"chunk_id": "gen_2", "pmid": "30001", "title": "Diagnostics",
         "text": "Laboratory testing supports diagnosis.", "score": 0.65},
    ]

    td = apply_topic_consistency(
        query,
        retrieved,
        top_k=6,
        pool_k=8,
        min_avg_cov=0.15,
        min_good_chunks=2,
    )

    # Tier0 anchors must contain CRP
    assert "crp" in td.signals.get("tier0_anchors", []), \
        "CRP missing from tier0 anchors"

    # Units / verbs must never appear as anchors
    for bad in ("mg", "l", "dl", "indicate"):
        assert bad not in td.signals.get("tier0_anchors", []), \
            f"Noise token '{bad}' leaked into tier0 anchors"

    # Behavior-based assertion: no gout chunks survive
    _assert_no_gout_chunks(td.filtered)

    # Must not fail due to missing anchors
    assert td.reason in ("pass", "topic_insufficient")

    print("✓ TEST 1 PASSED")


# ---------------------------------------------------------
# TEST 2: IL-6 + RA — EXCLUDE GOUT AND MIXED DOMINANCE
# ---------------------------------------------------------
def test_2_il6_ra_excludes_gout_and_mixed():
    query = "What do IL-6 inhibitors do in rheumatoid arthritis?"

    retrieved = [
        # Relevant RA / IL-6
        {"chunk_id": "ra_1", "pmid": "100", "title": "IL-6 in RA",
         "text": "IL-6 inhibitors such as tocilizumab are effective in rheumatoid arthritis.", "score": 0.96},
        {"chunk_id": "ra_2", "pmid": "101", "title": "RA Cytokines",
         "text": "Rheumatoid arthritis involves IL-6 driven inflammation.", "score": 0.94},
        {"chunk_id": "ra_3", "pmid": "102", "title": "Biologics in RA",
         "text": "IL-6 blockade improves outcomes in RA.", "score": 0.92},
        {"chunk_id": "ra_4", "pmid": "103", "title": "RA Therapy",
         "text": "IL-6 inhibitors are standard RA therapy.", "score": 0.90},

        # Mixed-topic chunk (RA mentioned once, gout dominates)
        {"chunk_id": "mixed_1", "pmid": "999", "title": "Inflammatory Arthritis",
         "text": (
             "Rheumatoid arthritis is an inflammatory disease. "
             "However, gout is caused by monosodium urate crystals and "
             "urate-lowering therapy such as allopurinol is essential."
         ),
         "score": 0.89},

        # Pure gout (high score)
        {"chunk_id": "gout_1", "pmid": "200", "title": "Gout Pathophysiology",
         "text": "Gout is driven by urate crystal deposition.", "score": 0.88},
        {"chunk_id": "gout_2", "pmid": "201", "title": "Colchicine",
         "text": "Colchicine treats acute gout.", "score": 0.86},
    ]

    td = apply_topic_consistency(
        query,
        retrieved,
        top_k=6,
        pool_k=6,
        min_avg_cov=0.15,
        min_good_chunks=2,
    )

    final_ids = [c["chunk_id"] for c in td.filtered]

    # Must contain RA chunks
    assert sum(cid.startswith("ra_") for cid in final_ids) >= 3, \
        f"Too few RA chunks: {final_ids}"

    # Must exclude gout AND mixed-dominant chunk
    assert "mixed_1" not in final_ids, "Mixed-topic chunk leaked"
    _assert_no_gout_chunks(td.filtered)

    print("✓ TEST 2 PASSED")


# ---------------------------------------------------------
# TEST 3: pool_k STRICTLY ENFORCED
# ---------------------------------------------------------
def test_3_pool_k_enforced():
    query = "myocardial infarction treatment"

    retrieved = []

    # Top 32 off-topic
    for i in range(32):
        retrieved.append({
            "chunk_id": f"off_{i}",
            "pmid": str(1000 + i),
            "title": "Generic Topic",
            "text": "General medical information.",
            "score": 0.99 - i * 0.001,
        })

    # Relevant beyond pool_k
    for i in range(32, 80):
        retrieved.append({
            "chunk_id": f"rel_{i}",
            "pmid": str(2000 + i),
            "title": "Myocardial Infarction",
            "text": "Myocardial infarction is treated with reperfusion.",
            "score": 0.50,
        })

    td = apply_topic_consistency(
        query,
        retrieved,
        top_k=8,
        pool_k=32,
        min_avg_cov=0.10,
        min_good_chunks=1,
    )

    final_ids = [c["chunk_id"] for c in td.filtered]

    # No relevant chunks allowed (they are outside pool_k)
    assert not any(cid.startswith("rel_") for cid in final_ids), \
        f"pool_k violated: {final_ids}"

    # Enforce boundary by score
    boundary_score = retrieved[31]["score"]
    for c in td.filtered:
        assert c["score"] >= boundary_score, \
            "Chunk below pool_k boundary leaked"

    print("✓ TEST 3 PASSED")


# ---------------------------------------------------------
# TEST 4: SMALL POOL MUST NOT OVER-FILTER WHEN TIER0 IS EMPTY
# ---------------------------------------------------------
def test_4_small_pool_no_overfilter_when_tier0_empty():
    """
    Diabetes query typically has no Tier0 anchors if Tier0=biomarker_tokens only.
    In that case, entity filtering must be DISABLED, otherwise relevant chunks
    that don't mention the picked anchor get dropped.
    """
    query = "diabetes management insulin"

    retrieved = [
        {"chunk_id": "d1", "pmid": "400", "title": "Diabetes",
         "text": "Diabetes requires insulin therapy.", "score": 0.95},
        {"chunk_id": "d2", "pmid": "401", "title": "Insulin Therapy",
         "text": "Insulin is essential in diabetes management.", "score": 0.93},
        {"chunk_id": "d3", "pmid": "402", "title": "Glucose Control",
         "text": "Insulin regulates glucose metabolism.", "score": 0.91},
        {"chunk_id": "d4", "pmid": "403", "title": "Endocrinology",
         "text": "Insulin deficiency causes hyperglycemia.", "score": 0.89},
        {"chunk_id": "d5", "pmid": "404", "title": "Metabolism",
         "text": "Insulin signaling controls metabolism.", "score": 0.87},
        {"chunk_id": "d6", "pmid": "405", "title": "Pancreas",
         "text": "The pancreas produces insulin.", "score": 0.85},
    ]

    td = apply_topic_consistency(
        query,
        retrieved,
        top_k=6,
        pool_k=6,
        min_entity_keep=5,
        min_avg_cov=0.05,      # keep permissive so it should pass if not over-filtered
        min_good_chunks=1,
        good_chunk_threshold=0.10,
        topic_cov_threshold=0.20,
    )

    # Entity filter must NOT be applied when tier0 anchors are empty
    assert td.signals.get("entity_filter_applied") is False, \
        f"Entity filter should be disabled without Tier0 anchors. signals={td.signals}"

    # Must not drop chunks due to entity filtering
    assert len(td.filtered) == 6, \
        f"Should retain all 6 chunks when entity filter is disabled. got={len(td.filtered)}"

    # With permissive thresholds, this should pass
    assert td.ok is True, \
        f"Small pool should pass when not over-filtered. ok={td.ok}, reason={td.reason}, signals={td.signals}"

    print("✓ TEST 4 PASSED")

def test_5_crp_ra_prefers_ra_over_generic_inflammation_and_sepsis():
    """
    TEST 5: High-coverage but wrong disease should not outrank on-topic RA chunks.

    Query: "CRP in rheumatoid arthritis"
    We include:
      - RA+CRP chunks (desired)
      - Sepsis+CRP chunks (high CRP overlap but wrong disease)
      - Generic CRP inflammation chunks

    Assertions:
      - Top results must be RA+CRP-heavy
      - Sepsis chunk must not appear in top 3
    """
    query = "CRP in rheumatoid arthritis"

    retrieved = [
        # On-topic RA + CRP (should dominate)
        {"chunk_id": "ra_crp_1", "pmid": "5001", "title": "CRP and Rheumatoid Arthritis",
         "text": "In rheumatoid arthritis, CRP is commonly used to assess disease activity and inflammation.",
         "score": 0.92},
        {"chunk_id": "ra_crp_2", "pmid": "5002", "title": "RA Disease Activity Biomarkers",
         "text": "Rheumatoid arthritis disease activity correlates with CRP and ESR in multiple cohorts.",
         "score": 0.90},
        {"chunk_id": "ra_crp_3", "pmid": "5003", "title": "CRP Monitoring in RA",
         "text": "CRP monitoring supports rheumatoid arthritis management and treatment response evaluation.",
         "score": 0.88},

        # Wrong-topic but high lexical overlap: sepsis + CRP
        {"chunk_id": "sepsis_crp_1", "pmid": "6001", "title": "CRP in Sepsis",
         "text": "C-reactive protein (CRP) is elevated in sepsis and is used as an inflammatory marker in infection.",
         "score": 0.93},  # intentionally high score to simulate retriever bias

        # Generic CRP inflammation chunks
        {"chunk_id": "gen_crp_1", "pmid": "7001", "title": "CRP in Inflammation",
         "text": "CRP is an acute phase reactant elevated in systemic inflammation.",
         "score": 0.89},
        {"chunk_id": "gen_crp_2", "pmid": "7002", "title": "Acute Phase Proteins",
         "text": "CRP rises during inflammation and infection.",
         "score": 0.87},

        # Filler
        {"chunk_id": "gen_1", "pmid": "7100", "title": "General Medicine",
         "text": "Clinical context matters in interpreting biomarkers.",
         "score": 0.70},
    ]

    td = apply_topic_consistency(
        query,
        retrieved,
        top_k=5,
        pool_k=7,
        min_avg_cov=0.10,
        min_good_chunks=1,
        good_chunk_threshold=0.15,
        topic_cov_threshold=0.20,
    )

    final_ids = [c["chunk_id"] for c in td.filtered]
    top3 = final_ids[:3]

    # Must have at least 2 RA chunks in top 3
    ra_top3 = sum(cid.startswith("ra_crp_") for cid in top3)
    assert ra_top3 >= 2, f"Expected >=2 RA chunks in top3, got top3={top3}"

    # Sepsis chunk must NOT appear in top 3 (high overlap but wrong disease)
    assert "sepsis_crp_1" not in top3, f"Sepsis chunk leaked into top3={top3}"

    print("✓ TEST 5 PASSED")



if __name__ == "__main__":
    test_1_crp_lab_query_no_unit_verb_anchors()
    test_2_il6_ra_excludes_gout_and_mixed()
    test_3_pool_k_enforced()
    test_4_small_pool_no_overfilter_when_tier0_empty()
    test_5_crp_ra_prefers_ra_over_generic_inflammation_and_sepsis()
    print("\n✓✓✓ ALL RIGOROUS TESTS PASSED ✓✓✓")

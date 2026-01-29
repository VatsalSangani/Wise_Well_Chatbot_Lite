# scripts/guardrails/topic_consistency.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import re
import logging

from guardrails.config_loader import load_guardrails_config

CFG = load_guardrails_config()
logger = logging.getLogger(__name__)

STOPWORDS = set(CFG.get("stopwords", []))
GENERIC_TOKENS = set(CFG.get("generic_tokens", []))
SYNONYMS: Dict[str, List[str]] = CFG.get("synonyms", {})


@dataclass
class TopicDecision:
    ok: bool
    reason: str
    signals: Dict[str, Any]
    filtered: List[Dict[str, Any]]


def _normalize(s: str) -> str:
    return (s or "").lower()


def _tokenize_query_preserve_order(query: str) -> List[str]:
    # Keeps hyphenated biomedical tokens like IL-6, TNF-alpha.
    raw = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", query or "")
    toks: List[str] = []
    for t in raw:
        tl = t.lower()
        if tl in STOPWORDS:
            continue
        # keep uppercase acronyms like "CRP"
        if len(tl) < 3 and not t.isupper():
            continue
        toks.append(tl)
    return toks


def _expand_synonyms(tokens: List[str]) -> List[str]:
    out = set(tokens)
    for t in list(tokens):
        if t in SYNONYMS:
            for v in SYNONYMS[t]:
                out.add(v.lower())

        for k, vals in SYNONYMS.items():
            vals_low = [x.lower() for x in vals]
            if t in vals_low:
                out.add(k.lower())
                for v in vals_low:
                    out.add(v)
    return sorted(out)


def _is_specific_token(tok: str) -> bool:
    if tok in GENERIC_TOKENS:
        return False
    if "-" in tok:
        return True
    if any(ch.isdigit() for ch in tok):
        return True
    return len(tok) >= 5


def _has_token(text: str, tok: str) -> bool:
    return re.search(rf"\b{re.escape(tok)}\b", text) is not None


def _has_phrase(text: str, phrase: str) -> bool:
    p = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.search(rf"\b{p}\b", text) is not None


def _extract_anchors_v2(
    query: str,
    biomarker_tokens: set,
    unit_noise_tokens: set,
    verb_noise_tokens: set,
    anchor_synonyms: dict,
) -> Dict[str, Any]:
    """
    Extract Tier0 anchors: biomarker_tokens only (current design).
    Remove unit noise, verb noise, and pure numbers.
    Apply anchor_synonyms normalization consistently.

    NOTE: If you later expand Tier0 beyond biomarkers (diseases/drugs),
    change tier0 membership here (not elsewhere).
    """
    # Normalize anchor_synonyms keys/values to lowercase (ignore non-str entries safely)
    anchor_synonyms_lower: Dict[str, str] = {}
    for k, v in (anchor_synonyms or {}).items():
        if isinstance(k, str) and isinstance(v, str):
            anchor_synonyms_lower[k.lower()] = v.lower()

    ordered = _tokenize_query_preserve_order(query)

    # Apply anchor synonyms to ordered tokens
    normalized_ordered: List[str] = []
    for t in ordered:
        normalized_ordered.append(anchor_synonyms_lower.get(t, t))

    uniq = sorted(set(normalized_ordered))

    # Tier0 anchors = biomarkers (after normalization), excluding noise
    tier0_anchors: List[str] = []
    for t in uniq:
        if t.isdigit():
            continue
        if t in unit_noise_tokens:
            continue
        if t in verb_noise_tokens:
            continue
        if t in biomarker_tokens:
            tier0_anchors.append(t)

    # Expand synonyms (legacy SYNONYMS dict) from normalized tokens
    uniq_expanded = _expand_synonyms(uniq)

    # Specific tokens excluding noise/verbs/units/numbers
    tokens_specific = [
        t for t in uniq_expanded
        if _is_specific_token(t)
        and t not in unit_noise_tokens
        and t not in verb_noise_tokens
        and not t.isdigit()
    ]
    tokens_generic = [t for t in uniq_expanded if t not in tokens_specific]

    # Phrases from normalized_ordered (keeps order)
    phrases = set()
    for n in (2, 3, 4):
        for i in range(0, max(0, len(normalized_ordered) - n + 1)):
            gram = normalized_ordered[i : i + n]
            if any(g in STOPWORDS for g in gram):
                continue
            phrase = " ".join(gram)
            if any(st in phrase for st in tokens_specific):
                phrases.add(phrase)

    return {
        "tier0_anchors": tier0_anchors,
        "tokens_specific": tokens_specific,
        "tokens_generic": tokens_generic,
        "phrases": sorted(phrases),
    }


def _coverage_v2(
    anchors: Dict[str, Any],
    chunk: Dict[str, Any],
    topic_cov_threshold: float = 0.34,
) -> Tuple[float, Dict[str, Any]]:
    """
    Coverage v2: Pass if chunk matches at least 1 Tier0 anchor
    OR reaches topic_cov threshold based on tokens_specific/phrases.
    """
    text = _normalize(chunk.get("title", "")) + " " + _normalize(chunk.get("text", ""))

    tier0_anchors = anchors.get("tier0_anchors", [])
    sp_tokens = anchors.get("tokens_specific", [])
    phrases = anchors.get("phrases", [])

    matched_anchors = [a for a in tier0_anchors if a and _has_token(text, a)]
    token_hits = [t for t in sp_tokens if t and _has_token(text, t)]
    phrase_hits = [p for p in phrases if p and _has_phrase(text, p)]

    total = (3 * len(phrases)) + (2 * len(sp_tokens))
    hit_score = (3 * len(phrase_hits)) + (2 * len(token_hits))
    cov = hit_score / max(1, total)

    pass_criteria = bool(matched_anchors) or cov >= topic_cov_threshold

    if pass_criteria:
        fail_reason = None
    elif matched_anchors:
        fail_reason = "low_coverage"
    else:
        fail_reason = "no_anchor_match"

    dbg = {
        "matched_anchors": matched_anchors[:5],
        "token_hits": token_hits[:10],
        "phrase_hits": phrase_hits[:10],
        "topic_cov": cov,
        "passed": pass_criteria,
        "fail_reason": fail_reason,
    }
    return cov, dbg


def _pick_entity_anchors(anchors: Dict[str, Any]) -> List[str]:
    """
    Pick 1–2 strongest entity anchors to prevent topic drift.
    Prefer Tier0 anchors first; fallback to strong phrases if no Tier0 anchors.
    """
    tier0_anchors = anchors.get("tier0_anchors", [])
    phrases = anchors.get("phrases", [])

    picks: List[str] = []
    if tier0_anchors:
        picks.extend(tier0_anchors[:2])

    if not picks and phrases:
        phrases_sorted = sorted(phrases, key=len, reverse=True)
        for p in phrases_sorted:
            if p:
                picks.append(p)
                break

    return picks[:2]


def _entity_match(text: str, anchor: str) -> bool:
    anchor = (anchor or "").strip()
    if not anchor:
        return False
    if " " in anchor:
        return _has_phrase(text, anchor)
    return _has_token(text, anchor)


def apply_topic_consistency(
    query: str,
    retrieved: List[Dict[str, Any]],
    *,
    top_k: int,
    pool_k: int = 32,
    min_avg_cov: float = 0.18,
    min_good_chunks: int = 4,
    good_chunk_threshold: float = 0.20,
    topic_cov_threshold: float = 0.34,
    require_entity_filter: bool = True,
    min_entity_keep: int = 5,
) -> TopicDecision:
    """
    Topic Consistency v2:
    - Build explicit pool from top pool_k by retriever score
    - Score/filter within pool
    - Apply entity filter only when Tier0 anchors exist (strong anchors)
    - Do not refill from entity_dropped when entity filtering is applied
    """
    if not retrieved:
        return TopicDecision(False, "no_candidates", {"top_k": top_k, "pool_k": 0}, [])

    # Build explicit pool: top pool_k by retriever score
    sorted_by_score = sorted(retrieved, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    pool = sorted_by_score[: max(1, int(pool_k))]
    actual_pool_size = len(pool)

    # Config tokens
    BIOMARKER_TOKENS = set(CFG.get("biomarker_tokens", []))
    UNIT_NOISE_TOKENS = set(CFG.get("unit_noise_tokens", []))
    VERB_NOISE_TOKENS = set(CFG.get("verb_noise_tokens", []))
    ANCHOR_SYNONYMS = CFG.get("anchor_synonyms", {}) or {}

    anchors = _extract_anchors_v2(
        query,
        BIOMARKER_TOKENS,
        UNIT_NOISE_TOKENS,
        VERB_NOISE_TOKENS,
        ANCHOR_SYNONYMS,
    )

    # If nothing anchor-like, return pool slice without filtering (old behavior)
    if len(anchors.get("tier0_anchors", [])) == 0 and len(anchors.get("phrases", [])) == 0:
        return TopicDecision(
            True,
            "no_specific_anchors",
            {
                "top_k": top_k,
                "pool_k": actual_pool_size,
                "tier0_anchors": [],
                "phrases": [],
                "entity_filter_applied": False,
                "entity_kept_count": 0,
                "required_keep": 0,
            },
            pool[:top_k],
        )

    # Score candidates in pool
    scored: List[Tuple[float, float, Dict[str, Any], Dict[str, Any]]] = []
    for c in pool:
        cov, det = _coverage_v2(anchors, c, topic_cov_threshold)
        scored.append((cov, float(c.get("score", 0.0)), det, c))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    ranked: List[Dict[str, Any]] = []
    for cov, sc, det, c in scored:
        cc = dict(c)
        cc["topic_cov"] = cov
        cc["topic_cov_debug"] = det
        ranked.append(cc)

    # --- Option A: disable entity filtering if Tier0 anchors are empty ---
    tier0 = anchors.get("tier0_anchors", [])
    if require_entity_filter and not tier0:
        require_entity_filter = False

    entity_anchors = _pick_entity_anchors(anchors)
    entity_kept: List[Dict[str, Any]] = []
    entity_dropped: List[Dict[str, Any]] = []

    entity_filter_applied = bool(require_entity_filter and entity_anchors and tier0)

    if entity_filter_applied:
        for c in ranked:
            text = (c.get("title", "") + " " + c.get("text", "")).lower()
            if any(_entity_match(text, a) for a in entity_anchors):
                entity_kept.append(c)
            else:
                entity_dropped.append(c)
    else:
        entity_kept = ranked[:]
        entity_dropped = []

    # Final = kept only when entity filter applied; otherwise can include all ranked
    final: List[Dict[str, Any]] = entity_kept[:top_k]
    # Enforce output length
    if len(final) > top_k:
        final = final[:top_k]

    avg_cov = sum(x.get("topic_cov", 0.0) for x in final) / max(1, len(final))
    good_chunks = sum(1 for x in final if x.get("topic_cov", 0.0) >= good_chunk_threshold)

    # adaptive required_keep based on evaluated pool (ranked)
    required_keep = min(int(min_entity_keep), len(ranked)) if ranked else 0

    ok = (avg_cov >= min_avg_cov or good_chunks >= min_good_chunks) and (len(entity_kept) >= required_keep)
    reason = "pass" if ok else "topic_insufficient"

    return TopicDecision(
        ok,
        reason,
        {
            "top_k": top_k,
            "pool_k": actual_pool_size,
            "avg_cov": avg_cov,
            "good_chunks": good_chunks,
            "required_keep": required_keep,
            "entity_kept_count": len(entity_kept),
            "entity_filter_applied": entity_filter_applied,
            "entity_anchors": entity_anchors,
            "tier0_anchors": tier0[:10],
            "phrases": anchors.get("phrases", [])[:5],
        },
        final,
    )

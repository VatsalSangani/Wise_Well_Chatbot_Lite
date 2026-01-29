# ==================================
# scripts/guardrails/query_specificity.py
# ==================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from guardrails.config_loader import load_guardrails_config

CFG = load_guardrails_config()

STOPWORDS = set(CFG.get("stopwords", []))
GENERIC_TOKENS = set(CFG.get("generic_tokens", []))

UNDERSPECIFIED_PATTERNS: List[str] = CFG.get("underspecified_patterns", [])
UNIT_PATTERNS: List[str] = CFG.get("unit_patterns", [])
DRUG_SUFFIX_PATTERNS: List[str] = CFG.get("drug_suffix_patterns", [])
BIOMARKER_TOKENS: List[str] = [x.lower() for x in CFG.get("biomarker_tokens", [])]

# Optional config extension (safe defaults if not present):
# You can add these keys in config later; code works without them.
DRUG_TOKENS: List[str] = [x.lower() for x in CFG.get("drug_tokens", [])]  # optional
THERAPY_TOKENS: List[str] = [x.lower() for x in CFG.get("therapy_tokens", [])]  # optional

MIN_SCORE = int(CFG.get("specificity_min_score", 2))


# Small hardcoded whitelist (covers common meds + your known failure case).
# This is intentionally small—expand over time if needed.
# If you add "drug_tokens" in config, that list will be checked too.
DRUG_WHITELIST = {
    "metformin", "insulin", "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
    "warfarin", "heparin", "statin", "atorvastatin", "simvastatin",
    "prednisone", "dexamethasone",
    "adalimumab", "infliximab", "etanercept", "tocilizumab",
}

# Therapy / pharmacology hint terms (not advice; just specificity cues)
THERAPY_HINTS = {
    "inhibitor", "antagonist", "agonist", "blocker", "antibody",
    "therapy", "treatment", "dose", "dosing", "efficacy", "mechanism",
    "moa", "pharmacokinetics", "pharmacodynamics",
}


@dataclass
class SpecificityDecision:
    specific: bool
    reason: str
    signals: Dict[str, Any]
    clarification: Optional[Dict[str, Any]] = None


def _tokenize(q: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", q or "")
    out: List[str] = []
    for t in toks:
        tl = t.lower()
        if tl in STOPWORDS:
            continue
        out.append(tl)
    return out


def _matches_any(patterns: List[str], text: str) -> bool:
    t = (text or "").lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def _drug_like_token(toks: List[str], qlow: str) -> bool:
    """
    Drug-like detection should NOT rely only on suffixes.
    We use:
      1) suffix heuristics from config
      2) whitelist (hardcoded + optional config list)
      3) presence of pharmacology/therapy hints + at least one long token (to avoid false positives)
    """
    # 1) suffix heuristics
    for t in toks:
        for suf in DRUG_SUFFIX_PATTERNS:
            if re.search(suf, t):
                return True

    # 2) whitelist (hardcoded + config)
    drug_vocab = DRUG_WHITELIST.union(set(DRUG_TOKENS))
    if any(t in drug_vocab for t in toks):
        return True

    # 3) therapy hint heuristic
    # If query talks like pharmacology AND has at least one reasonably long token,
    # treat as drug-like enough to proceed to retrieval.
    hint_hit = any(h in qlow for h in THERAPY_HINTS) or any(h in qlow for h in THERAPY_TOKENS)
    long_token_hit = any(len(t) >= 7 for t in toks if t not in GENERIC_TOKENS)
    if hint_hit and long_token_hit:
        return True

    return False


def assess_query_specificity(query: str) -> SpecificityDecision:
    q = (query or "").strip()
    qlow = q.lower()
    toks = _tokenize(q)

    # Negative signal: known underspecified phrasing
    underspecified_hit = _matches_any(UNDERSPECIFIED_PATTERNS, qlow)

    # Positive: unit present (lab values)
    unit_hit = _matches_any(UNIT_PATTERNS, qlow)

    # Positive: biomarker token present
    biomarker_hit = any(re.search(rf"\b{re.escape(b)}\b", qlow) for b in BIOMARKER_TOKENS)

    # Positive: drug-like token present (IMPROVED)
    drug_like_hit = _drug_like_token(toks, qlow)

    # Positive: count of "specific" tokens (not generic, len>=5 or has hyphen or digits)
    def is_specific_token(t: str) -> bool:
        if t in GENERIC_TOKENS:
            return False
        if "-" in t:
            return True
        if any(ch.isdigit() for ch in t):
            return True
        return len(t) >= 5

    specific_tokens = [t for t in toks if is_specific_token(t)]
    specific_token_count = len(set(specific_tokens))

    # Slightly improved: treat common biomedical bigrams as specificity bump
    # (e.g., "type 2 diabetes", "rheumatoid arthritis", etc.)
    # This is deliberately conservative.
    disease_phrase_hit = bool(re.search(r"\btype\s*2\s*diabetes\b", qlow)) or bool(re.search(r"\brheumatoid\s*arthritis\b", qlow))

    # Basic scoring
    score = 0
    score += 2 if biomarker_hit else 0
    score += 2 if unit_hit else 0
    score += 2 if drug_like_hit else 0
    score += 1 if specific_token_count >= 2 else 0
    score += 1 if disease_phrase_hit else 0
    score -= 2 if underspecified_hit else 0
    score -= 1 if len(toks) < 5 else 0

    signals = {
        "score": score,
        "min_score": MIN_SCORE,
        "underspecified_hit": underspecified_hit,
        "unit_hit": unit_hit,
        "biomarker_hit": biomarker_hit,
        "drug_like_hit": drug_like_hit,
        "disease_phrase_hit": disease_phrase_hit,
        "specific_token_count": specific_token_count,
        "tokens": toks[:20],
        "specific_tokens": sorted(set(specific_tokens))[:20],
    }

    if score >= MIN_SCORE and not underspecified_hit:
        return SpecificityDecision(True, "specific_enough", signals, None)

    clarification = {
        "need": [
            "What specific biomarker/drug/condition are you asking about?",
            "What context should the answer focus on (mechanism, association with a disease, prognosis, monitoring, etc.)?",
        ],
        "optional": [
            "Any relevant population/setting (e.g., acute infection vs chronic inflammatory disease)?",
            "Any measurement values and units (if a lab test question)?",
        ],
        "examples": [
            "What is the mechanism of action of metformin in type 2 diabetes?",
            "What does a CRP of 18 mg/L indicate in infection vs autoimmune disease?",
        ],
    }

    return SpecificityDecision(False, "insufficient_context", signals, clarification)

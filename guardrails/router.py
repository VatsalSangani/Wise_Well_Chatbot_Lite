"""
guardrails/router.py

Front-door router for WiseWell. Runs at the very FRONT of qa_check.answer_query,
BEFORE input_validation (which rejects short greetings at min-10-chars) and
BEFORE specificity. It classifies each input into a lane:

  - CHITCHAT : greeting / thanks / meta ("hi", "thanks", "what can you do",
               "who are you") with NO medical content. Gets a friendly response;
               retrieval and ALL medical gates are skipped.
  - MEDICAL  : everything substantive. Proceeds into the existing pipeline
               (input_validation -> safety_intent -> retrieval -> gates -> answer).

The router classifies the LANE only. It does NOT bypass safety: the MEDICAL lane
still runs safety_intent (the hard-refuse for individual Dx/Rx lives there). The
router only bypasses the OVER-FIRING gates (input_validation min-length,
specificity) for the chitchat lane.

It also surfaces `is_personal` (the user is describing their own case) so the
answer layer can apply the 50/50 personal handling downstream. This mirrors the
personal-context signals in safety_intent (kept in sync deliberately).

Design stance: DEFAULT IS MEDICAL. Chitchat only fires on an explicit greeting/
meta match with no medical token — so a medical question can never be misrouted
into the no-safety chitchat lane.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from guardrails.config_loader import load_guardrails_config

CFG = load_guardrails_config()

# Medical-signal vocab (reused from guardrails config so there's one source).
_BIOMARKERS = [b.lower() for b in CFG.get("biomarker_tokens", [])]
_DRUG_SUFFIXES = CFG.get("drug_suffix_patterns", [])
_EVIDENCE_ANCHORS = [a.lower() for a in CFG.get("evidence_anchors", [])]
_UNIT_PATTERNS = [u.lower() for u in CFG.get("unit_patterns", [])]

# Small drug/health lexicon for common terms not covered by biomarker/anchor lists.
_DRUG_WHITELIST = {
    "metformin", "insulin", "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
    "warfarin", "heparin", "statin", "atorvastatin", "simvastatin", "prednisone",
    "dexamethasone", "adalimumab", "infliximab", "etanercept", "tocilizumab",
    "antibiotic", "antibiotics", "vaccine", "vitamin",
}
_HEALTH_TERMS = {
    "disease", "symptom", "symptoms", "diagnosis", "diagnose", "infection",
    "cancer", "tumor", "tumour", "diabetes", "hypertension", "cholesterol",
    "blood", "pressure", "inflammation", "immune", "immunity", "arthritis",
    "thyroid", "kidney", "liver", "heart", "cardiac", "stroke", "fever",
    "medication", "medicine", "drug", "dose", "dosage", "treatment", "therapy",
    "diet", "exercise", "pregnancy", "microbiome", "metabolic", "syndrome",
}

# Chitchat vocab — a query made ENTIRELY of these (after normalization) is chitchat.
_GREETING_WORDS = {
    "hi", "hii", "hiya", "hey", "heya", "hello", "helloo", "yo", "howdy", "sup",
    "greetings", "gm", "morning", "evening", "afternoon",
}
_THANKS_WORDS = {"thanks", "thank", "thx", "ty", "cheers", "appreciate", "appreciated"}
_BYE_WORDS = {"bye", "goodbye", "cya", "later", "farewell"}
_SOCIAL_FILLER = {
    "you", "u", "there", "again", "so", "much", "lot", "lots", "very", "really",
    "ok", "okay", "kk", "cool", "nice", "great", "awesome", "good", "please",
    "wisewell", "bot", "assistant", "the", "a", "and", "well", "everyone",
    "it", "that", "this", "all", "for", "your", "help",  # 'help' alone handled by meta too
    "to", "me",
}
_CHITCHAT_VOCAB = _GREETING_WORDS | _THANKS_WORDS | _BYE_WORDS | _SOCIAL_FILLER

# Meta patterns — match the WHOLE normalized query (anchored). Substring-anchoring
# matters: "what can you do for high cholesterol" must NOT match (it's medical).
_META_PATTERNS = [
    r"what can (you|u) do",
    r"what (do|can) (you|u) help( me)?( with)?",
    r"what (are|r) (you|u)( capable of)?",
    r"who (are|r) (you|u)",
    r"what is this( bot| service| app)?",
    r"what('?s| is) wisewell",
    r"how (do|does) (you|this|it) work",
    r"how (are|r) (you|u)( doing| today)?",
    r"are (you|u) (a )?(doctor|human|real|ai|a bot|bot|an ai)",
    r"can (you|u) help( me)?",
    r"good (morning|afternoon|evening|day)",
    r"nice to meet (you|u)",
    r"how('?s| is) it going",
    r"help",
    r"menu",
    r"start",
]
_META_RE = [re.compile(rf"^{p}$") for p in _META_PATTERNS]

# Personal-context signals — mirror of safety_intent.classify_intent's set.
_PERSONAL_PATTERNS = [
    r"\bi\s+(have|had|am|was|get|got|feel|felt|notice|noticed)\b",
    r"\bmy\s+\w+",
    r"\bi'?m\s+(taking|on|experiencing|having|feeling)\b",
    r"\bfor\s+me\b",
    r"\bshould\s+i\b",
    r"\bcan\s+i\b",
    r"\bdo\s+i\b",
]


@dataclass
class RouterDecision:
    lane: str  # "chitchat" | "medical"
    is_personal: bool
    signals: Dict[str, Any] = field(default_factory=dict)


def _normalize(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"[^\w\s'/%-]", " ", q)   # drop punctuation but keep / % - '
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _has_medical_signal(norm: str, tokens: List[str]) -> bool:
    tokset = set(tokens)
    if any(b in norm for b in _BIOMARKERS):
        return True
    if any(u in norm for u in _UNIT_PATTERNS):
        return True
    if tokset & _DRUG_WHITELIST or tokset & _HEALTH_TERMS:
        return True
    if any(a in tokset for a in _EVIDENCE_ANCHORS):
        return True
    for t in tokens:
        if any(re.search(suf, t) for suf in _DRUG_SUFFIXES) and len(t) >= 5:
            return True
    return False


def _is_chitchat(norm: str, tokens: List[str]) -> bool:
    if not norm:
        return False
    # A medical token anywhere => never chitchat (even "hi, what is CRP?").
    if _has_medical_signal(norm, tokens):
        return False
    # Whole-query meta match ("what can you do", "who are you", "help").
    if any(rx.match(norm) for rx in _META_RE):
        return True
    # Greeting-only: every token is greeting/thanks/bye/social filler.
    if tokens and all(t in _CHITCHAT_VOCAB for t in tokens):
        return True
    return False


def _is_personal(norm: str) -> bool:
    return any(re.search(p, norm) for p in _PERSONAL_PATTERNS)


def classify_lane(query: str) -> RouterDecision:
    """Classify a query into a front-door lane. Default is MEDICAL (safe)."""
    norm = _normalize(query)
    tokens = norm.split()

    medical_signal = _has_medical_signal(norm, tokens)
    chitchat = _is_chitchat(norm, tokens)
    personal = _is_personal(norm)

    lane = "chitchat" if chitchat else "medical"
    return RouterDecision(
        lane=lane,
        is_personal=personal and not chitchat,
        signals={
            "normalized": norm,
            "medical_signal": medical_signal,
            "meta_match": any(rx.match(norm) for rx in _META_RE),
            "token_count": len(tokens),
        },
    )

"""
guardrails/responses.py

Every code-inserted string WiseWell emits lives here, so the tone is authored
once and is identical every time. The LLM writes educational *content*; this
module guarantees the trust signals (disclaimer / defer / refusal / escalation)
are present and worded warmly.

Tone target: warm, plain-spoken, respectful, human — never cold or bureaucratic.
Hard constraint: warm must NOT become false reassurance. Never minimise a
concern, never soften away a needed "see a doctor", never downplay a serious
sign. Warm but honest, not soothing.

Region: UK-first (999, Samaritans 116 123), with 112 as the Europe-wide
fallback. Make region-aware later if US/other support is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ── Tier 2: physical emergency escalation (led-with, short, conditional) ──────
# {desc} is the matched red-flag description from red_flags.py.
def tier2_physical(desc: str) -> str:
    desc = desc or "a possible medical emergency"
    return (
        f"Please call 999 right now (or 112 anywhere in Europe). If you or someone with "
        f"you is experiencing {desc}, this can be a medical emergency and needs help "
        f"straight away — I can't assess it myself, so please don't wait, make the call."
    )


# ── Tier 2: suicide / self-harm (distinctly warm, non-clinical, NOT templated) ─
TIER2_SUICIDE: str = (
    "I'm really glad you told me, and I'm sorry you're carrying something this heavy "
    "right now. You don't have to get through it on your own. If you're in the UK, you "
    "can talk to Samaritans for free, any time of day or night, on 116 123 — or text "
    "SHOUT to 85258 if you'd rather not call. If you feel you might act on these thoughts, "
    "or you're in immediate danger, please call 999. You deserve support, and reaching "
    "out to someone who can be with you in this really can help."
)


# ── Tier 1: soft defer (default; calm — never "immediately") ──────────────────
TIER1_SOFT_DEFER: str = (
    "For your specific situation, a doctor who knows your history is the right person to "
    "give you personalised guidance."
)


# ── General mode: code-inserted disclaimer banner ────────────────────────────
GENERAL_MODE_DISCLAIMER: str = (
    "Just so you know, this is general medical knowledge rather than a specific study from "
    "my research database. For anything that applies to you personally, a healthcare "
    "professional is the best person to weigh in."
)


# ── Hard refuse (warm; offers the general alternative instead of a wall) ──────
def hard_refuse(topic: str | None = None) -> str:
    if topic:
        offer = f"I'm happy to explain how {topic} generally work, or what the research broadly says, if that would help."
    else:
        offer = "I'm happy to explain how things work in general terms, though, if that would help."
    return (
        "I can't advise on individual medication, dosing, or treatment decisions — those "
        "really need a doctor who knows your full history. " + offer
    )


# ── Clarify (ask ONE short question; topic/intent only, never history) ────────
CLARIFY_NO_ANCHOR: str = (
    "Could you tell me a bit more about what you'd like to know? I lost the thread — "
    "just point me at the topic and I'll help."
)

OFFER_TO_NARROW: str = (
    "Did you want a general overview, or were you asking about something more specific? "
    "Happy to go deeper either way."
)


# ── Chitchat lane (friendly, brief — not a formal capability dump) ───────────
def chitchat_response(query: str) -> str:
    q = (query or "").lower()
    if any(w in q for w in ("thank", "thx", "ty", "cheers", "appreciate")):
        return "You're very welcome! Ask me anything else about a health or medical topic whenever you like."
    if any(w in q for w in ("bye", "goodbye", "cya", "farewell", "take care")):
        return "Take care! Come back any time you have a health question."
    if any(w in q for w in ("doctor", "human", "real", "a bot", "an ai", " ai")):
        return (
            "I'm not a doctor — I'm an AI assistant that explains medical topics using published "
            "research. I can help you understand the general picture, but for anything about your "
            "own health, a healthcare professional is the right person to see. What would you like to know?"
        )
    if any(w in q for w in ("what can you", "what do you", "who are you", "what are you", "how do you work", "help", "what is this")):
        return (
            "I'm WiseWell, a medical information assistant. I can explain health and medical topics — "
            "what conditions, biomarkers, and treatments mean — using evidence from recent medical "
            "research. I can't give personal medical advice, but I can help you understand the general "
            "picture. What would you like to explore?"
        )
    # default greeting
    return (
        "Hi! I'm WiseWell — I help make sense of medical and health topics using research from recent "
        "studies. What would you like to know?"
    )


# ── RAG mode: code-built source block from retrieved metadata ────────────────
def build_source_block(snippets: List[Dict[str, Any]]) -> str:
    """Code-inserted citation block (not LLM-generated). De-dupes by PMID,
    preserves ranking order."""
    seen = set()
    lines: List[str] = []
    for s in snippets:
        pmid = s.get("pmid")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        title = (s.get("title") or "").strip()
        year = s.get("year")
        year_str = f" ({year})" if year else ""
        lines.append(f"- PMID {pmid}{year_str}: {title}")
    if not lines:
        return ""
    return "Sources (from PubMed):\n" + "\n".join(lines)

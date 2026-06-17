"""
guardrails/red_flags.py

Emergency red-flag detector (Tier 2 escalation). Runs EARLY — right after the
router, before normal answer composition — and can short-circuit the whole
RAG/general/50-50 flow: an active emergency does not wait for retrieval.

It is ADDITIVE to safety_intent, not a replacement. safety_intent refuses
individual Dx/Rx; this escalates active emergencies to emergency services /
crisis support. It is a router-to-help, NOT a treater — it never diagnoses or
manages the emergency, only points to immediate help.

THE CORE DISCRIMINATOR — asking about an emergency vs describing one happening:
  - "what is a heart attack" / "what are stroke symptoms"   -> educational, DO NOT escalate
  - "I think I'm having a heart attack" / "his face is
     drooping and he can't speak"                            -> active, ESCALATE

Rule: escalate = red_flag_symptom AND NOT educational_framing
                 AND (active_now OR NOT historical_framing)

The set is deliberately NARROW (genuine emergencies only). "I have a mild
headache" matches nothing here. Because the Tier 2 message is conditionally
phrased ("If you or someone with you is experiencing ..."), a rare borderline
escalation reads fine rather than alarming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Red-flag symptom categories (narrow, emergency-only) ──────────────────────
# (category, [symptom regexes], conditional description for the Tier 2 message)
_CATEGORIES: List[Tuple[str, List[str], str]] = [
    ("cardiac", [
        r"chest pain", r"chest pressure", r"chest tightness", r"tightness in (my |the )?chest",
        r"chest (is )?hurt", r"crushing (chest|pain)", r"heart attack",
        r"pain (spreading|radiating) (to|down|into) (my |the )?(arm|jaw|neck)",
        r"pain (in|down) (my |the )?(left )?arm and",
    ], "chest pain or pressure, or pain spreading to the arm or jaw"),
    ("breathing", [
        r"can'?t breathe", r"cannot breathe", r"can not breathe", r"cant breathe",
        r"trouble breathing", r"difficulty breathing", r"struggling to breathe",
        r"hard to breathe", r"short(ness)? of breath", r"gasping( for (air|breath))?",
        r"choking",
    ], "sudden trouble breathing"),
    ("stroke", [
        r"face (is )?drooping", r"drooping face", r"slurred speech", r"slurr(ing|ed)",
        r"can'?t speak", r"cant speak", r"can'?t talk", r"sudden(ly)? (weak|numb)",
        r"weakness (on|down) one side", r"numbness (on|down) one side",
        r"one side of (my|his|her|their) (face|body)", r"sudden severe headache",
        r"worst headache (of|ever|i)", r"\bstroke\b",
    ], "face drooping, slurred speech, or sudden weakness or numbness on one side"),
    ("anaphylaxis", [
        r"throat (is )?closing", r"throat closing", r"can'?t swallow",
        r"tongue (is )?swelling", r"swelling (of|in) (my |the )?(throat|tongue|lips|face)",
        r"severe allergic reaction", r"anaphyla(xis|ctic)", r"lips (are )?swelling",
    ], "throat tightening, swelling, or a severe allergic reaction"),
    ("unconscious_seizure", [
        r"passed out", r"pass(ing)? out", r"unconscious", r"fainted", r"collapsed",
        r"won'?t wake up", r"not waking up", r"unresponsive", r"\bseizure\b",
        r"convuls(ing|ions)", r"having a fit",
    ], "loss of consciousness or a seizure"),
    ("severe_bleeding", [
        r"severe bleeding", r"bleeding (heavily|badly|a lot)", r"won'?t stop bleeding",
        r"can'?t stop (the )?bleeding", r"bleeding won'?t stop", r"h(a)?emorrhage",
        r"losing a lot of blood", r"blood everywhere",
    ], "severe bleeding that won't stop"),
    ("overdose", [
        r"overdose", r"overdosed", r"took too many (pills|tablets)", r"too many pills",
        r"\bod'?d\b",
    ], "a possible overdose"),
    ("vision_loss", [
        r"sudden(ly)? (vision loss|loss of vision|blind)", r"lost (my )?vision",
        r"can'?t see (suddenly|anything)", r"sudden blindness",
    ], "sudden loss of vision"),
    ("severe_abdominal", [
        r"severe (abdominal|stomach|belly) pain", r"worst (stomach|abdominal) pain",
        r"excruciating (stomach|abdominal|belly)",
    ], "sudden severe abdominal pain"),
]

# Suicide / self-harm — handled first and with its own crisis message.
_SUICIDE_PATTERNS = [
    r"kill(ing)? myself", r"end(ing)? my life", r"take my (own )?life", r"\bsuicidal\b",
    r"\bsuicide\b", r"want(ing)? to die", r"wanna die", r"don'?t want to (live|be here|exist)",
    r"hurt(ing)? myself", r"harm(ing)? myself", r"self[ -]harm", r"end it all",
    r"no reason to live", r"better off dead", r"feel like dying",
]

# ── Framing discriminators ────────────────────────────────────────────────────
_EDUCATIONAL = [
    r"^\s*what (is|are|'?s|was|does|do|can)\b",
    r"\bwhat causes?\b",
    r"\b(symptoms?|signs?|causes?|definition|meaning|types?|risk factors?) of\b",
    r"\btell me about\b", r"\bexplain\b", r"\bdescribe\b", r"\bdefine\b",
    r"\bhow (do|can) (i|you|we) (recognize|identify|spot|tell)\b",
    r"\bwhat (should i|to) (look|watch) (for|out)\b",
    r"\bwhat does .* (feel|look) like\b",
]
# Bare "had" is NOT historical on its own ("she had a seizure" is likely acute);
# only an explicit past-time marker or recovery framing counts as historical.
_HISTORICAL = [
    r"\bhistory of\b", r"\bprevious(ly)?\b", r"\blast (year|month|week)\b",
    r"\b(years?|months?|weeks?|days?) ago\b", r"\brecovered\b", r"\bafter my\b",
    r"\bsurvived\b", r"\bsince my\b", r"\byesterday\b", r"\bused to (have|get)\b",
]
_ACTIVE_NOW = [
    r"\b(now|right now|currently|at the moment|happening)\b",
    r"\bis (drooping|happening|closing|swelling|bleeding)\b",
    r"\bcan'?t (breathe|speak|swallow|see|stop)\b",
    r"\b(i'?m|he'?s|she'?s|they'?re) having\b",
    r"\bthink (i'?m|i am|he'?s|she'?s|its|it'?s) (having|a)\b",
    r"\bhaving a\b",
]


@dataclass
class RedFlagDecision:
    escalate: bool
    category: Optional[str] = None          # "suicide" | "cardiac" | ... | None
    is_suicide: bool = False
    signals: Dict[str, Any] = field(default_factory=dict)


def _any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _match_category(text: str) -> Optional[Tuple[str, str]]:
    """Return (category, description) of the first matching red-flag, or None.
    Suicide is checked first and signalled via category == 'suicide'."""
    if _any(_SUICIDE_PATTERNS, text):
        return ("suicide", "")
    for cat, patterns, desc in _CATEGORIES:
        if _any(patterns, text):
            return (cat, desc)
    return None


def detect_red_flag(query: str) -> RedFlagDecision:
    """Decide whether a query is an ACTIVE emergency that must be escalated."""
    text = (query or "").lower().strip()
    if not text:
        return RedFlagDecision(False)

    matched = _match_category(text)
    if matched is None:
        return RedFlagDecision(False)

    category, desc = matched
    educational = _any(_EDUCATIONAL, text)
    historical = _any(_HISTORICAL, text)
    active_now = _any(_ACTIVE_NOW, text)

    # Educational framing never escalates ("what is a heart attack").
    # Historical framing suppresses unless there's an explicit active-now signal
    # ("had a heart attack last year" -> no; "had a stroke, his face is drooping
    # now" -> yes).
    escalate = (not educational) and (active_now or not historical)

    return RedFlagDecision(
        escalate=escalate,
        category=category if escalate else None,
        is_suicide=(category == "suicide" and escalate),
        signals={
            "matched_category": category,
            "educational": educational,
            "historical": historical,
            "active_now": active_now,
            "description": desc,
        },
    )

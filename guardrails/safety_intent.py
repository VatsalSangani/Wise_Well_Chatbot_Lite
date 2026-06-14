from __future__ import annotations

from typing import Tuple, Dict, Any
import re
import logging

from guardrails.config_loader import load_guardrails_config

CFG = load_guardrails_config()
logger = logging.getLogger(__name__)

REFUSE_PATTERNS = CFG.get("refuse_patterns", [])


def classify_intent(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Safety Intent v3 (NARROWED for the 50/50 personal handling).

    Hard-refuse ONLY individual diagnosis + individual treatment/medication/
    dosing decisions. General personal questions ("my CRP is 18, what does it
    mean / what should I generally do") are NO LONGER refused here — they flow
    on and get the 50/50 treatment (general answer + code-inserted doctor defer)
    in the answer layer. The "consult a doctor" suffix does not make individual
    prescribing/diagnosis safe, so those stay firmly refused.

    Returns (intent, debug_info)
      intent: "ok" | "refuse"
    """

    q = (query or "").strip().lower()
    if not q:
        return "ok", {
            "detected_personal_context": False,
            "detected_rx_intent": False,
            "decision": "allow",
            "reason": "empty_query",
        }

    # PERSONAL CONTEXT signals
    personal_signals = [
        r"\bi\s+(have|had|am|was|get|got|feel|felt)\b",
        r"\bmy\s+\w+",
        r"\bi'?m\s+(taking|on|experiencing)\b",
        r"\bfor\s+me\b",
        r"\bshould\s+i\b",
        r"\bcan\s+i\b",
        r"\bdo\s+i\b",
    ]
    detected_personal = any(re.search(p, q) for p in personal_signals)

    # INDIVIDUAL Rx / DOSING / DIAGNOSIS signals (the hard line). These are
    # specific medication/treatment/dose/diagnosis decisions about the person —
    # NOT general "what should I do".
    rx_signals = [
        r"\bshould\s+i\s+(take|start|stop|use|switch|increase|decrease|continue|change|try)\b",
        r"\bcan\s+i\s+(take|use|start|stop|combine|mix)\b",
        r"\bwhat\s+(dose|dosage)\b",
        r"\bhow\s+much\s+.*\b(take|should\s+i|to\s+take)\b",
        r"\b(my|the)\s+(dose|dosage)\b",
        r"\b(increase|decrease|change|adjust)\s+(my|the)\s+(dose|dosage|medication|meds?)\b",
        r"\bstop\s+(taking|my)\b",
        r"\bstart\s+taking\b",
        r"\bswitch\s+(to|my)\b",
        r"\bprescri(be|ption)\b",
        r"\bwhat\s+(drug|medication|antibiotic|statin|med)\b.*\b(should|do)\s+i\b",
        r"\bwhich\s+(drug|medication|antibiotic|statin|med)\b",
        r"\bdiagnos(e|is)\s+me\b",
        r"\bdo\s+i\s+have\b",
        r"\bis\s+it\s+safe\s+for\s+me\s+to\s+(take|stop|start|use)\b",
    ]
    detected_rx = any(re.search(p, q) for p in rx_signals)

    # HARD REFUSE: individual Rx/dosing/diagnosis decision.
    if detected_rx:
        return "refuse", {
            "detected_personal_context": detected_personal,
            "detected_rx_intent": True,
            "decision": "refuse",
            "reason": "personal_action_request",
        }

    # Config-driven hard refusal patterns — narrowed to the prescriptive ones
    # only (the broad "i have"/"for me"/"should i" entries would over-refuse the
    # 50/50 cases, so they are intentionally NOT auto-refused here).
    _PRESCRIPTIVE_CFG = {"prescribe", "my dose", "my dosage", "can i take"}
    for p in REFUSE_PATTERNS:
        if p in _PRESCRIPTIVE_CFG and re.search(re.escape(p), q):
            return "refuse", {
                "detected_personal_context": detected_personal,
                "detected_rx_intent": True,
                "decision": "refuse",
                "reason": "config_pattern_match",
            }

    # Otherwise allow. Personal-but-not-prescriptive -> 50/50 downstream.
    return "ok", {
        "detected_personal_context": detected_personal,
        "detected_rx_intent": False,
        "decision": "allow",
        "reason": "informational_or_educational",
    }


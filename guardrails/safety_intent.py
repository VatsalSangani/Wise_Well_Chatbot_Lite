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
    Safety Intent Tightening v2: Refuse ONLY explicit action/dosing/prescriptive asks.
    Lab interpretation is allowed informational.
    
    Returns (intent, debug_info)
      intent: "ok" | "refuse"
      debug_info: dict with detected_personal_context, detected_action_intent, decision, reason
    """
    
    q = (query or "").strip().lower()
    if not q:
        return "ok", {
            "detected_personal_context": False,
            "detected_action_intent": False,
            "decision": "allow",
            "reason": "empty_query",
        }

    # PERSONAL CONTEXT signals
    personal_signals = [
        r"\bi\s+(have|had|am|was|get|got)\b",  # I have, I am, I was
        r"\bmy\s+(doctor|patient|symptoms|results|values|test)\b",
        r"\bi'm\s+taking\b",
        r"\bfor\s+me\b",
        r"\bshould\s+i\b",
    ]
    
    detected_personal = any(re.search(p, q) for p in personal_signals)
    
    # ACTION REQUEST signals (prescriptive)
    action_signals = [
        r"\bshould\s+i\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\bcan\s+i\s+take\b",
        r"\bshould\s+i\s+take\b",
        r"\bcan\s+i\s+use\b",
        r"\bshould\s+i\s+use\b",
        r"\bdo\s+i\s+need\b",
        r"\bprescribe\b|\bprescription\b",
        r"\bdose\b|\bdosage\b",
        r"\bhow\s+much\b",
        r"\bstart\s+taking\b|\bstop\s+taking\b",
        r"\bincrease\s+|decrease\s+",
        r"\bswitch\s+to\b",
        r"\bis\s+it\s+safe\s+for\s+me\b",
        r"\brecommend\b",
        r"\btreat\s+me\b|\bcure\s+me\b",
        r"\bdiagnos(e|is)\s+me\b",
    ]
    
    detected_action = any(re.search(p, q) for p in action_signals)
    
    # REFUSE only if BOTH personal context AND action request
    if detected_personal and detected_action:
        return "refuse", {
            "detected_personal_context": True,
            "detected_action_intent": True,
            "decision": "refuse",
            "reason": "personal_action_request",
        }
    
    # Additional config-driven hard refusal patterns (tight)
    for p in REFUSE_PATTERNS:
        if re.search(p, q):
            return "refuse", {
                "detected_personal_context": detected_personal,
                "detected_action_intent": detected_action,
                "decision": "refuse",
                "reason": "config_pattern_match",
            }
    
    # Otherwise allow (may abstain later in evidence gate)
    return "ok", {
        "detected_personal_context": detected_personal,
        "detected_action_intent": detected_action,
        "decision": "allow",
        "reason": "informational_or_educational",
    }


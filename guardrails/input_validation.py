from __future__ import annotations

import re
from typing import Optional, Tuple

from guardrails.config_loader import load_guardrails_config

CFG = load_guardrails_config()

MIN_QUERY_CHARS = int(CFG.get("min_query_chars", 3))
MAX_QUERY_CHARS = int(CFG.get("max_query_chars", 512))
INJECTION_PATTERNS = CFG.get("injection_patterns", [])


def validate_query(q: str) -> Tuple[bool, Optional[str]]:
    if q is None:
        return False, "empty_query"

    q = q.strip()
    if len(q) < MIN_QUERY_CHARS:
        return False, "too_short"
    if len(q) > MAX_QUERY_CHARS:
        return False, "too_long"

    lowered = q.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return False, "prompt_injection_detected"

    return True, None

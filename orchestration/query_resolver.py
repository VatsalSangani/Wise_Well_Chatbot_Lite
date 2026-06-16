"""
Multi-turn query resolution: resolve genuine references in a follow-up using
recent history — CONSERVATIVELY.

DESIGN (reference-only, never additive):
  1. Resolution only fires when the current query has an ACTUAL unresolved
     reference: a third-person pronoun/deictic ("it", "that", "those", ...) or a
     bare fragment that is incomplete on its own ("how is it treated", "what
     about side effects", "and the symptoms?").
  2. A COMPLETE, self-contained question (its own explicit subject — "is coffee
     bad for me", "should I stop my BP medication", "write me a Python function")
     is passed through UNCHANGED with was_resolved=False. It never reaches the LLM.
  3. When it does resolve, the LLM may ONLY substitute the referent for the
     pronoun/fragment. It must NEVER add topical qualifiers ("if I have RA"),
     import clauses from earlier unrelated turns ("while I'm having a heart
     attack"), inject new subject matter, or change the question's intent.

This runs BEFORE the router, so the guardrails see a clean self-contained query.
A resolver that fabricates context corrupts retrieval (measured: faithfulness 0.0
/ hallucination_rate 1.0) and can cause false ESCALATE — hence the conservatism.
"""

import re
from typing import Dict, List, Optional
import structlog

from anthropic import Anthropic
from config import ANTHROPIC_MODEL_ID, DEFAULT_HISTORY_TURNS

logger = structlog.get_logger()

# Third-person referential pronouns/deictics that point at a prior subject.
# First-person (I/me/my/we/us/our) is intentionally EXCLUDED — it is present in
# self-contained personal queries ("should I stop my meds") and must not trigger.
_REFERENCE_PRONOUNS = re.compile(
    r"\b(it|its|it's|that|this|these|those|them|they|their|theirs)\b", re.IGNORECASE
)

# Fragment prefixes — query is a continuation, incomplete on its own.
_FRAGMENT_PREFIX = re.compile(
    r"^\s*(what about|how about|and\b|but\b|also\b|or\b|what else|anything else)",
    re.IGNORECASE,
)

# Bare attribute questions: ask about an attribute with no named subject
# ("what are the side effects"). If the query names a subject via "of X"
# ("benefits of exercise") it is self-contained and must NOT trigger.
_ATTRIBUTE = re.compile(
    r"\b(side[- ]?effects?|symptoms?|causes?|treatments?|treated|dosages?|doses?|"
    r"risks?|risk factors?|prevention|prognosis|complications?|benefits?|mechanism)\b",
    re.IGNORECASE,
)


def _needs_resolution(query: str) -> bool:
    """True only when the query has a genuine unresolved reference/fragment."""
    q = (query or "").strip()
    if not q:
        return False
    if _REFERENCE_PRONOUNS.search(q):
        return True
    if _FRAGMENT_PREFIX.search(q):
        return True
    # Bare attribute question: attribute word, no "of <subject>", and short.
    if _ATTRIBUTE.search(q) and " of " not in q.lower() and len(q.split()) <= 6:
        return True
    return False


_RESOLVER_SYSTEM_PROMPT = """You resolve references in a follow-up question using recent conversation history.

You do SUBSTITUTION ONLY. Replace the pronoun or complete the fragment with the
specific subject it refers to from the history. Nothing else.

STRICT RULES:
1. If the question already stands alone (has its own explicit subject), return it
   VERBATIM, unchanged.
2. You may ONLY substitute the referent for a pronoun/fragment. NEVER:
   - add topical qualifiers (do NOT turn "is coffee bad for me" into
     "is coffee bad for me if I have rheumatoid arthritis")
   - import clauses from earlier unrelated turns (do NOT turn "should I stop my
     blood pressure medication" into "...while I'm having a heart attack")
   - inject new subject matter (do NOT turn "write me a Python function" into
     "...to calculate methotrexate pharmacokinetics")
   - change the question's meaning or intent
3. Preserve the original phrasing and all its safety signals exactly; only the
   referent is substituted.

GOOD (genuine reference resolved):
- History mentions rheumatoid arthritis. Follow-up: "how is it treated?"
  -> "how is rheumatoid arthritis treated?"
- History mentions rheumatoid arthritis. Follow-up: "what are the side effects?"
  -> "what are the side effects of rheumatoid arthritis treatment?"

BAD (must NOT happen — return verbatim instead):
- Follow-up "is coffee bad for me" -> MUST stay "is coffee bad for me"
- Follow-up "does stress affect sleep" -> MUST stay "does stress affect sleep"
- Follow-up "should I stop taking my blood pressure medication"
  -> MUST stay "should I stop taking my blood pressure medication"
- Follow-up "write me a Python function" -> MUST stay "write me a Python function"

Recent history:
{history}

Follow-up question:
{query}

Output ONLY the resolved query (or the verbatim query if it stands alone). No explanation."""


def resolve_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> tuple[str, bool]:
    """
    Resolve a genuine reference in a follow-up against recent history.

    Returns (resolved_query, was_resolved). Self-contained queries and no-history
    cases pass through UNCHANGED with was_resolved=False (no LLM call).
    """
    # Single-turn: no history.
    if not history or len(history) == 0:
        return query, False

    # Conservative gate: only resolve genuine references/fragments. Self-contained
    # queries pass through unchanged and never reach the LLM.
    if not _needs_resolution(query):
        logger.info("query_resolution_skipped", reason="self_contained", query=query[:120])
        return query, False

    try:
        max_turns = DEFAULT_HISTORY_TURNS * 2
        history_to_use = history[-max_turns:] if len(history) > max_turns else history
        history_text = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in history_to_use)

        client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        response = client.messages.create(
            model=ANTHROPIC_MODEL_ID,
            max_tokens=200,
            temperature=0.0,
            system=_RESOLVER_SYSTEM_PROMPT.format(history=history_text, query=query),
            messages=[{"role": "user", "content": query}],
        )
        resolved = response.content[0].text.strip()

        # If the LLM returned the query unchanged, report was_resolved=False.
        changed = resolved.strip().lower() != query.strip().lower()
        logger.info(
            "query_resolved",
            original=query,
            resolved=resolved,
            changed=changed,
        )
        return resolved, changed

    except Exception as e:
        logger.error("query_resolution_failed", error=str(e), error_type=type(e).__name__)
        return query, False

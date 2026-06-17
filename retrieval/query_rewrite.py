"""
retrieval/query_rewrite.py

Lab-value query rewrite for dense retrieval.

PROBLEM (validated on the full corpus): dense retrieval is weak on lab-value
queries like "CRP 18 mg/L" (~0.52, drifts to irrelevant chunks). A keyword
filter did not fix it. FIX: detect a biomarker-with-number pattern, extract the
biomarker, and re-query conceptually ("CRP clinical significance elevated")
instead of embedding the raw numeric string.

Tokens come from config/guardrails.yaml (biomarker_tokens, unit_patterns,
unit_noise_tokens) — the same lists the guardrails already use, so we don't
introduce a second source of truth.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple

from guardrails.config_loader import load_guardrails_config

# Conceptual template the brief specifies, e.g. "CRP clinical significance elevated".
_REWRITE_TEMPLATE = "{biomarker} clinical significance elevated"

# A bare integer ("type 2", "stage 3") is NOT a lab value. A lab value is a
# number that is one of:
#   - a decimal            7.2
#   - a percent            7.2%  /  140 %
#   - unit-attached        18 mg/L, 450 ng/ml   (units from guardrails.yaml)
#   - adjacent to a biomarker token   "CRP 18", "cholesterol 240"
# This rejects "HbA1c and type 2 diabetes" (the 2 is a bare integer, not a lab
# value) while still catching the real lab-value queries.
_DECIMAL_RE = re.compile(r"(?<![a-z0-9])\d+\.\d+")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")


class RewriteResult(NamedTuple):
    query: str            # the query to embed (rewritten or original)
    rewritten: bool       # True if a lab-value pattern was detected and rewritten
    biomarker: str | None # the biomarker that drove the rewrite (None if not)
    original: str         # the untouched input, for logging
    kind: str = "none"    # "none" | "lab_value" | "concept_expansion"


# Significance-seeking phrasing — "what does an elevated X mean / indicate".
# These biomarker concept queries undermatch dense retrieval (e.g. "what does
# elevated CRP mean" scored 0.55, just under the RAG bar), but appending
# "clinical significance" lifts them over it. Same insight as the lab-value
# rewrite, applied to the no-number case.
_SIGNIFICANCE_WORDS = (
    "significance", "mean", "means", "meaning", "indicate", "indicates",
    "elevated", "high", "raised", "low", "level", "levels", "interpret",
    "interpretation", "abnormal",
)


def _config():
    return load_guardrails_config()


def _biomarker_tokens() -> List[str]:
    toks = list(_config().get("biomarker_tokens", []))
    # Longest first so multi-word tokens ("c-reactive protein") win over their
    # acronym ("crp") when both appear, and we report the most specific match.
    return sorted({t.lower() for t in toks}, key=len, reverse=True)


def _unit_alternation() -> str:
    cfg = _config()
    units = set(cfg.get("unit_patterns", [])) | set(cfg.get("unit_noise_tokens", []))
    # Drop "%" (handled separately) and ambiguous bare words that aren't units.
    units = {u.lower() for u in units if u not in {"%", "per", "day", "week", "month", "year"}}
    # Longest first so "mg/dl" wins over "mg".
    return "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))


def _token_present(token: str, text_lc: str) -> bool:
    """Whole-token match that tolerates hyphens/spaces inside the token
    (il-6, c-reactive protein) without matching inside a larger word."""
    pattern = r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])"
    return re.search(pattern, text_lc) is not None


def _has_unit_number(text_lc: str) -> bool:
    return re.search(rf"\d+(?:\.\d+)?\s*(?:{_unit_alternation()})(?![a-z0-9])", text_lc) is not None


def _number_adjacent_to_biomarker(text_lc: str, biomarker: str) -> bool:
    """True if a bare number token sits immediately before/after the biomarker
    token (e.g. 'CRP 18', 'cholesterol 240'), with no other word between."""
    bm = re.escape(biomarker)
    after = re.search(rf"(?<![a-z0-9]){bm}(?![a-z0-9])\s+\d+(?:\.\d+)?\b", text_lc)
    before = re.search(rf"\b\d+(?:\.\d+)?\s+{bm}(?![a-z0-9])", text_lc)
    return bool(after or before)


def _is_lab_value(text_lc: str, biomarker: str) -> bool:
    return (
        _DECIMAL_RE.search(text_lc) is not None
        or _PERCENT_RE.search(text_lc) is not None
        or _has_unit_number(text_lc)
        or _number_adjacent_to_biomarker(text_lc, biomarker)
    )


def rewrite_query(query: str) -> RewriteResult:
    """
    If the query is a biomarker-with-lab-value query, rewrite it to a conceptual
    query about that biomarker. Otherwise return it unchanged.
    """
    original = query
    text_lc = query.lower()

    for tok in _biomarker_tokens():
        if not _token_present(tok, text_lc):
            continue
        # Lab-value query (biomarker + a real value) -> conceptual replacement.
        if _is_lab_value(text_lc, tok):
            return RewriteResult(
                query=_REWRITE_TEMPLATE.format(biomarker=tok),
                rewritten=True, biomarker=tok, original=original, kind="lab_value",
            )
        # Biomarker significance question (no number) -> AUGMENT with "clinical
        # significance" to lift the undermatching dense score over the RAG bar.
        if any(w in text_lc for w in _SIGNIFICANCE_WORDS):
            return RewriteResult(
                query=f"{original} clinical significance",
                rewritten=True, biomarker=tok, original=original, kind="concept_expansion",
            )

    return RewriteResult(query=original, rewritten=False, biomarker=None, original=original, kind="none")

"""
WiseWell — LLM Synthesis via the Anthropic API (Claude Haiku)

Takes approved evidence snippets (passed all 8 guardrail stages) and generates
structured JSON answers that are assembled into clean markdown. Auth via ANTHROPIC_API_KEY.

CRITICAL: Only evidence that cleared every guardrail stage reaches here.
Safety paths (ESCALATE/REFUSE/CHITCHAT/CLARIFY) bypass synthesis entirely.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
import structlog

from config import ANTHROPIC_MODEL_ID, SYNTHESIS_MAX_TOKENS, SYNTHESIS_TEMPERATURE, SYNTHESIS_MIN_LENGTH
from orchestration import observability as obs

logger = structlog.get_logger()

# Shared tone instruction — applies to BOTH RAG and general modes
_TONE = (
    "Explain in warm, clear, plain language a non-expert can follow. Be kind and direct. "
    "Never minimize a concern or give false reassurance. Don't lecture. If the topic is "
    "serious, stay honest and calm rather than alarming or dismissive. Keep the individual "
    "general: explain what things mean generally, never diagnose the person or tell them "
    "what to take or do for their specific case."
)

# Off-topic redirect — SINGLE SOURCE OF TRUTH. OFFTOPIC_MARKER is the stable core
# phrase used BOTH inside the prompt's redirect summary AND by query.py to detect
# the redirect, so the two cannot drift. It deliberately excludes leading branding
# and punctuation (em-dash) so detection survives tone tweaks to the surrounding
# wording. OFFTOPIC_SUMMARY is the full summary text the prompt asks for, built
# from the marker so any reword keeps detection in sync.
OFFTOPIC_MARKER = "I can only help with health and medical questions"
OFFTOPIC_SUMMARY = f"I'm WiseWell, a medical information assistant — {OFFTOPIC_MARKER}."

# Domain scope — answer ONLY health/medical questions (shared by both modes).
# The redirect summary is injected from OFFTOPIC_MARKER via .replace() (rather
# than an f-string) to avoid escaping every brace in the JSON example block.
_DOMAIN_SCOPE = """DOMAIN SCOPE — WiseWell answers ONLY health and medical questions.
If the question is NOT about health, medicine, biology, the body, conditions,
symptoms, treatments, medications, wellness, nutrition, fitness, mental health,
sleep, or related medical topics, do NOT answer it (no code, math, politics,
trivia, essays, general life advice, or anything outside health/medicine).

For off-topic questions, return exactly:
{
  "summary": "__OFFTOPIC_MARKER__",
  "key_points": [],
  "explanation": "I'd be glad to help with anything health-related — understanding a condition, what a medical term means, how a treatment works, or making sense of symptoms in general terms. What health topic can I help you with?",
  "when_to_see_doctor": null,
  "citations": [],
  "follow_up_suggestion": null
}

BE INCLUSIVE about health: nutrition, exercise, sleep, stress, mental health,
wellness, "is X bad for me," aging, diet are all IN-SCOPE — answer them.
Only redirect questions clearly OUTSIDE health/medicine entirely. When in
doubt, treat as medical and answer. Do NOT over-block legitimate health questions.""".replace(
    "__OFFTOPIC_MARKER__", OFFTOPIC_SUMMARY
)

# Follow-up suggestion rules — educational topics only, never personal advice.
_FOLLOWUP_RULES = """FOLLOW-UP SUGGESTION RULES (the "follow_up_suggestion" field):
- Suggest ONE specific, natural next step that explores a TOPIC: how something
  works, a related concept, what a term means, or broader context.
- It must be EDUCATIONAL only. Set it to null if there is no genuine natural follow-up.
- NEVER suggest anything about the individual's personal situation. FORBIDDEN:
  · personal diagnosis ("would you like to know if you have...")
  · dosing/medication advice ("...what dose to take", "...whether to stop your medication")
  · personal treatment ("what you should do about...")
  · inviting the user to share their case ("tell me about your symptoms")
- The suggestion is about exploring the SUBJECT, never about advising the person.
  Good: "Would you like to know how it's treated?"
  Good: "Would you like to understand what causes elevated levels?"
  Bad:  "Would you like to know if you should stop your medication?"
  Bad:  "Would you like to know what dose is right for you?\""""

# RAG mode: structured JSON synthesis with citations from evidence
_RAG_JSON_PROMPT = f"""You are WiseWell, a medical information assistant that synthesizes evidence into structured JSON.

{_DOMAIN_SCOPE}

CRITICAL RULES:
1. Return ONLY valid JSON, no markdown fences (```), no preamble, no explanation before or after the JSON
2. Use ONLY the evidence provided — do not add external knowledge
3. Every factual claim must be supported by a citation — cite ALL claims with PMID from the evidence
4. If evidence conflicts, acknowledge the disagreement in the explanation field
5. If evidence is insufficient, say so clearly

TONE: {_TONE}

{_FOLLOWUP_RULES}

SCHEMA (return EXACTLY this structure, no variations):
{{
  "summary": "1-2 sentence direct answer to the user's question",
  "key_points": ["key point 1", "key point 2", "key point 3"],
  "explanation": "the main educational content in warm, plain language (not a bulleted list)",
  "when_to_see_doctor": "when to seek professional care (or null if not applicable)",
  "citations": [
    {{"pmid": "12345678", "claim": "what this source supports"}},
    {{"pmid": "23456789", "claim": "what this source supports"}}
  ],
  "follow_up_suggestion": "a single natural educational next-step question, or null if none"
}}

Available Evidence:
{{EVIDENCE}}

Question: {{QUESTION}}

Output ONLY the JSON object. No markdown, no fences, no preamble."""

# General mode: structured JSON synthesis without citations
_GENERAL_JSON_PROMPT = f"""You are WiseWell, a medical information assistant answering from general medical knowledge.

{_DOMAIN_SCOPE}

CRITICAL RULES:
1. Return ONLY valid JSON, no markdown fences (```), no preamble, no explanation before or after the JSON
2. Give a helpful, general, educational explanation
3. Do NOT cite PMIDs or claim specific studies — you have none here
4. Do NOT invent references or numbers
5. Stay general and educational; do not diagnose the person or recommend specific treatment/medication

TONE: {_TONE}

{_FOLLOWUP_RULES}

SCHEMA (return EXACTLY this structure, no variations):
{{
  "summary": "1-2 sentence direct answer",
  "key_points": ["key point 1", "key point 2"],
  "explanation": "the main educational content in warm, plain language",
  "when_to_see_doctor": "when to seek professional care (or null if not applicable)",
  "citations": [],
  "follow_up_suggestion": "a single natural educational next-step question, or null if none"
}}

Question: {{QUESTION}}

Output ONLY the JSON object. No markdown, no fences, no preamble."""

# Plain-text fallback prompt (used if JSON parsing fails for general mode)
_GENERAL_PLAINTEXT_PROMPT = f"""You are WiseWell, a medical information assistant answering from general medical knowledge.

Rules:
1. Give a helpful, general, educational explanation in plain text.
2. Do NOT cite PMIDs or claim specific studies.
3. Do NOT invent references or numbers.
4. Stay general and educational; do not diagnose or prescribe for the individual.

TONE: {_TONE}

Question: {{QUESTION}}

Answer:"""


# Forbidden patterns for follow_up_suggestion — anything that points at personal
# advice/diagnosis rather than exploring a topic. Matched case-insensitively.
_FOLLOWUP_FORBIDDEN = [
    r"\byou should\b", r"\byour dose\b", r"\byour medication\b", r"\byour meds\b",
    r"should you (take|stop|start|increase|decrease|change)",
    r"\bdo you have\b", r"whether to (take|stop|start)",
    r"what (dose|dosage)", r"if you (have|should)", r"for you\b",
    r"what you should (do|take)",
    r"tell me (your|about your)", r"describe your", r"\bassess\b",
    r"\bdiagnose\b", r"your (symptoms|condition|situation)",
]


def sanitize_follow_up(suggestion: Optional[str]) -> Optional[str]:
    """
    Safety-net: drop a follow_up_suggestion that points at personal advice.
    Returns the suggestion unchanged if clean, or None if it matches a
    forbidden personal-advice pattern.
    """
    if not suggestion or not isinstance(suggestion, str):
        return None
    text = suggestion.strip()
    if not text:
        return None
    for pattern in _FOLLOWUP_FORBIDDEN:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning("followup_suggestion_dropped", suggestion=text, pattern=pattern)
            return None
    return text


def parse_structured_answer(text: str) -> dict:
    """
    Parse structured JSON answer with graceful fallback.

    Returns dict with keys: summary, key_points, explanation,
                           when_to_see_doctor, citations

    Never crashes, never shows raw JSON to user.
    """
    text = text.strip()

    # Try 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if all(key in parsed for key in ["summary", "key_points", "explanation"]):
            logger.info("structured_answer_parsed", method="direct")
            return parsed
    except json.JSONDecodeError:
        pass

    # Try 2: Strip markdown fences (```json ... ```)
    # Remove both opening and closing backtick fences with surrounding newlines
    text_stripped = re.sub(r'\n?```(?:json)?\n?', '\n', text).strip()
    if text_stripped != text.strip():
        try:
            parsed = json.loads(text_stripped)
            if all(key in parsed for key in ["summary", "key_points", "explanation"]):
                logger.info("structured_answer_parsed", method="fence_strip")
                return parsed
        except json.JSONDecodeError as e:
            logger.debug("fence_strip_parse_failed", error=str(e), preview=text_stripped[:100])
            pass

    # Try 3: Extract JSON object from text (with preamble/explanation)
    match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if all(key in parsed for key in ["summary", "key_points", "explanation"]):
                logger.info("structured_answer_parsed", method="regex_extract")
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: parsing failed completely — log the raw response for debugging
    logger.warning(
        "structured_answer_parse_failed",
        text_length=len(text),
        text_preview=text[:500],
        reason="All JSON parsing attempts failed"
    )
    return None  # Signal parse failure to caller


def validate_and_filter_citations(citations: List[Dict[str, str]], retrieved_snippets: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Validate structured citations against actually-retrieved snippets.
    Drop any PMID not in the retrieved set (prevents hallucinated citations).

    Args:
        citations: list of {pmid, claim} dicts from LLM
        retrieved_snippets: list of evidence snippets actually used

    Returns:
        Filtered list of validated citations
    """
    retrieved_pmids = {str(s.get("pmid")) for s in retrieved_snippets if s.get("pmid")}

    validated = []
    for citation in citations:
        pmid = str(citation.get("pmid", ""))
        if pmid in retrieved_pmids:
            validated.append(citation)
        else:
            logger.warning(
                "hallucinated_citation_dropped",
                pmid=pmid,
                claim=citation.get("claim", "")
            )

    return validated


class LLMSynthesizer:
    """
    Synthesizes guardrail-approved evidence into structured JSON answers
    using the Anthropic API (Claude Haiku). Auth via ANTHROPIC_API_KEY from env.
    """

    def __init__(self) -> None:
        self.client = Anthropic()
        self.model_id = ANTHROPIC_MODEL_ID
        logger.info("llm_synthesizer_ready", model=self.model_id)

    def synthesize(
        self,
        query: str,
        evidence_snippets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """RAG mode: structured JSON synthesis with citations from evidence."""
        try:
            context = self._build_context(evidence_snippets)
            system_prompt = _RAG_JSON_PROMPT.replace("{{EVIDENCE}}", context).replace("{{QUESTION}}", query)

            logger.info(
                "rag_synthesis_start",
                query_length=len(query),
                evidence_count=len(evidence_snippets),
                context_length=len(context),
                max_tokens=SYNTHESIS_MAX_TOKENS
            )

            with obs.generation("synthesis_rag", model=self.model_id, input=query) as gen:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=SYNTHESIS_MAX_TOKENS,
                    temperature=SYNTHESIS_TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": query}],
                )
                text = response.content[0].text.strip()

                # stop_reason="max_tokens" signals truncation (unparseable JSON).
                stop_reason = getattr(response, "stop_reason", None)
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None) if usage else None
                output_tokens = getattr(usage, "output_tokens", None) if usage else None

                # v4 usage_details keys are "input"/"output"; model drives cost.
                gen.update(
                    output=text,
                    usage_details={"input": input_tokens, "output": output_tokens},
                )

            logger.info(
                "rag_synthesis_response",
                response_length=len(text),
                stop_reason=stop_reason,
                output_tokens=output_tokens,
            )

            # Parse JSON
            parsed = parse_structured_answer(text)
            if parsed is None:
                # Parse failure → route to extractive fallback
                logger.warning(
                    "rag_json_parse_failed_using_extractive",
                    response_length=len(text),
                    stop_reason=stop_reason,
                    output_tokens=output_tokens,
                    response_preview=text[:300]
                )
                return self._extractive_fallback(evidence_snippets)

            # Log what the LLM returned before citation validation
            logger.debug(
                "parsed_json_before_citation_validation",
                citations_in_response=len(parsed.get("citations", [])),
                parsed_summary=parsed.get("summary", "")[:50]
            )

            # Validate and filter citations
            if "citations" in parsed and parsed["citations"]:
                before_count = len(parsed["citations"])
                parsed["citations"] = validate_and_filter_citations(
                    parsed["citations"],
                    evidence_snippets
                )
                after_count = len(parsed["citations"])
                if before_count != after_count:
                    logger.warning(
                        "citations_filtered",
                        before=before_count,
                        after=after_count
                    )

            # Safety-net: drop personal-advice follow-up suggestions
            parsed["follow_up_suggestion"] = sanitize_follow_up(parsed.get("follow_up_suggestion"))

            logger.info(
                "llm_synthesis_success",
                evidence_count=len(evidence_snippets),
                citations_count=len(parsed.get("citations", [])),
            )

            return {
                "synthesized_answer": parsed,
                "structured": True,
                "success": True,
                "fallback": False
            }

        except Exception as e:
            logger.error("llm_synthesis_error", error=str(e), error_type=type(e).__name__)
            return self._extractive_fallback(evidence_snippets)

    def synthesize_general(self, query: str) -> Dict[str, Any]:
        """General mode: structured JSON synthesis without citations."""
        try:
            system_prompt = _GENERAL_JSON_PROMPT.replace("{{QUESTION}}", query)

            with obs.generation("synthesis_general", model=self.model_id, input=query) as gen:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=SYNTHESIS_MAX_TOKENS,
                    temperature=SYNTHESIS_TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": query}],
                )
                text = response.content[0].text.strip()
                usage = getattr(response, "usage", None)
                gen.update(
                    output=text,
                    usage_details={
                        "input": getattr(usage, "input_tokens", None) if usage else None,
                        "output": getattr(usage, "output_tokens", None) if usage else None,
                    },
                )

            # Parse JSON
            parsed = parse_structured_answer(text)
            if parsed is None:
                # Parse failure → retry with plain-text prompt (one retry only)
                logger.warning("general_json_parse_failed_retrying_plaintext")
                return self._synthesize_general_plaintext(query)

            # Validate minimum length
            explanation_len = len(parsed.get("explanation", ""))
            if explanation_len < SYNTHESIS_MIN_LENGTH:
                logger.warning("general_answer_too_short", length=explanation_len)
                return self._synthesize_general_plaintext(query)

            # Safety-net: drop personal-advice follow-up suggestions
            parsed["follow_up_suggestion"] = sanitize_follow_up(parsed.get("follow_up_suggestion"))

            logger.info("llm_general_success", response_length=explanation_len)
            return {
                "synthesized_answer": parsed,
                "structured": True,
                "success": True
            }

        except Exception as e:
            logger.error("llm_general_error", error=str(e), error_type=type(e).__name__)
            # On exception, try plain-text fallback
            return self._synthesize_general_plaintext(query)

    def _synthesize_general_plaintext(self, query: str) -> Dict[str, Any]:
        """
        Fallback for general mode: one plain-text (non-JSON) retry.
        If this fails too, return None (caller shows clean error message).
        """
        try:
            system_prompt = _GENERAL_PLAINTEXT_PROMPT.replace("{{QUESTION}}", query)

            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=SYNTHESIS_MAX_TOKENS,
                temperature=SYNTHESIS_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": query}],
            )
            text = response.content[0].text.strip()

            if len(text) < SYNTHESIS_MIN_LENGTH:
                logger.warning("general_plaintext_too_short", length=len(text))
                return {"synthesized_answer": None, "success": False}

            logger.info("llm_general_plaintext_success", response_length=len(text))
            return {
                "synthesized_answer": text,
                "structured": False,
                "success": True
            }

        except Exception as e:
            logger.error("llm_general_plaintext_error", error=str(e), error_type=type(e).__name__)
            return {"synthesized_answer": None, "success": False}

    # ── helpers ────────────────────────────────────────────────

    def _build_context(self, snippets: List[Dict[str, Any]]) -> str:
        parts = []
        for i, s in enumerate(snippets, 1):
            parts.append(
                f"[{i}] PMID: {s.get('pmid', 'Unknown')} ({s.get('year', '')})\n"
                f"Title: {s.get('title', '')}\n"
                f"Evidence: {s.get('text', '')}\n"
            )
        return "\n".join(parts)

    def _extractive_fallback(self, snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback: return top 3 snippets as clean prose (not raw abstract sections).
        Removes common structured-abstract headers (INTRODUCTION/METHODS/RESULTS/etc)
        to show readable content.
        """
        top = snippets[:3]

        # Clean each snippet: remove structured-abstract headers and condense
        cleaned_snippets = []
        for s in top:
            text = s.get('text', '')
            # Remove common structured-abstract headers
            text = re.sub(
                r'\b(BACKGROUND|INTRODUCTION|METHODS|RESULTS|CONCLUSION|OBJECTIVE|PURPOSE|STUDY DESIGN|PARTICIPANTS?|DATA|ANALYSIS)[:\s]+',
                '',
                text,
                flags=re.IGNORECASE
            )
            # Remove excessive whitespace
            text = ' '.join(text.split())
            # Take first 200 chars to keep it concise
            if len(text) > 200:
                text = text[:197] + "..."

            title = s.get('title', '')
            pmid = s.get('pmid', 'Unknown')

            cleaned_snippets.append(f"{title}\n{text}\n[PMID: {pmid}]")

        # Join with clear separation, no duplicate source list
        answer = "\n\n".join(cleaned_snippets)

        return {
            "synthesized_answer": answer,
            "structured": False,
            "success": False,
            "fallback": True,
        }


_synthesizer: Optional[LLMSynthesizer] = None


def _get_synthesizer() -> LLMSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = LLMSynthesizer()
    return _synthesizer


def synthesize_response(
    query: str,
    evidence_snippets: List[Dict[str, Any]],
    enable_llm: bool = True,
) -> Dict[str, Any]:
    """RAG mode synthesis (with evidence)."""
    if not enable_llm:
        logger.info("llm_disabled_using_extractive")
        return _get_synthesizer()._extractive_fallback(evidence_snippets)
    return _get_synthesizer().synthesize(query, evidence_snippets)


def synthesize_general_response(query: str) -> Dict[str, Any]:
    """General mode synthesis (no evidence). Caller appends the code-inserted disclaimer."""
    return _get_synthesizer().synthesize_general(query)

"""
WiseWell — LLM Synthesis via the Anthropic API (Claude Haiku)

Takes approved evidence snippets (passed all 8 guardrail stages) and generates
a natural-language response. Auth via ANTHROPIC_API_KEY from the environment.

CRITICAL: Only evidence that cleared every guardrail stage reaches here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
import structlog

from config import ANTHROPIC_MODEL_ID, SYNTHESIS_MAX_TOKENS, SYNTHESIS_TEMPERATURE, SYNTHESIS_MIN_LENGTH

logger = structlog.get_logger()

# Shared tone instruction — applies to BOTH RAG and general modes. Warmth here
# governs LLM-generated content; code-inserted strings (disclaimers/defers) carry
# their own warm tone from guardrails/responses.py.
_TONE = (
    "Explain in warm, clear, plain language a non-expert can follow. Be kind and direct. "
    "Never minimize a concern or give false reassurance. Don't lecture. If the topic is "
    "serious, stay honest and calm rather than alarming or dismissive. Keep the individual "
    "general: explain what things mean generally, never diagnose the person or tell them "
    "what to take or do for their specific case."
)

_SYSTEM_PROMPT = f"""You are WiseWell, a medical information assistant that synthesizes evidence from medical literature.

CRITICAL RULES:
1. Use ONLY the evidence provided — do not add external knowledge
2. Cite EVERY factual claim with [PMID: XXXXX] inline
3. If evidence conflicts, acknowledge the disagreement
4. If evidence is insufficient, say so clearly

{_TONE}

NEVER: Add information not in the evidence · hallucinate PMIDs · give personal medical advice"""

# General mode: the LLM answers from its own general knowledge (retrieval was too
# weak for cited RAG). NO citations — a code-inserted disclaimer is added
# separately, so the model must NOT invent PMIDs or pretend to cite studies.
_GENERAL_SYSTEM_PROMPT = f"""You are WiseWell, a medical information assistant.

You are answering from general medical knowledge because no specific study in the
library closely matched this question.

RULES:
1. Give a helpful, general, educational explanation.
2. Do NOT cite PMIDs or claim a specific study — you have none here.
3. Do NOT invent references or numbers.
4. Stay general and educational; do not diagnose the person or recommend specific
   treatment, medication, or dosing for their individual case.

{_TONE}"""


class LLMSynthesizer:
    """
    Synthesizes guardrail-approved evidence into natural language using
    the Anthropic API (Claude Haiku). Auth via ANTHROPIC_API_KEY from env.
    """

    def __init__(self) -> None:
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model_id = ANTHROPIC_MODEL_ID
        logger.info("llm_synthesizer_ready", model=self.model_id)

    def synthesize(
        self,
        query: str,
        evidence_snippets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            context = self._build_context(evidence_snippets)
            user_prompt = self._build_prompt(query, context, evidence_snippets)

            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=SYNTHESIS_MAX_TOKENS,
                temperature=SYNTHESIS_TEMPERATURE,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()

            validation = self._validate(text, evidence_snippets)
            if validation["valid"]:
                logger.info(
                    "llm_synthesis_success",
                    evidence_count=len(evidence_snippets),
                    response_length=len(text),
                )
                return {"synthesized_answer": text, "citations_used": validation["citations_used"], "success": True, "fallback": False}

            logger.warning("llm_synthesis_validation_failed", reason=validation["reason"])
            return self._extractive_fallback(evidence_snippets)

        except Exception as e:
            logger.error("llm_synthesis_error", error=str(e), error_type=type(e).__name__)
            return self._extractive_fallback(evidence_snippets)

    def synthesize_general(self, query: str) -> Dict[str, Any]:
        """General mode: answer from the model's own general knowledge (no
        evidence). The code-inserted disclaimer is added by the caller; this
        method only produces the educational content."""
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=SYNTHESIS_MAX_TOKENS,
                temperature=SYNTHESIS_TEMPERATURE,
                system=_GENERAL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            text = response.content[0].text.strip()
            if len(text) < SYNTHESIS_MIN_LENGTH:
                return {"synthesized_answer": None, "success": False}
            logger.info("llm_general_success", response_length=len(text))
            return {"synthesized_answer": text, "success": True}
        except Exception as e:
            logger.error("llm_general_error", error=str(e), error_type=type(e).__name__)
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

    def _build_prompt(self, query: str, context: str, snippets: List[Dict[str, Any]]) -> str:
        pmid_list = ", ".join(str(s["pmid"]) for s in snippets if s.get("pmid"))
        return (
            f"Question: {query}\n\n"
            f"Available Evidence:\n{context}\n"
            f"Task: Synthesize the evidence into a natural response. "
            f"Cite every claim. Available PMIDs: {pmid_list}\n\nResponse:"
        )

    def _validate(self, text: str, snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
        cited  = set(re.findall(r"PMID:\s*(\d+)", text))
        approved = {str(s["pmid"]) for s in snippets if s.get("pmid")}
        hallucinated = cited - approved
        if hallucinated:
            return {"valid": False, "citations_used": [], "reason": f"Hallucinated PMIDs: {hallucinated}"}
        if len(text) < SYNTHESIS_MIN_LENGTH:
            return {"valid": False, "citations_used": [], "reason": "Response too short"}
        if not cited:
            return {"valid": False, "citations_used": [], "reason": "No citations found"}
        return {"valid": True, "citations_used": list(cited), "reason": None}

    def _extractive_fallback(self, snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
        top = snippets[:3]
        answer = " ".join(f"{s.get('text', '')} [PMID: {s.get('pmid', 'Unknown')}]" for s in top)
        return {
            "synthesized_answer": answer,
            "citations_used": [str(s["pmid"]) for s in top if s.get("pmid")],
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
    if not enable_llm:
        logger.info("llm_disabled_using_extractive")
        return _get_synthesizer()._extractive_fallback(evidence_snippets)
    return _get_synthesizer().synthesize(query, evidence_snippets)


def synthesize_general_response(query: str) -> Dict[str, Any]:
    """General-mode synthesis (no evidence). Caller appends the code-inserted
    disclaimer."""
    return _get_synthesizer().synthesize_general(query)

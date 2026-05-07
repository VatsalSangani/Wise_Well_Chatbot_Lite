"""
WiseWell — LLM Synthesis via AWS Bedrock (Claude Sonnet)

Takes approved evidence snippets (passed all 8 guardrail stages) and generates
a natural-language response. Uses IAM role auth — no API key required.

CRITICAL: Only evidence that cleared every guardrail stage reaches here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import boto3
import structlog

from config import BEDROCK_MODEL_ID, BEDROCK_REGION, SYNTHESIS_MAX_TOKENS, SYNTHESIS_TEMPERATURE, SYNTHESIS_MIN_LENGTH

logger = structlog.get_logger()

_SYSTEM_PROMPT = """You are a medical information assistant that synthesizes evidence from medical literature.

CRITICAL RULES:
1. Use ONLY the evidence provided — do not add external knowledge
2. Cite EVERY factual claim with [PMID: XXXXX] inline
3. Write in clear, natural language — avoid robotic bullet points
4. If evidence conflicts, acknowledge the disagreement
5. Stay objective — do not make recommendations or give personal advice
6. If evidence is insufficient, say so clearly

STYLE: Natural flowing prose · professional but accessible · short paragraphs

NEVER: Add information not in the evidence · hallucinate PMIDs · give medical advice"""


class LLMSynthesizer:
    """
    Synthesizes guardrail-approved evidence into natural language using
    AWS Bedrock Claude Sonnet. Auth is via IAM role (no API key).
    """

    def __init__(self) -> None:
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        self.model_id = BEDROCK_MODEL_ID
        logger.info("llm_synthesizer_ready", model=self.model_id, region=BEDROCK_REGION)

    def synthesize(
        self,
        query: str,
        evidence_snippets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            context = self._build_context(evidence_snippets)
            user_prompt = self._build_prompt(query, context, evidence_snippets)

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": SYNTHESIS_MAX_TOKENS,
                "temperature": SYNTHESIS_TEMPERATURE,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            text = result["content"][0]["text"].strip()

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

# orchestration/llm_synthesis.py
"""
LLM Synthesis Module for WiseWell

Takes approved evidence snippets and uses GPT-4o-mini to generate
natural language responses while maintaining strict citation requirements.

CRITICAL: LLM only sees evidence that passed all 8 guardrail stages.
Cannot add information not in the approved snippets.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger()


class LLMSynthesizer:
    """
    Synthesizes evidence snippets into natural language using GPT-4o-mini.
    
    Safety features:
    - Only uses pre-approved evidence
    - Requires citation for every claim
    - Validates no hallucinated PMIDs
    - Falls back to extractive if synthesis fails
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM synthesizer.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"  # Fast and cheap
        
        logger.info("llm_synthesizer_initialized", model=self.model)
    
    def synthesize(
        self,
        query: str,
        evidence_snippets: List[Dict[str, Any]],
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Synthesize evidence into natural language response.
        
        Args:
            query: Original user query
            evidence_snippets: Approved evidence from retrieval pipeline
            max_tokens: Maximum response length
            
        Returns:
            {
                "synthesized_answer": str,
                "citations_used": List[str],  # PMIDs referenced
                "success": bool,
                "fallback": bool  # True if fell back to extractive
            }
        """
        try:
            # Build evidence context
            context = self._build_evidence_context(evidence_snippets)
            
            # Create synthesis prompt
            prompt = self._create_synthesis_prompt(query, context, evidence_snippets)
            
            # Call GPT-4o-mini
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for factual responses
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            synthesized_text = response.choices[0].message.content.strip()
            
            # Validate response
            validation = self._validate_response(synthesized_text, evidence_snippets)
            
            if validation["valid"]:
                logger.info(
                    "llm_synthesis_success",
                    query_length=len(query),
                    evidence_count=len(evidence_snippets),
                    response_length=len(synthesized_text),
                    citations_used=len(validation["citations_used"])
                )
                
                return {
                    "synthesized_answer": synthesized_text,
                    "citations_used": validation["citations_used"],
                    "success": True,
                    "fallback": False
                }
            else:
                # Validation failed - fall back to extractive
                logger.warning(
                    "llm_synthesis_validation_failed",
                    reason=validation["reason"],
                    falling_back=True
                )
                return self._extractive_fallback(evidence_snippets)
                
        except Exception as e:
            logger.error(
                "llm_synthesis_error",
                error=str(e),
                error_type=type(e).__name__,
                falling_back=True
            )
            return self._extractive_fallback(evidence_snippets)
    
    def _get_system_prompt(self) -> str:
        """System prompt defining LLM behavior."""
        return """You are a medical information assistant that synthesizes evidence from medical literature.

CRITICAL RULES:
1. Use ONLY the evidence provided - do not add external knowledge
2. Cite EVERY factual claim with [PMID: XXXXX] inline
3. Write in clear, natural language - avoid robotic bullet points
4. If evidence conflicts, acknowledge the disagreement
5. Stay objective - do not make recommendations or give advice
6. If evidence is insufficient, say so clearly

STYLE:
- Natural, flowing prose
- Professional but accessible tone
- Short paragraphs (2-3 sentences max)
- Clear and concise

NEVER:
- Add information not in the provided evidence
- Make up or hallucinate PMIDs
- Give personal medical advice
- Make treatment recommendations"""
    
    def _build_evidence_context(self, snippets: List[Dict[str, Any]]) -> str:
        """Build formatted evidence context for LLM."""
        context_parts = []
        
        for i, snippet in enumerate(snippets, 1):
            pmid = snippet.get("pmid", "Unknown")
            year = snippet.get("year", "")
            title = snippet.get("title", "")
            text = snippet.get("text", "")
            
            context_parts.append(
                f"[{i}] PMID: {pmid} ({year})\n"
                f"Title: {title}\n"
                f"Evidence: {text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_synthesis_prompt(
        self,
        query: str,
        context: str,
        snippets: List[Dict[str, Any]]
    ) -> str:
        """Create the synthesis prompt."""
        pmid_list = ", ".join([
            s.get("pmid", "Unknown") 
            for s in snippets 
            if s.get("pmid")
        ])
        
        return f"""Question: {query}

Available Evidence (from medical literature):
{context}

Task: Synthesize the above evidence into a natural, flowing response that answers the question.

Requirements:
- Use ALL relevant evidence snippets
- Cite EVERY factual claim with [PMID: XXXXX]
- Write in natural language (not bullet points)
- Be concise but complete (3-5 sentences typically)
- Only use information from the evidence above
- Available PMIDs to cite: {pmid_list}

Response:"""
    
    def _validate_response(
        self,
        response: str,
        evidence_snippets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that response only uses approved PMIDs.
        
        Returns:
            {
                "valid": bool,
                "citations_used": List[str],
                "reason": str (if invalid)
            }
        """
        import re
        
        # Extract PMIDs from response
        cited_pmids = set(re.findall(r'PMID:\s*(\d+)', response))
        
        # Get approved PMIDs
        approved_pmids = {
            str(s.get("pmid")) 
            for s in evidence_snippets 
            if s.get("pmid")
        }
        
        # Check for hallucinated PMIDs
        hallucinated = cited_pmids - approved_pmids
        
        if hallucinated:
            return {
                "valid": False,
                "citations_used": [],
                "reason": f"Hallucinated PMIDs: {hallucinated}"
            }
        
        # Check if response is too short (likely failed)
        if len(response) < 50:
            return {
                "valid": False,
                "citations_used": [],
                "reason": "Response too short"
            }
        
        # Check if response has citations
        if not cited_pmids:
            return {
                "valid": False,
                "citations_used": [],
                "reason": "No citations found"
            }
        
        return {
            "valid": True,
            "citations_used": list(cited_pmids),
            "reason": None
        }
    
    def _extractive_fallback(
        self,
        evidence_snippets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fallback to extractive answer if synthesis fails.
        
        Simply concatenates the top evidence snippets.
        """
        # Take top 3 snippets
        top_snippets = evidence_snippets[:3]
        
        # Build extractive answer
        parts = []
        for snippet in top_snippets:
            text = snippet.get("text", "")
            pmid = snippet.get("pmid", "Unknown")
            parts.append(f"{text} [PMID: {pmid}]")
        
        answer = " ".join(parts)
        
        pmids = [
            str(s.get("pmid")) 
            for s in top_snippets 
            if s.get("pmid")
        ]
        
        return {
            "synthesized_answer": answer,
            "citations_used": pmids,
            "success": False,
            "fallback": True
        }


# Singleton instance
_synthesizer: Optional[LLMSynthesizer] = None


def get_synthesizer() -> LLMSynthesizer:
    """Get or create LLM synthesizer singleton."""
    global _synthesizer
    
    if _synthesizer is None:
        _synthesizer = LLMSynthesizer()
    
    return _synthesizer


def synthesize_response(
    query: str,
    evidence_snippets: List[Dict[str, Any]],
    enable_llm: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to synthesize response.
    
    Args:
        query: User query
        evidence_snippets: Approved evidence
        enable_llm: If False, returns extractive answer
        
    Returns:
        Synthesis result dict
    """
    if not enable_llm or not os.getenv("OPENAI_API_KEY"):
        # Fall back to extractive
        logger.info("llm_disabled_using_extractive")
        return get_synthesizer()._extractive_fallback(evidence_snippets)
    
    return get_synthesizer().synthesize(query, evidence_snippets)
# backend/schemas.py
"""
Pydantic schemas for WiseWell API

All request/response models for type safety and validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class QueryRequest(BaseModel):
    """Request schema for /query endpoint.

    SINGLE-TURN BY DESIGN: each /query is fully independent and stateless. There
    is intentionally no `history` / `session` field — do NOT add an in-process
    conversation store. Multi-turn memory is a deferred v2 feature and will use
    an external session store (Redis keyed by session id), not server-side state.
    See README "Stateless design / scaling" for the reasoning (safety guardrails
    evaluate single messages, not assembled history).
    """

    query: str = Field(
        ...,
        description="Medical information query",
        min_length=1,
        max_length=500,
        examples=["What is CRP and what does it indicate?"]
    )
    debug: bool = Field(
        default=False,
        description="Include debug information (timings, signals)"
    )
    
    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v):
        """Validate that query is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or whitespace')
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the mechanism of action of metformin?",
                    "debug": False
                }
            ]
        }
    }


class EvidenceSnippet(BaseModel):
    """Evidence snippet from retrieval."""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    year: Optional[int] = Field(None, description="Publication year")
    title: str = Field(default="", description="Article title")
    journal: str = Field(default="", description="Journal name")
    text: str = Field(..., description="Evidence text")
    score: float = Field(..., description="Relevance score")
    hits: Dict[str, bool] = Field(
        default_factory=dict,
        description="Which retrievers matched (bm25, faiss)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "chunk_id": "pubmed:12345678:y2023:c0",
                    "pmid": "12345678",
                    "year": 2023,
                    "title": "Metformin and AMPK pathway",
                    "journal": "Journal of Clinical Endocrinology",
                    "text": "Metformin activates AMPK...",
                    "score": 0.85,
                    "hits": {"bm25": True, "faiss": True}
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response schema for /query endpoint."""
    
    trace_id: str = Field(..., description="Unique trace ID for request tracking")
    decision: str = Field(
        ...,
        description="Pipeline decision",
        pattern="^(ANSWER|ABSTAIN|REFUSE|ESCALATE|CHITCHAT|CLARIFY)$"
    )
    mode: Optional[str] = Field(
        None,
        description="Answer mode: rag|general|escalate|chitchat|refuse|clarify|abstain"
    )
    is_personal: Optional[bool] = Field(
        None,
        description="Whether the query described the user's own situation (50/50 handling)"
    )
    reason: Optional[str] = Field(
        None,
        description="Explanation for decision"
    )
    answer: Optional[str] = Field(
        None,
        description="Generated answer (only present if decision=ANSWER)"
    )
    snippets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence snippets supporting the answer"
    )
    
    # Debug information (only when debug=True)
    timings_ms: Optional[Dict[str, float]] = Field(
        None,
        description="Stage-by-stage timing breakdown"
    )
    signals: Optional[Dict[str, Any]] = Field(
        None,
        description="Guardrail signals and decisions"
    )
    llm_synthesized: Optional[bool] = Field(
        None,
        description="Whether response was synthesized by LLM (vs extractive)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trace_id": "abc123-def456",
                    "decision": "ANSWER",
                    "reason": None,
                    "answer": "CRP (C-Reactive Protein) is an inflammatory marker...",
                    "snippets": [
                        {
                            "chunk_id": "pubmed:12345678:y2023:c0",
                            "pmid": "12345678",
                            "year": 2023,
                            "title": "CRP as inflammatory marker",
                            "text": "CRP is synthesized by the liver...",
                            "score": 0.89
                        }
                    ],
                    "timings_ms": None,
                    "signals": None
                }
            ]
        }
    }


class HealthComponent(BaseModel):
    """Health status of a system component."""
    
    status: str = Field(..., pattern="^(healthy|unhealthy|degraded)$")
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    
    status: str = Field(
        ...,
        description="Overall system health",
        pattern="^(healthy|degraded|unhealthy)$"  # Changed from 'regex' to 'pattern'
    )
    timestamp: str = Field(..., description="Health check timestamp (ISO 8601)")
    components: Dict[str, Any] = Field(
        default_factory=dict,
        description="Health status of individual components"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2026-01-28T10:30:00Z",
                    "components": {
                        "retriever": {
                            "status": "healthy",
                            "indexes_root": "/app/kb/indexes",
                            "years": ["2023", "2024"]
                        },
                        "guardrails": {
                            "status": "healthy"
                        }
                    }
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    trace_id: str = Field(..., description="Trace ID for error tracking")
    error: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp (ISO 8601)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trace_id": "abc123-def456",
                    "error": "Query too long (max 500 characters)",
                    "timestamp": "2026-01-28T10:30:00Z"
                }
            ]
        }
    }
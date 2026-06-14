"""
retrieval/embedder.py

The single embedding seam for WiseWell. Everything that needs a vector goes
through embed() — query embedding for Pinecone, and (later) any re-embedding.

WHY a seam:
  - MiniLM (sentence-transformers + torch) is heavy to import and load. Loading
    it eagerly at module import would block FastAPI cold start. So the model is
    lazy-loaded on the first embed() call, behind this function.
  - Keeping the embedder behind one function means swapping MiniLM for a hosted
    embedder later (e.g. a Pinecone/Bedrock embedding endpoint) is a one-file
    change — callers never import SentenceTransformer directly.

Contract: embed(text) -> list[float] of length EMBEDDING_DIM, L2-normalized
(normalize_embeddings=True), matching the Pinecone index (384-dim, cosine).
"""

from __future__ import annotations

import threading
from typing import List, Optional

from config import EMBEDDING_MODEL, EMBEDDING_DIM

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load MiniLM on first use. import torch stays out of cold start."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Imported here, not at module top, so cold start never pays
                # the torch/transformers import cost unless embedding is needed.
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> List[float]:
    """
    Embed a single string into a normalized 384-dim vector (cosine-ready).
    Returns a plain Python list so it can be passed straight to Pinecone.
    """
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    out = vec.tolist()
    if len(out) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dim mismatch: got {len(out)}, expected {EMBEDDING_DIM}"
        )
    return out


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Batch variant — same contract, one model.encode call."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def is_loaded() -> bool:
    """True once the model has been lazy-loaded (for health/warmup checks)."""
    return _model is not None


def warmup() -> None:
    """Optionally force-load the model (e.g. a background warmup task)."""
    _get_model()

"""
retrieval package.

HybridRetriever (BM25 + FAISS) is the pre-Pinecone rollback path. It is NOT
imported eagerly anymore: doing so ran `import faiss` at cold start. It stays
lazily accessible (`from retrieval import HybridRetriever`) for rollback — the
faiss import only happens if the name is actually accessed.

The active retriever is retrieval.pinecone_retriever.PineconeRetriever.
"""

__all__ = ["HybridRetriever"]


def __getattr__(name: str):
    # PEP 562 lazy attribute: defer the heavy faiss import until rollback uses it.
    if name == "HybridRetriever":
        from retrieval.hybrid_retriever import HybridRetriever

        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

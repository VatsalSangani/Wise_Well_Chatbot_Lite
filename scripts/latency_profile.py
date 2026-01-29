"""
Latency profiling utilities for the WiseWell guardrails pipeline.
Provides timing instrumentation with minimal overhead.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


class LatencyTimer:
    """Context manager for measuring stage latency using perf_counter()."""
    
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time: Optional[float] = None
        self.elapsed_ms: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        return False


@contextmanager
def profile_stage(stage_name: str):
    """
    Context manager to profile a pipeline stage.
    Usage:
        with profile_stage("retrieve") as timer:
            results = retriever.retrieve(...)
        latency_ms = timer.elapsed_ms
    """
    timer = LatencyTimer(stage_name)
    with timer:
        yield timer


class PipelineProfiler:
    """Accumulates stage latencies for a single query execution."""
    
    STAGES = [
        "validate",
        "safety_intent",
        "specificity",
        "retrieve",
        "topic_consistency",
        "evidence_gate",
        "compose",
        "citation_verify",
        "total",
    ]
    
    def __init__(self):
        self.timings: Dict[str, float] = {stage: 0.0 for stage in self.STAGES}
        self._total_start: Optional[float] = None
    
    def start_total(self):
        """Mark the start of total pipeline execution."""
        self._total_start = time.perf_counter()
    
    def end_total(self):
        """Mark the end of total pipeline execution."""
        if self._total_start is not None:
            self.timings["total"] = (time.perf_counter() - self._total_start) * 1000.0
    
    def record_stage(self, stage_name: str, elapsed_ms: float):
        """Record the elapsed time for a stage."""
        if stage_name in self.timings:
            self.timings[stage_name] = elapsed_ms
    
    def get_timings(self) -> Dict[str, float]:
        """Return a dict of {stage_name: elapsed_ms}."""
        return dict(self.timings)
    
    def summary(self) -> str:
        """Return a human-readable summary of timings."""
        lines = []
        for stage in self.STAGES:
            ms = self.timings.get(stage, 0.0)
            lines.append(f"  {stage:20s}: {ms:8.2f} ms")
        return "\n".join(lines)


def get_query_stats(retrieved_chunks: Any) -> Dict[str, int]:
    """
    Extract statistics from retrieval results for logging.
    
    Args:
        retrieved_chunks: List of chunk dicts from retriever.retrieve()
    
    Returns:
        dict with keys: num_chunks, num_unique_pmids, num_years
    """
    if not retrieved_chunks:
        return {"num_chunks": 0, "num_unique_pmids": 0, "num_years": 0}
    
    pmids = set()
    years = set()
    for chunk in retrieved_chunks:
        if "pmid" in chunk:
            pmids.add(chunk["pmid"])
        if "year" in chunk:
            years.add(chunk["year"])
    
    return {
        "num_chunks": len(retrieved_chunks),
        "num_unique_pmids": len(pmids),
        "num_years": len(years),
    }


if __name__ == "__main__":
    # Example usage
    profiler = PipelineProfiler()
    
    profiler.start_total()
    
    # Simulate stages
    for i, stage in enumerate(PipelineProfiler.STAGES[:-1]):
        profiler.record_stage(stage, 10.5 * (i + 1))
    
    profiler.end_total()
    
    print("Latency Profile:")
    print(profiler.summary())

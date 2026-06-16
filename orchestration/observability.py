"""
WiseWell observability — Langfuse v4 tracing + off-path RAG evaluation.

DESIGN CONTRACT: this module is PURE OBSERVATION. It must NEVER change request
behavior and NEVER add a failure mode. Every Langfuse/OpenAI call is wrapped so
that if the backend is unreachable, keys are missing, or the SDK errors, the
request still succeeds. When Langfuse is not configured, all helpers no-op and
`observe` becomes a pass-through decorator.

v4 SDK (verified from live docs, March 2026 rewrite):
  from langfuse import get_client, observe
  langfuse.update_current_trace(input=, output=, metadata=, tags=)
  with langfuse.start_as_current_observation(as_type="generation"|"span", ...) as o: o.update(...)
  usage_details uses {"input": n, "output": n}  (NOT input_tokens)
  langfuse.create_score(trace_id=, name=, value=, data_type="NUMERIC", comment=)
"""

from __future__ import annotations

import json
import os
import re
import random
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

# ── Client bootstrap (safe) ────────────────────────────────────
_LF_ENABLED = False
_client = None
_lf_observe = None

try:
    from langfuse import get_client, observe as _lf_observe  # type: ignore

    # Bridge env-var name: this project's .env uses LANGFUSE_BASE_URL, the v4 SDK
    # reads LANGFUSE_HOST. Normalize so the configured host is honored.
    if not os.getenv("LANGFUSE_HOST") and os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")

    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        _client = get_client()
        _LF_ENABLED = True
        logger.info("langfuse_enabled", host=os.getenv("LANGFUSE_HOST", "default"))
    else:
        logger.info("langfuse_disabled_no_keys")
except Exception as e:  # SDK missing / init error → stay disabled, never crash
    logger.warning("langfuse_unavailable", error=str(e), error_type=type(e).__name__)


def is_enabled() -> bool:
    return _LF_ENABLED


# ── @observe (safe) ────────────────────────────────────────────
def observe(*args, **kwargs):
    """Safe @observe. Delegates to the real v4 decorator when enabled; otherwise
    a transparent pass-through. Supports both @observe and @observe(name=...)."""
    if _LF_ENABLED and _lf_observe is not None:
        try:
            return _lf_observe(*args, **kwargs)
        except Exception:
            pass
    # Pass-through. @observe (bare) -> args=(fn,); @observe(...) -> return decorator.
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _decorator(fn):
        return fn

    return _decorator


# ── No-op observation (used when disabled or on error) ─────────
class _NoopObs:
    def update(self, *a, **k):
        return None


@contextmanager
def generation(name: str, model: Optional[str] = None, input: Any = None):
    """Context manager for an LLM generation observation. Yields an object with
    .update(output=, usage_details=). No-ops safely if disabled/erroring."""
    if not _LF_ENABLED:
        yield _NoopObs()
        return
    try:
        with _client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input
        ) as gen:
            yield _SafeObs(gen)
    except Exception as e:
        logger.warning("langfuse_generation_failed", error=str(e))
        yield _NoopObs()


@contextmanager
def span(name: str, input: Any = None):
    """Context manager for a span observation. Yields an object with .update()."""
    if not _LF_ENABLED:
        yield _NoopObs()
        return
    try:
        with _client.start_as_current_observation(
            as_type="span", name=name, input=input
        ) as sp:
            yield _SafeObs(sp)
    except Exception as e:
        logger.warning("langfuse_span_failed", error=str(e))
        yield _NoopObs()


class _SafeObs:
    """Wraps a real Langfuse observation so .update() never raises."""

    def __init__(self, obs):
        self._obs = obs

    def update(self, **kwargs):
        try:
            self._obs.update(**kwargs)
        except Exception:
            pass


def update_trace(input=None, output=None, metadata=None, tags=None) -> None:
    """Set trace-level fields using the real v4.7.1 API:
      - input/output -> set_current_trace_io
      - metadata     -> update_current_span (root observation metadata; queryable)
      - tags         -> folded into metadata (v4.7.1 has no public runtime tags setter)
    NOTE: v4.7.1 has NO update_current_trace method (the older docs are wrong for
    this version) — calling it silently no-oped and left trace I/O null."""
    if not _LF_ENABLED:
        return
    try:
        if input is not None or output is not None:
            _client.set_current_trace_io(input=input, output=output)
    except Exception as e:
        logger.warning("langfuse_trace_io_failed", error=str(e))
    try:
        md = dict(metadata or {})
        if tags:
            md["tags"] = tags
        if md:
            _client.update_current_span(metadata=md)
    except Exception as e:
        logger.warning("langfuse_trace_metadata_failed", error=str(e))


def current_trace_id() -> Optional[str]:
    if not _LF_ENABLED:
        return None
    try:
        return _client.get_current_trace_id()
    except Exception:
        return None


def create_score(trace_id: str, name: str, value: float, comment: Optional[str] = None) -> None:
    if not _LF_ENABLED or not trace_id:
        return
    try:
        _client.create_score(
            trace_id=trace_id, name=name, value=value, data_type="NUMERIC", comment=comment
        )
    except Exception as e:
        logger.warning("langfuse_score_failed", name=name, error=str(e))


def flush() -> None:
    if not _LF_ENABLED:
        return
    try:
        _client.flush()
    except Exception:
        pass


# ── Defensive JSON parse (reused for judge output) ─────────────
def _safe_json(text: str) -> Optional[dict]:
    """Parse JSON that may be wrapped in fences or have preamble. Returns None
    on total failure (caller skips that score rather than crashing)."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r"\n?```(?:json)?\n?", "\n", text).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ── RAGAS-style judge prompts (GPT-4o-mini) ────────────────────
_FAITHFULNESS_PROMPT = (
    "You are evaluating whether an answer is faithful to provided source contexts. "
    "Given CONTEXTS and ANSWER, score 0.0-1.0 how well every factual claim in the answer "
    "is supported by the contexts (1.0 = all claims supported, 0.0 = unsupported/contradicted). "
    "Penalize claims not present in the contexts.\n\n"
    "CONTEXTS:\n{contexts}\n\nANSWER:\n{answer}\n\n"
    'Return ONLY JSON: {{"score": float, "reason": str}}'
)

_ANSWER_RELEVANCY_PROMPT = (
    "You are evaluating whether an answer addresses the user's question. "
    "Given QUESTION and ANSWER, score 0.0-1.0 how directly and completely the answer responds "
    "(1.0 = fully on-point, 0.0 = off-topic/evasive).\n\n"
    "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
    'Return ONLY JSON: {{"score": float, "reason": str}}'
)

_CONTEXT_RELEVANCY_PROMPT = (
    "You are evaluating whether retrieved contexts are relevant to the question. "
    "Given QUESTION and CONTEXTS, score 0.0-1.0 the proportion of contexts useful for answering "
    "(1.0 = all relevant, 0.0 = none).\n\n"
    "QUESTION:\n{question}\n\nCONTEXTS:\n{contexts}\n\n"
    'Return ONLY JSON: {{"score": float, "reason": str}}'
)


def _judge(openai_client, model: str, metric: str, prompt: str) -> Optional[Dict[str, Any]]:
    """Run one judge call. Returns {score, reason} or None (logged) on any failure."""
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("judge_call_failed", metric=metric, error=str(e))
        return None

    logger.debug("judge_raw_response", metric=metric, raw_preview=txt[:160])

    data = _safe_json(txt)
    if not data or "score" not in data:
        logger.warning("judge_parse_failed", metric=metric, preview=txt[:120])
        return None
    try:
        data["score"] = float(data["score"])
    except (TypeError, ValueError):
        logger.warning("judge_score_not_numeric", metric=metric, value=data.get("score"))
        return None
    return data


def _flesch_reading_ease(text: str) -> Optional[float]:
    """Flesch reading-ease score (non-LLM). Higher = easier to read. None on failure."""
    if not text or not text.strip():
        return None
    try:
        import textstat

        return float(textstat.flesch_reading_ease(text))
    except Exception as e:
        logger.warning("readability_failed", error=str(e))
        return None


def run_rag_eval(
    trace_id: str,
    question: str,
    answer: str,
    contexts: List[str],
    mode: str,
) -> None:
    """
    Off-path RAG evaluation (called from a FastAPI BackgroundTask, AFTER the
    response is sent). Scores attach to the existing Langfuse trace by id.
    NEVER raises — a failed eval just produces no score.

      rag     -> faithfulness + answer_relevancy + context_relevancy
      general -> answer_relevancy only (no contexts)
    """
    if not _LF_ENABLED or not trace_id:
        return
    try:
        from openai import OpenAI

        from config import JUDGE_MODEL

        client = OpenAI()  # reads OPENAI_API_KEY from env
        ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) or "(none)"

        logger.debug(
            "rag_eval_inputs",
            mode=mode,
            question_chars=len(question or ""),
            answer_chars=len(answer or ""),
            contexts_count=len(contexts),
            ctx_chars=len(ctx_text),
        )

        if mode == "rag":
            metrics = [
                ("faithfulness", _FAITHFULNESS_PROMPT.format(contexts=ctx_text, answer=answer)),
                ("answer_relevancy", _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)),
                ("context_relevancy", _CONTEXT_RELEVANCY_PROMPT.format(question=question, contexts=ctx_text)),
            ]
        elif mode == "general":
            metrics = [
                ("answer_relevancy", _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)),
            ]
        else:
            return  # safety paths are deterministic — no eval

        scored = 0
        faithfulness_value = None
        for name, prompt in metrics:
            result = _judge(client, JUDGE_MODEL, name, prompt)
            if result is not None:
                create_score(trace_id, name, result["score"], comment=str(result.get("reason", ""))[:500])
                scored += 1
                if name == "faithfulness":
                    faithfulness_value = result["score"]

        # hallucination_rate: derived from faithfulness (1 - faithfulness), nearly
        # free, no extra judge call. Strong medical-bot signal. RAG only (general
        # mode has no contexts, so faithfulness/hallucination are undefined).
        if faithfulness_value is not None:
            create_score(
                trace_id, "hallucination_rate", 1.0 - faithfulness_value,
                comment="Derived: 1 - faithfulness",
            )
            scored += 1

        # readability: Flesch reading-ease on the answer (non-LLM, no judge call).
        # Relevant to the plain-language goal. Computed for both rag and general.
        readability = _flesch_reading_ease(answer)
        if readability is not None:
            create_score(
                trace_id, "readability", readability,
                comment="Flesch reading-ease (0-100; higher = easier)",
            )
            scored += 1

        flush()  # ensure scores are sent from the background task
        logger.info("rag_eval_complete", trace_id=trace_id, mode=mode, scored=scored)
    except Exception as e:
        # Eval must NEVER break — request already succeeded.
        logger.warning("rag_eval_failed", trace_id=trace_id, error=str(e), error_type=type(e).__name__)


def should_eval(sample_rate: float) -> bool:
    """Sampling gate for eval. Always False when Langfuse disabled."""
    if not _LF_ENABLED:
        return False
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    return random.random() < sample_rate

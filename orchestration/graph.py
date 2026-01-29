from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END  # type: ignore

from .state import QAState
from .nodes import (
    node_error_wrapper,
    node_validate,
    node_input_validation,
    node_safety_intent,
    node_specificity,
    node_retrieve,
    node_topic_consistency,
    node_evidence_gate,
    node_compose,
    node_citation_verify,
    node_finalize,
)


def build_graph(retriever: Any):
    g = StateGraph(QAState)

    # Nodes
    g.add_node("validate", lambda s: node_error_wrapper(node_validate, s, "validate"))
    g.add_node("input_validation", lambda s: node_error_wrapper(node_input_validation, s, "input_validation"))
    g.add_node("safety_intent", lambda s: node_error_wrapper(node_safety_intent, s, "safety_intent"))
    g.add_node("specificity", lambda s: node_error_wrapper(node_specificity, s, "specificity"))

    g.add_node("retrieve", lambda s: node_error_wrapper(node_retrieve, s, "retrieve", retriever))
    g.add_node("topic_consistency", lambda s: node_error_wrapper(node_topic_consistency, s, "topic_consistency"))
    g.add_node("evidence_gate", lambda s: node_error_wrapper(node_evidence_gate, s, "evidence_gate"))
    g.add_node("compose", lambda s: node_error_wrapper(node_compose, s, "compose"))
    g.add_node("citation_verify", lambda s: node_error_wrapper(node_citation_verify, s, "citation_verify"))
    g.add_node("finalize", lambda s: node_error_wrapper(node_finalize, s, "finalize"))

    # Entry
    g.set_entry_point("validate")

    # Linear backbone
    g.add_edge("validate", "input_validation")

    # Branch: invalid input -> finalize
    def route_after_input(s: QAState) -> str:
        if s.error:
            return "finalize"
        if s.input_ok is False:
            return "finalize"
        return "safety_intent"

    g.add_conditional_edges("input_validation", route_after_input, {
        "safety_intent": "safety_intent",
        "finalize": "finalize",
    })

    # Branch: refuse -> finalize
    def route_after_safety(s: QAState) -> str:
        if s.error:
            return "finalize"
        if s.intent == "refuse":
            return "finalize"
        return "specificity"

    g.add_conditional_edges("safety_intent", route_after_safety, {
        "specificity": "specificity",
        "finalize": "finalize",
    })

    # Branch: underspecified -> finalize
    def route_after_specificity(s: QAState) -> str:
        if s.error:
            return "finalize"
        if s.specificity_ok is False:
            return "finalize"
        return "retrieve"

    g.add_conditional_edges("specificity", route_after_specificity, {
        "retrieve": "retrieve",
        "finalize": "finalize",
    })

    g.add_edge("retrieve", "topic_consistency")

    # Branch: topic insufficient -> finalize
    def route_after_topic(s: QAState) -> str:
        if s.error:
            return "finalize"
        if s.topic_ok is False:
            return "finalize"
        return "evidence_gate"

    g.add_conditional_edges("topic_consistency", route_after_topic, {
        "evidence_gate": "evidence_gate",
        "finalize": "finalize",
    })

    # Branch: evidence fail -> finalize
    def route_after_evidence(s: QAState) -> str:
        if s.error:
            return "finalize"
        if s.evidence_ok is False:
            return "finalize"
        return "compose"

    g.add_conditional_edges("evidence_gate", route_after_evidence, {
        "compose": "compose",
        "finalize": "finalize",
    })

    g.add_edge("compose", "citation_verify")
    g.add_edge("citation_verify", "finalize")
    g.add_edge("finalize", END)

    return g.compile()

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

_MECH_QUERY_HINTS = (
    "mechanism", "mechanism of action", "moa", "pathway", "signaling",
    "mediated", "target", "how does", "via"
)

_MECH_MARKERS = (
    "ampk", "mtor", "nf-kb", "jak", "stat", "il-", "tnf", "tgf", "ifn",
    "cytokine", "receptor", "signaling", "pathway", "inhibit", "activation",
    "upregulate", "downregulate", "phosphorylation", "oxidative", "mitochond",
    "gluconeogenesis", "gut microbiota", "microbiome"
)

def is_mechanism_query(q: str) -> bool:
    ql = (q or "").lower()
    return any(h in ql for h in _MECH_QUERY_HINTS)

def mechanism_gate(query: str, results: List[Dict[str, Any]], *, min_mech_snippets: int = 2) -> Tuple[bool, Dict[str, Any]]:
    if not is_mechanism_query(query):
        return True, {"applied": False, "reason": "not_mechanism_query"}

    mech_hits = 0
    hit_pmids = []
    for r in results or []:
        blob = (str(r.get("title","")) + " " + str(r.get("text",""))).lower()
        if any(m in blob for m in _MECH_MARKERS):
            mech_hits += 1
            hit_pmids.append(r.get("pmid"))

    ok = mech_hits >= min_mech_snippets
    return ok, {"applied": True, "min_mech_snippets": min_mech_snippets, "mech_hits": mech_hits, "hit_pmids": hit_pmids[:10]}

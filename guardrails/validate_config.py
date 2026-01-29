# ==================================
# 3) scripts/guardrails/validate_config.py  (UPDATE)
# ==================================
# Replace your existing validate_config.py with this full version (it includes new keys but keeps old ones)

from __future__ import annotations

from guardrails.config_loader import load_guardrails_config


# Lists that must be present and be list type
REQUIRED_LIST_KEYS = [
    "stopwords",
    "generic_tokens",
    "injection_patterns",
    "refuse_patterns",
    "broad_question_patterns",
    "evidence_anchors",
    "underspecified_patterns",
    "unit_patterns",
    "drug_suffix_patterns",
    "biomarker_tokens",
    "unit_noise_tokens",
    "verb_noise_tokens",
]

# Dicts that must be present and be dict type
REQUIRED_DICT_KEYS = [
    "anchor_synonyms",
    "synonyms",
]

REQUIRED_SCALAR_KEYS = [
    "min_query_chars",
    "max_query_chars",
]


def validate() -> None:
    cfg = load_guardrails_config()

    # Check for missing list keys
    missing_lists = [k for k in REQUIRED_LIST_KEYS if k not in cfg]
    # Check for missing dict keys
    missing_dicts = [k for k in REQUIRED_DICT_KEYS if k not in cfg]
    # Check for missing scalar keys
    missing_scalars = [k for k in REQUIRED_SCALAR_KEYS if k not in cfg]

    all_missing = missing_lists + missing_dicts + missing_scalars
    if all_missing:
        raise ValueError(f"guardrails.yaml missing keys: {all_missing}")

    # Validate list types
    for k in REQUIRED_LIST_KEYS:
        if not isinstance(cfg[k], list):
            raise TypeError(f"{k} must be a list, got {type(cfg[k]).__name__}")

    # Validate dict types
    for k in REQUIRED_DICT_KEYS:
        if not isinstance(cfg[k], dict):
            raise TypeError(f"{k} must be a dict, got {type(cfg[k]).__name__}")

    # Validate scalar types
    for k in REQUIRED_SCALAR_KEYS:
        if not isinstance(cfg[k], int):
            raise TypeError(f"{k} must be an int")

    # Optional scalar field
    if "specificity_min_score" in cfg and not isinstance(cfg["specificity_min_score"], int):
        raise TypeError("specificity_min_score must be an int")

    print("guardrails.yaml OK")


if __name__ == "__main__":
    validate()

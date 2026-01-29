from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG: Dict[str, Any] | None = None


def load_guardrails_config(path: str = "../config/guardrails.yaml") -> Dict[str, Any]:
    """
    Loads config/guardrails.yaml once and caches it.

    Path is resolved relative to this file:
      guardrails/config_loader.py -> project_root/config/guardrails.yaml
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    base_dir = Path(__file__).resolve().parent  # guardrails/
    cfg_path = (base_dir / path).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing guardrails config: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f) or {}

    # Ensure new Topic Consistency v2 and Safety Intent v2 keys exist with safe defaults
    if "biomarker_tokens" not in _CONFIG:
        _CONFIG["biomarker_tokens"] = [
            "crp", "hba1c", "tsh", "ldl", "hdl", "esr", "alt", "ast",
            "ferritin", "troponin", "il-6", "il6", "tnf", "tnf-alpha"
        ]
    if "unit_noise_tokens" not in _CONFIG:
        _CONFIG["unit_noise_tokens"] = [
            "l", "dl", "ul", "iu", "kg", "lb", "mm", "cm", "mg", "g",
            "mmol", "mol", "pmol", "ng", "ug", "%", "per", "day"
        ]
    if "verb_noise_tokens" not in _CONFIG:
        _CONFIG["verb_noise_tokens"] = [
            "indicate", "indicates", "meaning", "mean", "cause", "causes",
            "treat", "treating", "help", "helps", "do", "does",
            "affect", "suggest", "suggests"
        ]
    if "anchor_synonyms" not in _CONFIG:
        _CONFIG["anchor_synonyms"] = {
            "il6": "il-6",
            "hb a1c": "hba1c",
            "hba1c": "hba1c",
            "r.a.": "ra",
            "ra": "ra",
        }

    return _CONFIG

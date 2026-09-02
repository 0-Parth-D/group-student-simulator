import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

LP_PISA_TO_LEGACY = {
    "mathematical_operation": "Operation",
    "mathematical_abstraction": "Abstraction",
    "logical_reasoning": "Logical_Reasoning",
    "mathematical_modeling": "Modeling",
    "intuitive_imagination": "Intuitive_Imagination",
    "data_analysis": "Data_Analysis",
}

MASTERY_LABELS = {
    0: "missing",
    1: "bad",
    2: "partial",
    3: "good",
}


@lru_cache(maxsize=1)
def load_knowledge_graph() -> dict:
    path = CONFIG_DIR / "knowledge_graph.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_misconception_catalog() -> dict:
    with open(CONFIG_DIR / "misconception_catalog.json", encoding="utf-8") as f:
        return json.load(f)


def list_construct_ids() -> List[str]:
    return list(load_knowledge_graph().get("constructs", {}).keys())


def get_construct_def(construct_id: str) -> Optional[dict]:
    return load_knowledge_graph().get("constructs", {}).get(construct_id)


def get_prerequisite_edges() -> List[Tuple[str, str, str]]:
    edges = []
    for row in load_knowledge_graph().get("prerequisites", []):
        if len(row) >= 2:
            rationale = row[2] if len(row) > 2 else ""
            edges.append((row[0], row[1], rationale))
    return edges


def get_prerequisites_of(construct_id: str) -> List[str]:
    return [src for src, dst, _ in get_prerequisite_edges() if dst == construct_id]


def lp_pisa_to_legacy_profile(baseline: Dict[str, int]) -> Dict[str, str]:
    """Map numeric 0-3 LP PISA baseline to low/med/high for legacy prompts."""
    out: Dict[str, str] = {}
    for lp_id, score in baseline.items():
        legacy_id = LP_PISA_TO_LEGACY.get(lp_id, lp_id)
        if score <= 1:
            out[legacy_id] = "low"
        elif score == 2:
            out[legacy_id] = "med"
        else:
            out[legacy_id] = "high"
    return out

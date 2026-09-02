import json
from pathlib import Path
from typing import Dict, List, Optional

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

_attrs_cache: Optional[dict] = None

ATTRIBUTE_IDS = [
    "Abstraction",
    "Logical_Reasoning",
    "Modeling",
    "Intuitive_Imagination",
    "Operation",
    "Data_Analysis",
]


def load_pisa_attributes() -> dict:
    global _attrs_cache
    if _attrs_cache is None:
        with open(CONFIG_DIR / "pisa_attributes.json", encoding="utf-8") as f:
            _attrs_cache = json.load(f)
    return _attrs_cache


def get_attribute(attr_id: str) -> Optional[dict]:
    for a in load_pisa_attributes()["attributes"]:
        if a["id"] == attr_id:
            return a
    return None


def weak_attributes(profile: Dict[str, str], required: List[str]) -> List[str]:
    """Return required attributes where the student is low."""
    return [a for a in required if profile.get(a, "med") == "low"]


def attribute_error_templates(profile: Dict[str, str], required: List[str]) -> List[str]:
    """Collect error templates from weak required attributes."""
    templates = []
    for attr_id in weak_attributes(profile, required):
        attr = get_attribute(attr_id)
        if attr:
            templates.extend(attr.get("weak_error_templates", []))
    return templates


def pisa_summary(profile: Dict[str, str]) -> List[dict]:
    return [
        {
            "id": aid,
            "name": get_attribute(aid)["name"] if get_attribute(aid) else aid,
            "level": profile.get(aid, "med"),
        }
        for aid in ATTRIBUTE_IDS
    ]

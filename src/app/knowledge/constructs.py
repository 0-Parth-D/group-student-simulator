import json
from typing import Dict, List, Optional

from app.config import CONFIG_DIR

_constructs_cache: Optional[dict] = None
_tasks_cache: Optional[dict] = None


def _load_json(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_constructs() -> dict:
    global _constructs_cache
    if _constructs_cache is None:
        _constructs_cache = _load_json("rational_number_constructs.json")
    return _constructs_cache


def load_task_metadata() -> dict:
    global _tasks_cache
    if _tasks_cache is None:
        _tasks_cache = _load_json("task_metadata.json")
    return _tasks_cache


def get_construct(construct_id: str) -> Optional[dict]:
    for c in load_constructs()["constructs"]:
        if c["id"] == construct_id:
            return c
    return None


def get_level_info(construct_id: str, level: int) -> Optional[dict]:
    construct = get_construct(construct_id)
    if not construct:
        return None
    for lv in construct["levels"]:
        if lv["level"] == level:
            return lv
    return None


def lookup_task_metadata(task_text: str, task_id: str = "") -> Optional[dict]:
    """Match task by task_id first, then LP description overlap or legacy keywords."""
    tasks = load_task_metadata()["tasks"]
    if task_id:
        for t in tasks:
            if t["task_id"] == task_id:
                return t

    text_lower = task_text.lower().strip()
    if not text_lower:
        return None

    for t in tasks:
        desc = (t.get("description") or "").lower()
        if desc and (desc in text_lower or text_lower in desc):
            return t

    best = None
    best_score = 0
    for t in tasks:
        keywords = t.get("match_keywords", [])
        if not keywords and t.get("description"):
            keywords = t["description"].lower().split()[:6]
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > best_score:
            best_score = score
            best = t
    return best if best_score > 0 else None


def construct_summary(construct_levels: Dict[str, int]) -> List[dict]:
    from app.knowledge.kg_config import MASTERY_LABELS, get_construct_def

    items = []
    for cid, level in construct_levels.items():
        cdef = get_construct_def(cid)
        if cdef:
            items.append(
                {
                    "construct_id": cid,
                    "name": cdef.get("label", cid),
                    "level": level,
                    "label": MASTERY_LABELS.get(level, "unknown"),
                }
            )
            continue
        construct = get_construct(cid)
        if not construct:
            continue
        lv = get_level_info(cid, level)
        items.append(
            {
                "construct_id": cid,
                "name": construct["name"],
                "level": level,
                "label": lv["label"] if lv else "unknown",
            }
        )
    return items

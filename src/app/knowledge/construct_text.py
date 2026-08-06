"""Construct-level prose read from the knowledge graph.

Both the eval yardstick (`app.student.expected_behavior`) and semantic scaffold detection
(`app.knowledge.scaffold_detector`) source their text here, so adding a scenario needs a task
entry and its constructs — not hand-written strings per task.
"""

import re
from functools import lru_cache
from typing import Dict, List, Tuple

from app.knowledge.kg_config import MASTERY_LABELS, get_construct_def, list_construct_ids

# DLM publishes five linkage levels against a 0-3 mastery scale where 3 is the
# construct's target_stack_level. "T" (Target) therefore anchors level 3, and
# "S" (Successor) describes above-target work that the scale does not reach.
DLM_LEVEL_BY_MASTERY = {0: "IP", 1: "DP", 2: "PP", 3: "T"}

_PI_CODE = re.compile(r"^M\.[A-Z]+\.\d+[a-z]?\s+")
_CITATION = re.compile(r"\([^)]*\d[^)]*\)")


def _strip_citations(text: str) -> str:
    """Drop the LPF indicator code and CCSS parentheticals, keeping the prose."""
    text = _PI_CODE.sub("", text)
    text = _CITATION.sub("", text)
    return " ".join(text.split())


@lru_cache(maxsize=128)
def construct_label(construct_id: str) -> str:
    cdef = get_construct_def(construct_id) or {}
    return str(cdef.get("label") or construct_id.replace("_", " "))


@lru_cache(maxsize=512)
def mastery_descriptor(construct_id: str, level: int) -> str:
    """What a student at `level` (0-3) does on this construct, in plain prose.

    Prefers `lpf_stack`, falls back to `dlm_linkage_levels`, then to the label.
    """
    level = max(0, min(3, int(level)))
    cdef = get_construct_def(construct_id) or {}

    stack = cdef.get("lpf_stack") or {}
    text = stack.get(MASTERY_LABELS.get(level, ""))
    if text:
        return str(text).strip()

    dlm = cdef.get("dlm_linkage_levels") or {}
    text = dlm.get(DLM_LEVEL_BY_MASTERY.get(level, ""))
    if text:
        return str(text).strip()

    label = construct_label(construct_id)
    if level == 0:
        return f"No usable approach to {label} yet"
    return f"{MASTERY_LABELS.get(level, 'partial').capitalize()} grasp of {label}"


@lru_cache(maxsize=128)
def construct_cues(construct_id: str) -> Tuple[str, ...]:
    """Surface phrases that signal this construct in teacher talk.

    Single source for scaffold detection and LP construct routing, which previously
    kept separate hand-maintained keyword dicts that drifted apart. Novel wording is
    the embedding fallback's job, not a reason to grow these lists.
    """
    cdef = get_construct_def(construct_id) or {}
    return tuple(
        str(c).lower().strip() for c in (cdef.get("cues") or []) if str(c).strip()
    )


@lru_cache(maxsize=1)
def all_construct_cues() -> Dict[str, Tuple[str, ...]]:
    return {cid: construct_cues(cid) for cid in list_construct_ids()}


@lru_cache(maxsize=1)
def math_cue_phrases() -> Tuple[str, ...]:
    """Every construct cue, for deciding whether a message is about the math at all."""
    seen: List[str] = []
    for cues in all_construct_cues().values():
        for cue in cues:
            if cue not in seen:
                seen.append(cue)
    return tuple(seen)


@lru_cache(maxsize=128)
def construct_vocabulary(construct_id: str) -> str:
    """Label plus LPF progress-indicator prose, for embedding-based matching."""
    cdef = get_construct_def(construct_id) or {}
    parts: List[str] = [construct_label(construct_id)]

    for indicator in cdef.get("lpf_progress_indicators") or []:
        cleaned = _strip_citations(str(indicator))
        if cleaned:
            parts.append(cleaned)

    stack = cdef.get("lpf_stack") or {}
    for key in ("partial", "good"):
        if stack.get(key):
            parts.append(str(stack[key]))

    dlm = cdef.get("dlm_linkage_levels") or {}
    for key in ("PP", "T"):
        if dlm.get(key):
            parts.append(str(dlm[key]))

    return " ".join(parts)
